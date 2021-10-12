import os, os.path, sys, glob
import pandas
import datetime
import re
import argparse
import logging
import pytz
import tzwhere.tzwhere as tzwhere
import difflib
import numpy as np
import lime
import lime.lime_tabular
import sklearn.preprocessing
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.append('../code-air-forecast')

import forecast.convertfmt
import forecast.loadcsv
import forecast.fileio
import forecast.modelsk
import forecast.modelsnn
import forecast.modelsln
import forecast.modelfit
import forecast.evaluations

parser = argparse.ArgumentParser(description='run air-forecast for all exp in the data path')

parser.add_argument('--wrf_path', type=str, required=False, default='./data/WRF', help='WRF data path')
parser.add_argument('--wrf_index', type=str, required=False, default='date', help='WRF data time index')
parser.add_argument('--wrf_indexfmt', type=str, required=False, default="%Y-%m-%d %H:%M:%S", help='WRF data time index format')

parser.add_argument('--sites', type=str, required=False, default='./data/sites.csv', help='site info')
parser.add_argument('--gmtoffset', type=str, required=False, default='./data/sites-gmtoffset.csv', help='site info with gmt-offset')
parser.add_argument('--siteid', type=str, required=False, default='AQS_ID', help='site ID')
parser.add_argument('--sitename', type=str, required=False, default='Sitename', help='site name')
parser.add_argument('--sitelati', type=str, required=False, default='Latitude', help='site latitude')
parser.add_argument('--sitelong', type=str, required=False, default='Longitude', help='site longitude')

parser.add_argument('--aqs_path', type=str, required=False, default='./data/AQS', help='AQS data path')
parser.add_argument('--aqs_o3subfolder', type=str, required=False, default='o3', help='AQS Ozone subfolder')
parser.add_argument('--aqs_pm25subfolder', type=str, required=False, default='pm', help='AQS PM2.5 subfolder')
parser.add_argument('--aqs_index', type=str, required=False, default='Datetime', help='AQS data time index')
parser.add_argument('--aqs_indexfmt', type=str, required=False,default="%Y-%m-%d %H:%M", help='AQS data time index format')
parser.add_argument('--aqs_target', type=str, required=False,default="sample_measurement", help='AQS data measurement column name')

parser.add_argument('--cmaq_path', type=str, required=False, default='./data/AQS/cmaq.h5', help='CMAQ data path')
parser.add_argument('--cmaq_o3header', type=str, required=False,default='OZONE_cmaq', help='AQS Ozone subfolder')
parser.add_argument('--cmaq_pm25header', type=str, required=False,default='PM2.5_cmaq', help='AQS PM2.5 subfolder')
parser.add_argument('--cmaq_index', type=str, required=False,default='DateTime', help='AQS data time index')
parser.add_argument('--cmaq_indexfmt', type=str, required=False,default="%Y-%m-%d %H:%M", help='AQS data time index format')
parser.add_argument('--cmaq_target', type=str, required=False, default="sample_measurement", help='AQS data measurement column name')
parser.add_argument('--cmaq_layer', type=int, required=False,default=3, help='AQS data measurement column name')

parser.add_argument('--out_path', type=str, required=False, default='./runoutput', help='output data path')
parser.add_argument('--wrffilepattern', type=str, required=False, default='wrf{}.csv', help='data file name pattern for WRF')
parser.add_argument('--o3filepattern', type=str, required=False, default='aqso3{}.csv', help='data file name pattern for AQS o3')
parser.add_argument('--pm25filepattern', type=str, required=False, default='aqspm{}.csv', help='data file name pattern for AQS pm2.5')
parser.add_argument('--inputfilepattern', type=str, required=False, default='input-{}-{}.csv', help='data file name pattern for model input')
parser.add_argument('--cmaqo3pattern', type=str, required=False, default="cmaqo3{}.csv", help='AQS data measurement column name')
parser.add_argument('--cmaqpm25pattern', type=str, required=False, default="cmaqpm{}.csv", help='AQS data measurement column name')

parser.add_argument('--flag2rf', type=int, required=False, default=1, help='flag for running 2rf model')
parser.add_argument('--flagautoml', type=int, required=False, default=0, help='flag for running nas model')
parser.add_argument('--flagdensexl', type=int, required=False, default=1, help='flag for running dense model')
parser.add_argument('--flagdensel', type=int, required=False, default=1, help='flag for running dense model')
parser.add_argument('--flagdensem', type=int, required=False, default=1, help='flag for running dense model')
parser.add_argument('--flagdenses', type=int, required=False, default=1, help='flag for running dense model')
parser.add_argument('--flagdensexs', type=int, required=False, default=1, help='flag for running dense model')
parser.add_argument('--logpath',  type=str, required=False, default='./log/compare.log', help='log file path')

parser.add_argument('--plotpath',  type=str, required=False, default='./plot', help='plot path')
parser.add_argument('--pipeloadwrf',  type=int, required=False, default=0, help='pipeline load wrf data')
parser.add_argument('--pipeloadaqs',  type=int, required=False, default=0, help='pipeline load aqs data')
parser.add_argument('--pipeloadcmaq',  type=int, required=False, default=0, help='pipeline load cmaq data')
parser.add_argument('--pipeml',  type=int, required=False, default=0, help='pipeline machine learning')

parser.add_argument('--pipeploteval',  type=int, required=False, default=1, help='pipeline plot prediction evaluation')
parser.add_argument('--pipeplotimportance',  type=int, required=False, default=1, help='pipeline plot feature importance')


class Predictor(object):
    def __init__(self, model, predict_func):
        self.model = model
        self.predict_func = predict_func

    def predict(self, x_test):
        return forecast.modelfit.predict(self.model, x_test, predict_func=self.predict_func)



def load_sites(sites_fn, site_id_col):
    sites = forecast.fileio.load_csv(sites_fn, index_col=site_id_col)
    sites.index = sites.index.astype(str)
    return sites


def load_timezone_from_geolocation(sites, idcol, laticol, longcol):
    tz = tzwhere.tzwhere(forceTZ=True)

    timezone_ls = []
    gmtoffset_ls = []

    for index, row in sites.iterrows():
        la = row[laticol]
        lo = row[longcol]

        timezone_str = tz.tzNameAt(la, lo, forceTZ=True)
        timezone = pytz.timezone(timezone_str)
        offset = timezone.utcoffset(datetime.datetime(2017, 1, 1, 0, 0, 0))
        offset_hours = int(offset.total_seconds() / (60 * 60))
        timezone_ls.append(timezone_str)
        gmtoffset_ls.append(offset_hours)

    sites['timezone'] = timezone_ls
    sites['gmtoffset'] = gmtoffset_ls
    return sites


def rename_wrf(df):
    cols = list(df)
    header = cols[0]
    for c in cols[1:]:
        match  = difflib.SequenceMatcher(None, header, c).find_longest_match(0, len(header), 0, len(c))
        header = c[match.b: match.b + match.size]
    logging.debug(header)
    colmap = {}
    for c in cols:
        colmap[c] = c.replace(header, '')
    df = df.rename(columns=colmap)
    return df


def load_csv_data_from_path(sites, in_path, in_index, in_indexfmt, out_path, savepattern, replace=True):
    csv_fls = forecast.fileio.get_files(in_path, pattern='*.csv')
    file_dict = {}
    for index, row in sites.iterrows():
        site_id = index
        for fn in csv_fls:
            basefn = os.path.basename(fn)
            if site_id in basefn:
                if site_id in file_dict:
                    file_dict[site_id].append((basefn, fn))
                else:
                    file_dict[site_id] = [(basefn, fn)]

    for site_id in file_dict:
        saveas = os.path.join(out_path, savepattern.format(site_id))
        process_tag = False
        if replace:
            process_tag = True
            logging.info('replacing or generating data file {}'.format(saveas))
        elif os.path.exists(saveas):
            process_tag = False
            logging.info('existed! data file {}'.format(saveas))
        else:
            process_tag = True
            logging.info('generating data file {}'.format(saveas))

        if process_tag:
            dfls = [forecast.fileio.load_csv(fn, index_col=None) for bfn, fn in file_dict[site_id]]
            df = pandas.concat(dfls, axis=0)
            df = df.groupby(by=[in_index]).mean()
            forecast.convertfmt.parse_datetime_index(df, fmt=in_indexfmt)
            df = df.sort_index()
            yield df, saveas
    return




def load_hdf5_data_from_path(sites, in_path, key_filter, in_index, out_path, savepattern, layers=3, replace=True):
    hdf5_fls = forecast.fileio.get_hdf5_key(in_path, layers=layers)
    file_dict = {}
    for index, row in sites.iterrows():
        site_id = index
        for fn in hdf5_fls:
            if key_filter in fn:
                basefn = os.path.basename(fn)
                if site_id in basefn:
                    if site_id in file_dict:
                        file_dict[site_id].append((basefn, fn))
                    else:
                        file_dict[site_id] = [(basefn, fn)]

    for site_id in file_dict:
        saveas = os.path.join(out_path, savepattern.format(site_id))
        process_tag = False
        if replace:
            process_tag = True
            logging.info('replacing or generating data file {}'.format(saveas))
        elif os.path.exists(saveas):
            process_tag = False
            logging.info('existed! data file {}'.format(saveas))
        else:
            process_tag = True
            logging.info('generating data file {}'.format(saveas))

        if process_tag:
            dfls = [forecast.fileio.load_hdf5_from_key(in_path, fn, in_index) for bfn, fn in file_dict[site_id]]
            df = pandas.concat(dfls, axis=0)
            df = df.groupby(by=[in_index]).mean()
            df = df.sort_index()
            yield df, saveas
    return


def load_wrf(sites, wrf_path, wrf_index, wrf_indexfmt, out_path, pattern, replace=True):
    for df, saveas in load_csv_data_from_path(sites, wrf_path, wrf_index, wrf_indexfmt, out_path, pattern, replace):
        df = rename_wrf(df)
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(saveas)


def load_cmaq_o3(sites, cmaq_path, cmaq_o3, cmaq_index, out_path, pattern, layer=3, replace=True):
    for df, saveas in load_hdf5_data_from_path(sites, cmaq_path, cmaq_o3, cmaq_index, out_path, pattern, layer, replace):
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(saveas)


def load_cmaq_pm25(sites, cmaq_path, cmaq_pm25, cmaq_index, out_path, pattern, layer=3, replace=True):
    for df, saveas in load_hdf5_data_from_path(sites, cmaq_path, cmaq_pm25, cmaq_index, out_path, pattern, layer, replace):
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(saveas)


def load_aqs_o3(sites, aqs_path, subpath, aqs_index, aqs_indexfmt, target_col, out_path, pattern, replace=True):
    for df, saveas in load_csv_data_from_path(sites, os.path.join(aqs_path, subpath), aqs_index, aqs_indexfmt, out_path, pattern, replace):
        df = df.get([target_col])
        df = df.rename(columns={target_col:'value'})
        df['date'] = [i.date() for i in df.index]
        df['month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        df['hour'] = [i.hour for i in df.index]
        df['weekday'] = [i.weekday() for i in df.index]
        # 8-hour daytime average for ozone
        dfdayavg = df.loc[(df['hour'] >= 9) & (df['hour'] <= 16)]
        dfdayavg = dfdayavg.groupby(by=['date']).mean()
        dfpast = df.get(['value'])
        dfpast = dfpast.rename(columns={'value':'last24h'})
        dfpast.index = [i + datetime.timedelta(days=1) for i in dfpast.index]
        dfpast = dfpast.get(['last24h'])
        df = pandas.concat([df, dfpast], axis=1)
        lastdayavg = []
        for index, row in df.iterrows():
            d = (index - datetime.timedelta(days=1)).date()
            v = None
            if d in dfdayavg.index:
                v = dfdayavg.loc[d, 'value']
            lastdayavg.append(v)
        df['lastday8havg'] = lastdayavg
        df.index = df.index.rename('index')
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(saveas)


def load_aqs_pm25(sites, aqs_paths, subpath, aqs_index, aqs_indexfmt, target_col, out_path, pattern, replace=True):
    for df, saveas in load_csv_data_from_path(sites, os.path.join(aqs_paths, subpath), aqs_index, aqs_indexfmt, out_path, pattern, replace):
        df = df.get([target_col])
        df = df.rename(columns={target_col:'value'})
        df['date'] = [i.date() for i in df.index]
        df['month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        df['hour'] = [i.hour for i in df.index]
        df['weekday'] = [i.weekday() for i in df.index]
        # 24-hour average for pm2.5
        dfdayavg = df.groupby(by=['date']).mean()
        dfpast = df.get(['value'])
        dfpast = dfpast.rename(columns={'value':'last24h'})
        dfpast.index = [i + datetime.timedelta(days=1) for i in dfpast.index]
        dfpast = dfpast.get(['last24h'])
        df = pandas.concat([df, dfpast], axis=1)
        lastdayavg = []
        for index, row in df.iterrows():
            d = (index - datetime.timedelta(days=1)).date()
            v = None
            if d in dfdayavg.index:
                v = dfdayavg.loc[d, 'value']
            lastdayavg.append(v)
        df['lastday8havg'] = lastdayavg
        df.index = df.index.rename('index')
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(saveas)


def load_model_input(sites, pollutant, wrf_index, wrfpattern, pollutent_index, pollutant_pattern, out_path, pattern, split_seasons=False):
    print('load_model_input')
    for index, row in sites.iterrows():
        site_id = index
        input_fn = os.path.join(out_path, pattern.format(pollutant,site_id))
        input_df = None
        if os.path.exists(input_fn):
            input_df = forecast.fileio.load_csv(input_fn, index_col='index')
            
        if input_df is not None:
            logging.info('load intermedia data for site {} {}, shape={}'.format(site_id, pollutant, input_df.shape))
            forecast.convertfmt.parse_datetime_index(input_df, fmt="%Y-%m-%d %H:%M:%S")
        else:
            wrf_fn = os.path.join(out_path, wrfpattern.format(site_id))
            plu_fn = os.path.join(out_path, pollutant_pattern.format(site_id))
            wrf = forecast.fileio.load_csv(wrf_fn, index_col=wrf_index)
            plu = forecast.fileio.load_csv(plu_fn, index_col=pollutent_index)
            if wrf is None:
                logging.info('no wrf data for site {}'.format(site_id))
            elif plu is None:
                logging.info('no pollutant={} data for site {}'.format(pollutant, site_id))
            else:
                forecast.convertfmt.parse_datetime_index(wrf, fmt="%Y-%m-%d %H:%M:%S")
                forecast.convertfmt.parse_datetime_index(plu, fmt="%Y-%m-%d %H:%M:%S")
                input_df = pandas.concat([wrf, plu], axis=1)
                input_df = input_df.dropna()
                input_df.index.rename('index', inplace=True)
                logging.info('gather data for site {} {}, shape={}'.format(site_id, pollutant, input_df.shape))
                
        if input_df is not None:
            ns, nf = input_df.shape
            if ns < 20:
                input_df = None
        
        if input_df is not None:
            logging.info('load_model_input {} {} {} {}'.format(site_id, input_df.shape, input_df.index[0], input_df.index[0].month))
            input_df = input_df.dropna()
            input_df_davg = input_df.get(['value', 'date']).groupby(by=['date']).mean()
            n_hours = input_df.shape[0]
            n_days = input_df_davg.shape[0]
            logging.info('hourly data size {}'.format(n_hours))
            logging.info('daily data size {}'.format(n_days))

            logging.info('save input file {}'.format(input_fn))
            input_df.to_csv(input_fn)

            save_bk = '../../data/AirQuality/data/USA/{}/{}.csv'.format(pollutant,site_id)
            os.makedirs(os.path.dirname(save_bk), exist_ok=True)
            logging.info('save backup file {}'.format(save_bk))
            ml_inputdata = input_df.copy()
            ml_inputdata['site'] = site_id
            ml_inputdata.to_csv(save_bk)

            if n_days > 730:
                cols = list(input_df)
                cols.remove('date')
                target_col = 'value'
                feature_cols = cols.copy()
                feature_cols.remove(target_col)
                input_df = input_df.get(cols)
 
                if split_seasons:
                    df_warm = input_df[(input_df.index.month >  4) & (input_df.index.month < 10)]
                    yield df_warm.get(feature_cols), df_warm[target_col], 'warmmonths', site_id
       
                    df_cold = input_df[(input_df.index.month > 10) | (input_df.index.month <  3)]
                    yield df_cold.get(feature_cols), df_cold[target_col], 'coldmonths', site_id
                
                else:
                    yield input_df.get(feature_cols), input_df[target_col], 'fullyear', site_id
                    

    return


def train_test_split(dfx, dsy, sep_factor=0.5):
    ns, nf = dfx.shape
    train_len = int(ns * sep_factor)
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    min_max_scaler.fit(dfx.iloc[0:train_len, :].values)
    xfull = min_max_scaler.transform(dfx.values)
    x_train = xfull[0:train_len, :]
    x_test  = xfull[train_len:, :]
    scaler = max(dsy.iloc[0:train_len].values)
    yfull = dsy.values / scaler
    y_train = yfull[0:train_len]
    y_test  = yfull[train_len:]
    return x_train, y_train, x_test, y_test, train_len, scaler



def lime_scores(x_train, x_test, time_test, feature_names, target_name, model, predict_func):
    ns, nf = x_train.shape
    categorical_features = np.argwhere(np.array([len(set(x_train[:,x])) for x in range(nf)]) <= 10).flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=[target_name], categorical_features=categorical_features, verbose=True, mode='regression')
    feature_scores = []
    ns_test, nf_test = x_test.shape
    if ns_test > 10:
        ns_test = 10
    for i in range(ns_test):
        ti = str(time_test[i])
        xi = x_test[i, :]

        predictor = Predictor(model, predict_func)
        exp = explainer.explain_instance(xi, predictor.predict, num_features=nf)
        exp_list = exp.as_list()

        feature_scores.append({ti:exp_list})
    return feature_scores


def evaluate_method(sites, pollutant, wrf_index, wrfpattern, pollutent_index, pollutant_pattern, out_path, pattern, split_seasons, model_funcs, epochs, batch_size, verbose, saveprefixes, replace=True):
    print('evaluate_method', model_funcs)
    mname, create_model, train_func, predict_func = model_funcs
    saveprefix_feature_statistics, saveprefix_prd, saveprefix_permute_train, saveprefix_permute_test, saveprefix_mae, saveprefix_importance = saveprefixes
    for dfx, dsy, datatag, site_id in load_model_input(sites, pollutant, wrf_index, wrfpattern, pollutent_index, pollutant_pattern, out_path, pattern, split_seasons=split_seasons):
        print('loop evaluate_method', model_funcs, site_id)
        feature_names = list(dfx)
        x_train, y_train, x_test, y_test, train_len, scaler = train_test_split(dfx,dsy)
        ns, nf = dfx.shape
        input_dim = nf
        output_dim = 1
        model = create_model(input_dim, output_dim)
        # saveprefix_prd, saveprefix_permute, saveprefix_mae, saveprefix_importance
        feature_statistics_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.json'.format(saveprefix_feature_statistics, mname, pollutant, site_id, datatag))
        evaluation_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.csv'.format(saveprefix_prd, mname, pollutant, site_id, datatag))
        permuted_prd_train_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.csv'.format(saveprefix_permute_train, mname, pollutant, site_id, datatag))
        permuted_prd_test_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.csv'.format(saveprefix_permute_test, mname, pollutant, site_id, datatag))
        permuted_mae_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.json'.format(saveprefix_mae, mname, pollutant, site_id, datatag))
        feature_importance_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.json'.format(saveprefix_importance, mname, pollutant, site_id, datatag))
        model_hyper_fn = os.path.join(
            out_path, '{}-{}-{}-{}-{}.json'.format('nasmodel', mname, pollutant, site_id, datatag))

        if os.path.exists(feature_statistics_fn):
            logging.info('feature statistics file {} exists'.format(feature_statistics_fn))
        else:
            logging.info('generating feature statistics file {}'.format(feature_statistics_fn))
            train_scores = {}
            for fi,featuren in enumerate(feature_names):
                xi = x_train[:,fi].flatten()
                train_scores[featuren] = {
                    'pearson r': scipy.stats.pearsonr(y_train, xi),
                    'spearmans rho': scipy.stats.spearmanr(y_train, xi),
                    'kendalls tau': scipy.stats.kendalltau(y_train, xi),
                }

            test_scores = {}
            for fi,featuren in enumerate(feature_names):
                xi = x_test[:,fi].flatten()
                test_scores[featuren] = {
                    'person r': scipy.stats.pearsonr(y_test, xi),
                    'correlation': scipy.stats.spearmanr(y_test, xi),
                    'kendalls tau': scipy.stats.kendalltau(y_test, xi),
                }

            feature_statistics = {
                'train': train_scores,
                'test': test_scores,
            }
            forecast.fileio.save_json(feature_statistics, feature_statistics_fn)

        process_tag = False
        if replace:
            process_tag = True
            logging.info('replacing or generating feature importance file {}'.format(feature_importance_fn))
        elif os.path.exists(feature_importance_fn):
            process_tag = False
            logging.info('existed! feature importance file {}'.format(feature_importance_fn))
        else:
            process_tag = True
            logging.info('generating feature importance file {}'.format(feature_importance_fn))

        print('evaluate_method process_tag', model_funcs, process_tag)
        if process_tag:
            res = forecast.modelfit.train(model, x_train, y_train, epochs, batch_size, verbose, train_func=train_func, threshold=0.02)
            if isinstance(res, tuple):
                m, res = res
                forecast.fileio.save_json(res, model_hyper_fn)
            pred = forecast.modelfit.predict(model, x_test, predict_func=predict_func)
            if len(pred.shape) > 1:
                pred = pred.flatten()
            truth = y_test * scaler
            pred = pred * scaler
            df_prd = pandas.DataFrame({
                'index': dfx.index[train_len:],
                'truth': truth,
                'prediction': pred,
            })
            df_prd.to_csv(evaluation_fn)
            ia = round(forecast.evaluations.index_of_agreement_d(truth, pred), 2)
            logging.info('evaluation for site {}, IA={}.'.format(site_id, ia))

            df_permute_prd_train, mae_permuted_train, permuted_importance_train = forecast.modelfit.permuted_predictions_and_importance(model, x_train, y_train, dfx.index[0:train_len], feature_names, predict_func=predict_func)
            df_permute_prd_train.to_csv(permuted_prd_train_fn)
            df_permute_prd_test, mae_permuted_test, permuted_importance_test = forecast.modelfit.permuted_predictions_and_importance(model, x_test, y_test, dfx.index[train_len:], feature_names, predict_func=predict_func)
            df_permute_prd_test.to_csv(permuted_prd_test_fn)
            lime_feature_importance = {} #lime_scores(x_train, x_test, dfx.index[train_len:], feature_names, 'concentration', model, predict_func)
            mae_permuted = {'train':mae_permuted_train, 'test':mae_permuted_test}
            forecast.fileio.save_json(mae_permuted, permuted_mae_fn)
            feature_importance = {'train':permuted_importance_train, 'test':permuted_importance_test, 'lime':lime_feature_importance}
            forecast.fileio.save_json(feature_importance, feature_importance_fn)





def review_evaluation(sites, sitename, out_path, saveprefix, plotpath):
    eval_ls = forecast.fileio.get_files(out_path, pattern=saveprefix+'*.csv')

    full_review_ls = []

    for efn in eval_ls:
        
        info = os.path.basename(efn)
        info_ls = info.replace('.csv', '').split('-')
        header, model, measurement, site_id, season = tuple(info_ls)
        site_name = sites.loc[site_id, sitename]
        dfx = pandas.read_csv(efn)
        ns, df = dfx.shape
        d = forecast.evaluations.index_of_agreement_d(dfx['truth'], dfx['prediction'])
        nme = forecast.evaluations.normalized_mean_error(dfx['truth'], dfx['prediction'])
        nmb = forecast.evaluations.normalized_mean_bias(dfx['truth'], dfx['prediction'])
        rmse = forecast.evaluations.root_mean_square_error(dfx['truth'], dfx['prediction'])
        r2 = forecast.evaluations.coefficient_of_determination_r2(dfx['truth'], dfx['prediction'])
        mape = forecast.evaluations.mean_absolute_percentage_error(dfx['truth'], dfx['prediction'])
        item = {
            'site': site_name,
            'd': d,
            'nmb':nmb,
            'nme':nme,
            'rmse':rmse,
            'r2':r2,
            'mape':mape,
            'model': model,
            'measurement': measurement,
            'season': season,
            'samples': ns,
        }
        full_review_ls.append(item)

    df = pandas.DataFrame(full_review_ls)
    save_name = os.path.join(plotpath, 'statistics_for_allsites_allspecies_allmodels.csv')
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name)
    
    models = df['model'].unique()
    measurements = df['measurement'].unique()
    seasons = df['season'].unique()

    models.sort()

    plot_index = {}

    for si in seasons:
        for mi in measurements:
            ls = []
            for modeli in models:
                dfi = df.loc[(df['model'] == modeli) & (df['measurement'] == mi) & (df['season'] == si)]
                dfi = dfi.set_index('site')
                col_names = ['d', 'nmb', 'nme', 'rmse', 'r2','mape']  #mape has many inf and causing problem? 
                colmap = {}
                for ci in col_names:
                    colmap[ci] = modeli + '-' + ci
                dfi = dfi.rename(columns=colmap)

                dfsize = dfi.get(['samples'])
                
                ## ls is causing a problem and I am not sure what is actual purpose doing this. so all commented out 
                ext_col=[colmap[x] for x in col_names] 
                ls.append(dfi.get(ext_col))           #yunha       ls.append(dfi.get([modeli]))
            
                # not useful save_name = os.path.join(plotpath, 'eval-{}-{}-{}-size.csv'.format(mi, si, modeli))
                # not useful os.makedirs(os.path.dirname(save_name), exist_ok=True)

                # not useful dfsize.to_csv(save_name)
            dfp = pandas.concat(ls, axis=1)
            
            # save all model performances for all metrics
            save_name = os.path.join(plotpath, 'eval-{}-{}.csv'.format(mi, si))
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            dfp.to_csv(save_name)
                              
            # plot all model performance for each metric
            save_name = os.path.join(plotpath, 'eval-{}-{}.pdf'.format(mi, si))
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            pdf = PdfPages(save_name)
            plt.rcParams.update({'font.size': 26}) # must set in top
            for ci in col_names:
                dfp_stat=dfp.filter(like='-' +ci)
                print(dfp_stat.columns, ci)
                dfp_stat = dfp_stat.sort_values(by=[models[0]+ '-' +ci])
                ax = dfp_stat.plot.bar(figsize=(30,12),  alpha=0.75, rot=90, title= mi+" in "+si, fontsize=16)
                ax.set_ylabel(ci,fontdict={'fontsize':26})
                pdf.savefig()
            pdf.close()
            plt.close()
    return


def review_feature_importance(sites, sitename, out_path, saveprefix, plotpath, plot_index=None):

    feature_importance_ls = forecast.fileio.get_files(out_path, pattern=saveprefix+'*.json')

    full_review_ls = []
    features = []

    for fifn in feature_importance_ls:
        info = os.path.basename(fifn)
        info_ls = info.replace('.json', '').split('-')
        #print("review feature impo 1 ", info, info_ls, fifn)
        header1, header2, model, measurement, site_id, season = tuple(info_ls)
        
        #print("review feature impo 2 ", header1, header2, model, measurement, site_id, season)
        
        site_name = sites.loc[site_id, sitename]
        print("feature info", site_name)
        d = forecast.fileio.load_json(fifn)
        features = list(d.keys())
        #print('debug feature impo ', d.keys(), features )
        d.update({
            'site': site_name,
            'model': model,
            'measurement': measurement,
            'season': season,
        })
        
        #print('debug feature impo after updated', d.keys(), features )
            
        full_review_ls.append(d)
        
    df = pandas.DataFrame(full_review_ls)
    save_name = os.path.join(plotpath, 'review-feature-importance.csv')
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name)

    models = df['model'].unique()
    measurements = df['measurement'].unique()
    seasons = df['season'].unique()

    features.sort()
    models.sort()
    
    # lime is not used, so remove it. 
    features.remove('lime')
    
    for si in seasons:
        for mi in measurements:
            ls = []
            for modeli in models:
                dfi = df.loc[(df['model'] == modeli) & (df['measurement'] == mi) & (df['season'] == si)]
                dfi = dfi.set_index('site')
                colmap = {}
                feature_cols = []
                for fi in features:
                    v = modeli + '-' + fi
                    colmap[fi] = v
                    feature_cols.append(v)
                dfi = dfi.rename(columns=colmap)
                print("review_eval dfi ",dfi.keys())
                ls.append(dfi.get(feature_cols))
            dfp = pandas.concat(ls, axis=1)
            
            print("review_eval dfp ",dfp.keys())
            
            # save all model importance 
            # off save_name = os.path.join(plotpath, 'eval-{}-{}.csv'.format(mi, si))
            # off os.makedirs(os.path.dirname(save_name), exist_ok=True)
            # off dfp.to_csv(save_name)
            
            save_name = os.path.join(plotpath, 'featureimportance-{}-{}.pdf'.format(mi, si))
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            pdf = PdfPages(save_name)
            plt.rcParams.update({'font.size': 16}) # must set in top
            
            for fi in dfp:
                #print(fi)
                dfp_feature= dfp[fi].apply(pandas.Series)
                ax=dfp_feature.plot.bar(figsize=(30,12),  alpha=0.75, rot=90, title= mi+" in "+si, fontsize=20)
                ax.set_ylabel(fi,fontdict={'fontsize':26})
                pdf.savefig()   
                
                normalized_df = dfp_feature.apply(lambda x: x / sum(x), axis=1)
                normalized_df = normalized_df.sort_values(by=['lastday8havg'])
                ax=normalized_df.plot.bar(figsize=(30,12),  alpha=0.75, rot=90, fontsize=20,stacked=True)
                ax.set_ylabel("normalized "+fi,fontdict={'fontsize':26})
                pdf.savefig()    
            pdf.close()
            plt.close()

                            

def ml_pipeline(sites, wrfpattern, o3_pattern, pm_pattern, out_path, pattern, flags, saveprefixes, replace=True):
    tworf_model_funcs = [
        '2rf',
        forecast.modelsln.twostage_randomforest,
        forecast.modelsln.train_2randomforest,
        forecast.modelsln.predict_2randomforest,
    ]
    dense_model_funcs_xl = [
        'denseXL',
        forecast.modelsnn.Dense_512_256_128,
        forecast.modelsnn.train_nn,
        forecast.modelsnn.predict_nn,
    ]
    dense_model_funcs_l = [
        'denseL',
        forecast.modelsnn.Dense_256_128_64,
        forecast.modelsnn.train_nn,
        forecast.modelsnn.predict_nn,
    ]
    dense_model_funcs_m = [
        'denseM',
        forecast.modelsnn.Dense_128_64_32,
        forecast.modelsnn.train_nn,
        forecast.modelsnn.predict_nn,
    ]
    dense_model_funcs_s = [
        'denseS',
        forecast.modelsnn.Dense_64_32_16,
        forecast.modelsnn.train_nn,
        forecast.modelsnn.predict_nn,
    ]
    dense_model_funcs_xs = [
        'denseXS',
        forecast.modelsnn.Dense_32_16_8,
        forecast.modelsnn.train_nn,
        forecast.modelsnn.predict_nn,
    ]


    flag2rf, flagdensexl, flagdensel, flagdensem, flagdenses, flagdensexs = flags
    print('flag2rf, flagdensexl, flagdensel, flagdensem, flagdenses, flagdensexs', flag2rf, flagdensexl, flagdensel, flagdensem, flagdenses, flagdensexs)

    for pollutant, pollutant_pattern in [('o3', o3_pattern), ('pm25', pm_pattern)]:
        for split_seasons in [False, True]:
            # 2rf
            if flag2rf:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, tworf_model_funcs,     None, None, None, saveprefixes,     replace=replace)
            # dense models
            if flagdensexl:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, dense_model_funcs_xl,  100,  None, None, saveprefixes,     replace=replace)
            if flagdensel:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, dense_model_funcs_l,   100,  None, None, saveprefixes,     replace=replace)
            if flagdensem:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, dense_model_funcs_m,   100,  None, None, saveprefixes,     replace=replace)
            if flagdenses:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, dense_model_funcs_s,   100,  None, None, saveprefixes,     replace=replace)
            if flagdensexs:
                evaluate_method(sites, pollutant,   'date', wrfpattern, 'index', pollutant_pattern, out_path, pattern, split_seasons, dense_model_funcs_xs,  100,  None, None, saveprefixes,     replace=replace)



def main(args):

    sites = None
    if os.path.exists(args.gmtoffset):
        sites = load_sites(args.gmtoffset, args.siteid)
    elif os.path.exists(args.sites):
        sites = load_sites(args.sites, args.siteid)
        sites = load_timezone_from_geolocation(sites, args.siteid, args.sitelati, args.sitelong)
        sites.to_csv(args.gmtoffset, index=True)

    if sites is None:
        return

    # saveprefix_feature_statistics, saveprefix_prd, saveprefix_permute_train, saveprefix_permute_test, saveprefix_mae, saveprefix_importance
    save_prefixes = ('feature-statistics', 'prd', 'permuted-train', 'permuted-test', 'mae', 'feature-importance')
    flags = (args.flag2rf, args.flagdensexl, args.flagdensel, args.flagdensem, args.flagdenses, args.flagdensexs)
    plot_index = None
    
    if args.pipeloadwrf:
        load_wrf(sites, args.wrf_path, args.wrf_index, args.wrf_indexfmt, args.out_path, args.wrffilepattern, replace=False)

    if args.pipeloadaqs:
        load_aqs_o3(sites, args.aqs_path, args.aqs_o3subfolder, args.aqs_index, args.aqs_indexfmt, args.aqs_target, args.out_path, args.o3filepattern, replace=False)
        load_aqs_pm25(sites, args.aqs_path, args.aqs_pm25subfolder, args.aqs_index, args.aqs_indexfmt, args.aqs_target, args.out_path, args.pm25filepattern, replace=False)

    if args.pipeloadcmaq:
        load_cmaq_o3(sites, args.cmaq_path, args.cmaq_o3header, args.cmaq_index, args.out_path, args.cmaqo3pattern, args.cmaq_layer, replace=False)
        load_cmaq_pm25(sites, args.cmaq_path, args.cmaq_pm25header, args.cmaq_index, args.out_path, args.cmaqpm25pattern, args.cmaq_layer, replace=False)
        
    if args.pipeml:
        ml_pipeline(sites, args.wrffilepattern, args.o3filepattern, args.pm25filepattern, args.out_path, args.inputfilepattern, flags, save_prefixes, replace=False)

    if args.pipeploteval:
        print('review eval passing vars are ', args.sitename, save_prefixes[1])
        plot_index = review_evaluation(sites, args.sitename, args.out_path, save_prefixes[1], args.plotpath)

    if args.pipeplotimportance:
        review_feature_importance(sites, args.sitename, args.out_path, save_prefixes[-1], args.plotpath, plot_index=plot_index)


if __name__ == '__main__':
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)

    os.makedirs(os.path.dirname(args.logpath), exist_ok=True)

    logging.basicConfig(
        filename=args.logpath,
        filemode='w',
        format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )

    logging.info('start')
    logging.info('parameters: {}'.format(vargs))

    main(args)

    logging.info('end preprocessing')
