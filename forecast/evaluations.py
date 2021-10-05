import math
import numpy as np
import sklearn.metrics as skmetrics
import forecast.goodness_of_fit as gof
import forecast.HydroErr as he


def max_min_data(df, index_prefix='day_', data_prefix='data_'):
    i_r = df.index.to_series().describe()
    d_r = df.describe()

    r = {}
    for k,v in i_r.to_dict().items():
        r[index_prefix+k] = v
    for k,v in d_r.to_dict().items():
        if isinstance(v, dict):
            for kk,vv in v.items():
                r[k+'_'+kk] = vv
        else:
            r[data_prefix+k] = v
    return r


def remove_zero_values(truth, prdct):
    truth_values = []
    prdct_values = []

    truthv = truth
    if isinstance(truthv, list):
        truthv = truth
    else:
        truthv = truth.tolist()
    prdctv = prdct
    if isinstance(prdctv, list):
        prdctv = prdct
    else:
        prdctv = prdct.tolist()

    for t,p in zip(truthv, prdctv):
        if t != 0.0:
            truth_values.append(t)
            prdct_values.append(p)
    return truth_values, prdct_values



def evaluate(truth, prd):
    mape = skmetrics.mean_absolute_percentage_error(truth, prd)
    mae  = skmetrics.mean_absolute_error(truth, prd)
    mse  = skmetrics.mean_squared_error(truth, prd)
    epv  = skmetrics.explained_variance_score(truth, prd)
    r2   = skmetrics.r2_score(truth, prd)

    r = {
        'mape':mape,
        'mae':mae,
        'mse':mse,
        'epv':epv,
        'r2':r2,
    }

    return mape, mae, mse, epv, r2, r

def __preprocessing(cal, obs, transform=None, eps=1e-6):
    """
    Helper to check input data.
    """
    cal = np.asarray(cal)
    obs = np.asarray(obs)

    are_comparable = cal.shape == obs.shape and cal.ndim == obs.ndim == 1
    if not are_comparable:
        raise ValueError("Arguments must be 1D numpy.ndarrays of the same shape!")

    if transform is not None:
        if isinstance(transform, str):
            if transform == 'sqrt':
                cal = np.sqrt(cal)
                obs = np.sqrt(obs)
            elif transform == 'log':
                cal[np.abs(cal) < eps] = eps
                cal = np.log(cal)
                obs[np.abs(obs) < eps] = eps
                obs = np.log(obs)
            elif transform == 'inv':
                cal[np.abs(cal) < eps] = eps
                cal = np.reciprocal(cal)
                obs[np.abs(obs) < eps] = eps
                obs = np.reciprocal(obs)
            elif transform == 'boxcox':
                cal = (np.power(cal, 0.25) - 0.01 * np.nanmean(cal)) * 4.0
                obs = (np.power(obs, 0.25) - 0.01 * np.nanmean(obs)) * 4.0
        elif isinstance(transform, int):
            cal = np.power(cal, transform)
            obs = np.power(obs, transform)
        else:
            raise ValueError(f'Incorrect transformation {transform}!')

    return cal, obs


def is_flat(signal, eps=1e-2):
    """
    Return True if the signal is flat.
    The signal is consider flat if the ration between standard deviation and mean of the signal is lower than a given threshold (eps).
    :param signal: (N,) array_like.
    :param eps: Tolerance criterion.
    :return: Bool.
    """
    s = np.asarray(signal)
    return np.std(s) < eps * np.nanmean(s)


def berry_and_mielke(cal, obs, transform=None, eps=1e-6):
    cal, obs = __preprocessing(cal, obs, transform=transform, eps=eps)
    delta = np.nanmean(np.abs(cal - obs))

    ns = obs.shape[0]

    mu = 0
    for i in range(ns):
        for j in range(ns):
            mu += abs(cal[j] - obs[i])
    mu = mu / (ns * ns)
    return 1 - (delta / mu)


def theil_inequality_coefficient(cal, obs, transform=None, eps=1e-6):
    cal, obs = __preprocessing(cal, obs, transform=transform, eps=eps)
    delta = np.nanmean(np.power(obs - cal, 2.0))
    mu = np.nanmean(np.power(obs, 2.0))

    return math.sqrt(delta) / math.sqrt(mu)


def legates_and_mccabe(cal, obs, transform=None, eps=1e-6):
    cal, obs = __preprocessing(cal, obs, transform=transform, eps=eps)
    delta = np.nansum(np.abs(obs - cal))
    amean = np.nanmean(obs)
    mu = np.nansum(np.abs(obs - amean))

    return 1 - (delta / mu)



def index_of_agreement_d(truth, prd):
    return he.d(prd, truth)


def normalized_mean_bias(truth, prd):
    return he.nmb(prd, truth)


def normalized_mean_error(truth, prd):
    return he.nme(prd, truth)


def root_mean_square_error(truth, prd):
    return he.rmse(prd, truth)


def mean_absolute_percentage_error(truth, prd):
    return he.mape(prd, truth)


def coefficient_of_determination_r2(truth, prd):
    return he.r_squared(prd, truth)


def pearson_correlation_coefficient(truth, prd):
    return he.pearson_r(prd, truth)


def mutual_information(truth, prd):
    return skmetrics.mutual_info_score(truth, prd)


def evaluate_expand(truth, prd):

    cal, obs = __preprocessing(prd, truth, transform=None, eps=1e-6)
    r = {
        # Mean Error
        # Mean Absolute Error
        # Root Mean Square Error
        # Normalized Root Mean Square Error
        # Pearson product-moment correlation coefficient
        # Coefficient of Determination
        'me'        : gof.me(prd, truth),
        'mabse'     : gof.mae(prd, truth),
        'rmse'      : gof.rmse(prd, truth),
        'nrmse'     : gof.nrmse(prd, truth),
        'pc'        : gof.r_pearson(prd, truth),
        'r^2'       : gof.r2(prd, truth),

        # Index of Agreement
        # Modified Index of Agreement
        # Relative Index of Agreement
        'ia'        : gof.d(prd, truth),
        'mia'       : gof.md(prd, truth),
        'ria'       : gof.rd(prd, truth),

        # Ratio of Standard Deviations
        'rsd'       : gof.rsd(prd, truth),

        # Nash-sutcliffe Efficiency
        # Modified Nash-sutcliffe Efficiency
        # Relative Nash-sutcliffe Efficiency
        'nse'       : gof.nse(prd, truth),
        'mnse'      : gof.mnse(prd, truth),
        'rnse'      : gof.rnse(prd, truth),

        # Kling Gupta Efficiency
        # Deviation of gain
        # Standard deviation of residual
        'kge'       : gof.kge(prd, truth),
        'dg'        : gof.dg(prd, truth),
        'sdr'       : gof.sdr(prd, truth),

        'r_squared' : he.r_squared(cal, obs),
        'spearman'  : he.spearman_r(cal, obs),
        'acc'       : he.acc(cal, obs),           # anomaly correlation coefficient (ACC).
        'dia'       : he.d(cal, obs),
        'd1ia'      : he.d1(cal, obs),
        'dmod'      : he.dmod(cal, obs),
        'dr'        : he.dr(cal, obs),            # refined index of agreement (dr).
        'mb_r'      : he.mb_r(cal, obs),          # Mielke-Berry R value (MB R).
        'kge_2009'  : he.kge_2009(cal, obs),      # Kling-Gupta efficiency (2009).
        'kge_2012'  : he.kge_2012(cal, obs),      # Kling-Gupta efficiency (2012).
        'lm_index'  : he.lm_index(cal, obs),      # Legate-McCabe Efficiency Index.
        'd1_p'      : he.d1_p(cal, obs),          # Legate-McCabe Efficiency Agreement.
        'mb_r'      : he.mb_r(cal, obs),          # Mielke-Berry R value (MB R).
        've'        : he.ve(cal, obs),            # Volumetric Efficiency (VE).
        'sa'        : he.sa(cal, obs),            # Spectral Angle (SA).
        'sc'        : he.sc(cal, obs),            # Spectral Correlation (SC).
        #'sid'       : he.sid(cal, obs),           # Spectral Information Divergence (SID).
        'sga'       : he.sga(cal, obs),           # Spectral Gradient Angle (SGA).

        'mielke'    : berry_and_mielke(prd, truth),
        'theil'     : theil_inequality_coefficient(prd, truth),
        'mccabe'    : legates_and_mccabe(prd, truth),
    }
    return r
