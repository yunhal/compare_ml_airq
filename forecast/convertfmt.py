import datetime
import xarray
import math
import pandas as pd
import numpy as np
import logging


def convert_xarray_to_dataframe(xrfn, index='datetime'):
    # load data
    xrdata = xarray.open_dataarray(xrfn)

    # reset index, xarray may have different index data
    data_name = 'value' # column name for values
    df = xrdata.to_dataframe(name=data_name, dim_order=None)
    df = df.reset_index()
    df = df.set_index([index])

    # get variables
    var_col = 'variables'
    cols = df[var_col].unique()

    # reformat table according to variables
    ls = []
    for c in cols:
        dfi = df.loc[df[var_col] == c]
        dfi = dfi.rename(columns={data_name:c})
        dfis = dfi[c]
        ls.append(dfis)

    df = pd.concat(ls, axis=1)
    return df


def numpy_to_dataframe(data, index, cols):
    df = pd.DataFrame(data=data, index=index, columns=cols)
    return df


def convert_to_float(df):
    return df.astype('float32')


def dataframe_rename_columns(df, colmap):
    return df.rename(columns=colmap)


def dataframe_rename_index(df, newname):
    df.index.rename(newname, inplace=True)


def parse_datetime_index(df, fmt="%Y-%m-%d %H:%M:%S"):
    df.index = pd.to_datetime(df.index, format=fmt, exact=False)


def parse_datetime(df, col, fmt="%Y-%m-%d %H:%M:%S"):
    return pd.to_datetime(df[col], format=fmt, exact=False)


def parse_date_index(df, fmt="%Y-%m-%d"):
    df.index = pd.to_datetime(df.index, format=fmt, exact=False)


def assign_class(df, col):
    df['cd10'] = [int(i/10.0) for i in df[col]]
    clog2 = []
    for i in df[col]:
        x = i
        if x < 0:
            logging.error('log value exception x={}'.format(x))
            x = 0
        xx = int(math.log(x+math.e)-1)
        clog2.append(xx)
    df['clog2'] = clog2
    return df


def assign_timeinfo(df):
    df['monthNo'] = [i.month for i in df.index]
    df['dayNo'] = [i.day for i in df.index]
    df['weekdayNo'] = [i.weekday() for i in df.index]
    return df


# index must be hourly timestamp
def hourly2daily(dfh):
    dfd = dfh.copy()
    dfd['date'] = [i.date() for i in dfd.index]
    dfd = dfd.sort_values(by=['date'], ascending=True)
    dfd = dfd.groupby(by=["date"]).mean()
    return dfd
