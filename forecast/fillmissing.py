import datetime
import pandas as pd
import numpy as np



def get_year_sample(df, start_year, end_year):
    df_year = df.copy()
    cols = list(df)

    total_days = pd.date_range(start='1/1/{}'.format(start_year), end='12/31/{}'.format(end_year))
    df_year_full = df_year.reindex(total_days)
    df_year_full['month'] = [i.month for i in df_year_full.index]
    df_year_full['day'] = [i.day for i in df_year_full.index]

    # get a year with 366 days for a year-sample
    date_range = []
    for i in range(start_year, end_year+1):
        dri = pd.date_range(start='1/1/{}'.format(i), end='12/31/{}'.format(i))
        l = len(dri)
        if l > len(date_range):
            date_range = dri.copy()

    df_year_sample = df_year_full.loc[date_range,:]
    df_year_sample = df_year_sample.reset_index()
    df_year_sample = df_year_sample.set_index(['month','day'])
    df_year_sample = df_year_sample.get(cols)

    for i in range(start_year, end_year+1):
        date_range = pd.date_range(start='1/1/{}'.format(i), end='12/31/{}'.format(i))
        di = df_year_full.loc[date_range,:]
        di = di.reset_index()
        di = di.set_index(['month','day'])
        di = di.get(cols)
        df_year_sample = df_year_sample.fillna(di)

    return df_year_sample



def repeat_data_year_sample(df_year_sample, start_year, end_year):
    ls = []
    for i in range(start_year, end_year+1):
        dfy = df_year_sample.copy()
        dfy = dfy.reset_index()
        ns, nc = dfy.shape
        dfy['year'] = [i] * ns
        dfy['date'] = [
            '{:04}-{:02}-{:02}'.format(
                int(row['year']), int(row['month']), int(row['day'])
            ) for index, row in dfy.iterrows()
        ]
        dfy = dfy.set_index(['date'])

        date_range = pd.date_range(start='1/1/{}'.format(i), end='12/31/{}'.format(i))
        date_range = [i.strftime('%Y-%m-%d') for i in date_range]

        dfy = dfy.loc[date_range,:]

        ls.append(dfy.copy())

    df_fill_ys = pd.concat(ls, axis=0)
    df_fill_ys.index = pd.to_datetime(df_fill_ys.index, format="%Y-%m-%d", exact=False)
    return df_fill_ys



def fillna_by_sampling(df):
    min_day, max_day = min(df.index), max(df.index)
    start_year, end_year = min_day.year, max_day.year
    df_year_sample = get_year_sample(df, start_year, end_year)
    df_fill_ys = repeat_data_year_sample(df_year_sample, start_year, end_year)
    data = df.fillna(df_fill_ys)
    return data



def fillna_by_interpolate(df):
    max_day, min_day = max(df.index), min(df.index)
    total_days = pd.date_range(min_day, max_day)
    data = df.reindex(total_days)
    data = data.interpolate(method='linear')
    return data



# input df is one column
# note frequency url: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
# D: calendar day frequency
# W: weekly frequency
# H: hourly frequency
# T, min: minutely frequency
# S: secondly frequency
def series_fillna_by_interpolate(df, freq='H'):
    index = df.index.copy()
    max_day, min_day = max(df.index), min(df.index)
    total_index = pd.date_range(min_day, max_day, freq=freq)
    data = df.reindex(total_index)
    data = data.interpolate(method='linear')
    data['obs'] = [0] * len(total_index)
    data.loc[index, 'obs'] = [1] * len(index)
    return data
