import numpy
import pandas


def rolling(df, col, winsize=[2,3,4,5,6,7], wintype=None):
    out = df.copy()
    dfx = df.get([col])
    for ws in winsize:
        x = dfx.rolling(ws, min_periods=1, win_type=wintype).mean()
        key = '{}Roll{:05}'.format(col, ws)
        out[key] = x

    return out
