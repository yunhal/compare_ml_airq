import datetime
import pandas as pd
import numpy as np


#
#
# ocols: original cols
# rcols: renamed cols
# target: first column is treated as target by default, otherwise, switch cols
# cols: column-selection
# #
def load_csv(csvfn, index_col=None, rename_index=None, indexdtfmt=None, ocols=[], rcols=[], target=None, cols=None):
    df = pd.read_csv(csvfn, index_col=index_col)
    if indexdtfmt is not None:
        df.index = pd.to_datetime(df.index, format=indexdtfmt, exact=False)

    if rename_index is not None:
        df.index.rename(rename_index, inplace=True)

    if isinstance(ocols, str):
        ocols = ocols.split(',')
    if isinstance(rcols, str):
        rcols = rcols.split(',')
    if isinstance(cols, str):
        cols = cols.split(',')

    colmap = {}
    for f,t in zip(ocols,rcols):
        colmap[f] = t

    df = df.rename(columns=colmap)

    if len(rcols) == 0:
        rcols = list(df)

    # use new cols if not specified
    if cols is None:
        cols = rcols.copy()
    elif len(cols) == 0:
        cols = rcols.copy()

    if target is None:
        target = cols[0]
        feature_cols = cols[1:]
    else:
        feature_cols = cols.copy()
        feature_cols.remove(target)
        feature_cols.sort()
        cols = [target] + feature_cols

    # return original dataframe and column-selection
    return df, df.get(cols)
