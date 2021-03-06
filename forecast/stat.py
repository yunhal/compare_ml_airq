import pandas
import numpy as np
from scipy.stats import gmean, rankdata
import warnings

def value_percentage(df, col, value):
    ns, nc = df.shape
    nvs, nc = df.loc[df[col] == value].shape
    return ns, nvs / float(ns)


