import pandas

def value_percentage(df, col, value):
    ns, nc = df.shape
    nvs, nc = df.loc[df[col] == value].shape
    return ns, nvs / float(ns)



from __future__ import division
import numpy as np
from scipy.stats import gmean, rankdata
import warnings


def nme(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):

    # Treating missing values
    simulated_array, observed_array = treat_values(simulated_array, observed_array,
                                                   replace_nan=replace_nan,
                                                   replace_inf=replace_inf,
                                                   remove_neg=remove_neg,
                                                   remove_zero=remove_zero)
    a = np.abs(simulated_array - observed_array)
    return 100 * np.sum(a) /np.sum(observed_array)


def nmb(simulated_array, observed_array, replace_nan=None, replace_inf=None,
       remove_neg=False, remove_zero=False):

    # Treating missing values
    simulated_array, observed_array = treat_values(simulated_array, observed_array,
                                                   replace_nan=replace_nan,
                                                   replace_inf=replace_inf,
                                                   remove_neg=remove_neg,
                                                   remove_zero=remove_zero)
    a = np.sum(simulated_array - observed_array)
    return 100 * a / np.sum(observed_array)
