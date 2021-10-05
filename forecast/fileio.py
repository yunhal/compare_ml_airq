import os, os.path
import glob
import pandas as pd
import json as jsonio
import numpy
import pickle
import logging
import h5py

def load_csv(filename, index_col=None):
    df = None
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=index_col)
        except:
            logging.error('cannot load csv {}'.format(filename))
    else:
        logging.error('file does not exist -- csv {}'.format(filename))
    return df


def save_csv(df, filename, index=False):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=index)


def load_json(filename):
    data = None
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = jsonio.load(f)
        except:
            logging.error('cannot load json {}'.format(filename))
    else:
        logging.error('file does not exist -- json {}'.format(filename))
    return data


def save_json(data, filename):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        jsonio.dump(data, f, sort_keys=True, indent='   ')



def load_npz(filename):
    data = None
    if os.path.exists(filename):
        try:
            data = numpy.load(filename)
        except:
            logging.error('cannot load numpy data {}'.format(filename))
    else:
        logging.error('file does not exist -- numpy data {}'.format(filename))

    if data is not None:
        if 'data' in data:
            data = data['data']
    return data



def save_npz(data, filename):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    numpy.savez_compressed(filename, data=data)



def get_files(path, pattern='*.npz'):
    result = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], pattern))]
    return result


def get_subdir_list(data_path, item=None):
    ls = os.listdir(data_path)
    r = {}
    for i in ls:
        if item is None:
            if '.' not in i:
                p_path = os.path.join(data_path, i)
                if os.path.isdir(p_path):
                    r[i] = p_path
        else:
            p_path = os.path.join(data_path, i, item)
            if os.path.exists(p_path):
                r[i] = p_path
    return r


def get_keys(hdf5obj, key, layer):
    ls = []

    if key is None:
        ls = list(hdf5obj.keys())
    else:
        if isinstance(hdf5obj[key], h5py._hl.group.Group):
            ls = list(hdf5obj[key].keys())
        else:
            return [], layer+1

    if key is None:
        ls = ls
    else:
        ls = [os.path.join(key, i) for i in ls]

    return ls, layer+1



def get_hdf5_key(hdf5fn, layers=2):
    hf = h5py.File(hdf5fn, 'r')
    keys, layer_count = get_keys(hf, None, 0)

    while layer_count < layers:
        ls = []
        lc = layer_count
        for i in keys:
            lsi, lc = get_keys(hf, i, layer_count)
            ls.extend(lsi)
        keys, layer_count = ls, lc

    hf.close()
    return keys


def load_hdf5_from_key(filename, key, index_col):
    df = pd.read_hdf(filename, key=key, index_col=index_col)
    return df
