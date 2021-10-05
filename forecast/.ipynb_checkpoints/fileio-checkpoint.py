import os, os.path
import glob
import pandas as pd
import json
import pickle
import logging

def load_csv(filename, index_col):
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=index)


def load_json(filename):
    data = None
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except:
            logging.error('cannot load json {}'.format(filename))
    else:
        logging.error('file does not exist -- json {}'.format(filename))
    return data


def save_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True, indent='   ')
        
        
        
def get_exp_list(data_path, predict='predict'):
    ls = os.listdir(data_path)
    r = {}
    for i in ls:
        p_path = os.path.join(data_path, i, predict)
        if os.path.exists(p_path):
            r[i] = p_path
            
    return r