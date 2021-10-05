import os, os.path, sys
import json
import uuid
import platform
import datetime
import jsonpickle

jsonpickle.set_preferred_backend('simplejson')
jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Configure(object):

    def __init__(self, path, name):
        self.exp_name = name
        self.exp_path = path
        self.exp_dir  = os.path.join(self.exp_path, self.exp_name)
        self.confpath = os.path.join(self.exp_dir, 'conf.json')
        self.conf = {}
        self.data_path = 'data'
        self.prdt_path = 'predict'
        if os.path.exists(self.confpath):
            self.from_file()
        else:
            self.initialize()


    def initialize(self):
        self.conf = {
            'data_path'     : os.path.join(self.exp_dir, self.data_path).replace('\\', '/'),
            'filename'      : os.path.join(self.exp_dir, self.data_path, 'data.csv').replace('\\', '/'),
            'index_col'     : 'date',
            'ocols'         : [],
            'rcols'         : [],
            'target'        : 'pm25',
            'figsize'       : (50,9),
            'plotindex'     : [0],
            'fillna'        : 'interpolate',
            "sep_year"      : 2020,
            'lookback'      : 7,
            'prdlen'        : 1,
            'epoch'         : 200,
            'batch_size'    : 40,
            'result_path'   : os.path.join(self.exp_dir, self.prdt_path).replace('\\', '/'),
        }
        if not os.path.exists(self.confpath):
            self.save_file()
        self.attr = AttrDict()
        self.attr.update(self.conf)
        self.valid = True


    def from_file(self):
        if os.path.isfile(self.confpath):
            if os.path.exists(self.confpath):
                with open(self.confpath, 'r') as f:
                    self.data = f.read()
                    self.conf = jsonpickle.decode(self.data)
                    self.attr = AttrDict()
                    self.attr.update(self.conf)


    def save_file(self):
        os.makedirs(os.path.dirname(self.confpath), exist_ok=True)
        with open(self.confpath, 'w') as f:
            f.write(jsonpickle.encode(self.conf))





if __name__ == '__main__':
    c = Configure('../expdata', 'wsu')
    c.save_file()
    c = Configure('../expdata', 'debw')
    c.save_file()
    c = Configure('../expdata', 'xian')
    c.save_file()
