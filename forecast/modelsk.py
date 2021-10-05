#
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import Input, Activation, Dropout, Embedding, Flatten
from tensorflow.keras.layers import TimeDistributed, RepeatVector, BatchNormalization
from tensorflow.keras.layers import Attention, AdditiveAttention, MultiHeadAttention
from tensorflow.keras  import callbacks
from tensorflow.keras import optimizers

try:
    import autokeras as ak
except:
    ak = None
    print('auto-keras is not installed')



def dense__d_64_32_16__nn(input_dim, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """Dense_{64,32,16}"""
    name = 'Dense_{64,32,16}'
    dense__d_64_32_16__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, input_dim=input_dim, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name



def dense__d_128_32__nn(input_dim, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """Dense_{128,32}"""
    name = 'Dense_{128,32}'
    dense__d_128_32__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, input_dim=input_dim, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def dense__d_128_t3__nn(input_dim, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """Dense_{128-t3}"""
    name = 'Dense_{128-t3}'
    dense__d_128_t3__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, input_dim=input_dim, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def dense__d_128_t5__nn(input_dim, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """Dense_{128-t5}"""
    name = 'Dense_{128-t5}'
    dense__d_128_t5__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, input_dim=input_dim, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_32__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{32}"""
    name = 'LSTM_{32}'
    lstm__l_32__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_64__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{64}"""
    name = 'LSTM_{64}'
    lstm__l_64__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(64, activation=activation, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{128}"""
    name = 'LSTM_{128}'
    lstm__l_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(128, activation=activation, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_256__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{256}"""
    name = 'LSTM_{256}'
    lstm__l_256__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(256, activation=activation, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_32__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{32}_Dense_{128}"""
    name = 'LSTM_{32}_Dense_{128}'
    lstm__l_32__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=False))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_32_t2__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{32-t2}_Dense_{128}"""
    name = 'LSTM_{32-t2}_Dense_{128}'
    lstm__l_32_t2__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_32_t3__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{32-t3}_Dense_{128}"""
    name = 'LSTM_{32-t3}_Dense_{128}'
    lstm__l_32_t3__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_32_t5__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{32-t5}_Dense_{128}"""
    name = 'LSTM_{32-t5}_Dense_{128}'
    lstm__l_32_t5__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__bn__l_32_t3__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """BN_LSTM_{32-t3}_Dense_{128}"""
    name = 'BN_LSTM_{32-t3}_Dense_{128}'
    lstm__bn__l_32_t3__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__bn__l_32_t5__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """BN_LSTM_{32-t4}_Dense_{128}"""
    name = 'BN_LSTM_{32-t4}_Dense_{128}'
    lstm__bn__l_32_t5__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(32, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__bn__l_64_t4__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """BN-LSTM_{64-t4}_Dense_{128}"""
    name = 'BN-LSTM_{64-t4}_Dense_{128}'
    lstm__bn__l_64_t4__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(64, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__bn__l_64_t5__d_128__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """BN_LSTM_{64-t5}_Dense_{128}"""
    name = 'BN_LSTM_{64-t5}_Dense_{128}'
    lstm__bn__l_64_t5__d_128__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(64, activation=activation, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=True))
    model.add(LSTM(64, activation=activation, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_150_rt2__td_64__d_64__nn(input_shape, output_dim=1, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{150-r2}_TD_{64}_Dense_{64}"""
    name = 'LSTM_{150-r2}_TD_{64}_Dense_{64}'
    lstm__l_150_rt2__td_64__d_64__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(150, activation=activation, input_shape=input_shape))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(150, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(64, activation=activation)))
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def lstm__l_150_rt2__mat__td_64__d_64__nn(input_shape, output_dim=1, nhead=2, keydim=2, activation='relu', out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """LSTM_{150-r25}_MAT_TD_{64}_Dense_{64}"""
    name = 'LSTM_{150-r25}_MAT_TD_{64}_Dense_{64}'
    lstm__l_150_rt2__mat__td_64__d_64__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(LSTM(150, activation=activation, input_shape=input_shape))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(150, activation=activation, return_sequences=True))
    model.add(MultiHeadAttention(num_heads=nhead, key_dim=keydim, attention_axes=None))
    model.add(TimeDistributed(Dense(64, activation=activation)))
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def cnn__c2d_32_f3__f__dp__d_64_32_16__nn(input_shape, output_dim=1, activation='relu', dropout=0.5, out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """CNN_{32}{3}_Dense_{64,32,16}"""
    name = 'CNN_{32}{3}_Dense_{64,32,16}'
    cnn__c2d_32_f3__f__dp__d_64_32_16__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def cnn__c2d_128_64_32_f3__f__dp__d_64_32_16__nn(input_shape, output_dim=1, activation='relu', dropout=0.5, out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """CNN_{128,64,32}{3}_Dense_{64,32,16}"""
    name = 'CNN_{128,64,32}{3}_Dense_{64,32,16}'
    cnn__c2d_128_64_32_f3__f__dp__d_64_32_16__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(128, (3, 3), padding='same', activation=activation))
    model.add(Conv2D( 64, (3, 3), padding='same', activation=activation))
    model.add(Conv2D( 32, (3, 3), padding='same', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def cnn__c2d_32_f1__f__dp__d_64_32_16__nn(input_shape, output_dim=1, activation='relu', dropout=0.5, out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """CNN_{32}{1}_Dense_{64,32,16}"""
    name = 'CNN_{32}{1}_Dense_{64,32,16}'
    cnn__c2d_32_f1__f__dp__d_64_32_16__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (1, 1), padding='same', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name


def cnn__c2d_128_64_32_f1__f__dp__d_64_32_16__nn(input_shape, output_dim=1, activation='relu', dropout=0.5, out_activation='linear', loss='mean_absolute_error', opt='adam'):
    """CNN_{128,64,32}{1}_Dense_{64,32,16}"""
    name = 'CNN_{128,64,32}{1}_Dense_{64,32,16}'
    cnn__c2d_128_64_32_f1__f__dp__d_64_32_16__nn.__annotations__['name'] = name
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(128, (1, 1), padding='same', activation=activation))
    model.add(Conv2D( 64, (1, 1), padding='same', activation=activation))
    model.add(Conv2D( 32, (1, 1), padding='same', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(output_dim, activation=out_activation))
    model.compile(loss=loss, optimizer=opt)
    return model, name



def neural_architecture_search(indim=None, outdim=None, max_trials=10, overwrite=True, *args):
    """nas"""
    name = 'nas'
    model = ak.StructuredDataRegressor(max_trials=max_trials, overwrite=overwrite)
    return [model, name]





def get_loss_function(loss_name):
    return loss_name
