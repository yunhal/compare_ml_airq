import datetime
import pandas as pd 
import numpy as np

import sklearn.preprocessing


def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset)-lookback):
        dataX.append(dataset[i:(i+lookback), :])
        dataY.append(dataset[i + lookback, 0])
    return np.array(dataX), np.array(dataY)


def create_dataset_x_y_cls(dataset, clscols, lookback):
    dataX, dataY,  = [], []
    classlabels = []
    for i in range(len(dataset)-lookback):
        dataX.append(dataset[i:(i+lookback), :])
        dataY.append(dataset[i + lookback, 0])
        labels = [dataset[i + lookback - 2, ci] for ci in clscols] + [dataset[i + lookback - 1, ci] for ci in clscols]
        classlabels.append(labels)
    return np.array(dataX), np.array(dataY), np.array(classlabels)


def generate_batch_gminmax(data, sep_year, lookback):
    training_section = data.loc[data.index < datetime.datetime(sep_year,1,1,0,0,0)]
    ns, _ = training_section.shape

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training_section.values)
    scaled = scaler.transform(data.values) 

    train_size, ncol = training_section.shape
    total_size, ncol = data.shape
    test_size = total_size - train_size
    
    train, test = scaled[0:train_size+lookback, :], scaled[train_size:total_size, :]
    train_t, test_t = data.index[lookback:train_size+lookback], data.index[train_size+lookback:total_size]

    X_train, y_train = create_dataset(train, lookback)
    X_test, y_test = create_dataset(test, lookback)

    return X_train, y_train, X_test, y_test, scaler, train_t, test_t


def generate_train_val_test_batch(data, scale_cols, cls_cols, sep_val_year, sep_test_year, lookback):
    train_section = data.loc[data.index < datetime.datetime(sep_val_year,1,1,0,0,0)]
    val_section = data.loc[(data.index >= datetime.datetime(sep_val_year,1,1,0,0,0)) & (data.index < datetime.datetime(sep_test_year,1,1,0,0,0))]
    test_section = data.loc[data.index >= datetime.datetime(sep_test_year,1,1,0,0,0)]
    nstrain, _ = train_section.shape
    nsval, _ = val_section.shape
    nstest, _ = test_section.shape
    nstotal, _ = data.shape

    data_matrix = data.values

    #scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(train_section.get(scale_cols).values)
    #scaled = scaler.transform(data.get(scale_cols).values)
    #scaled_data = data.copy()
    #scaled_data.loc[:, scale_cols] = scaled
    #scaled_data = scaled_data.values
    
    train, val, test = data_matrix[0:nstrain+lookback, :], data_matrix[nstrain:nstrain+nsval+lookback, :], data_matrix[nstrain+nsval:nstotal, :]
    train_t, val_t, test_t = data.index[lookback:nstrain+lookback], data.index[nstrain+lookback:nstrain+nsval+lookback], data.index[nstrain+nsval+lookback:nstotal]

    colslist = list(data)
    cls_col_index = []
    for cv in cls_cols:
        ci = colslist.index(cv)
        cls_col_index.append(ci)

    X_train, y_train, c_train = create_dataset_x_y_cls(train, cls_col_index, lookback)
    X_val, y_val, c_val = create_dataset_x_y_cls(val, cls_col_index, lookback)
    X_test, y_test, c_test = create_dataset_x_y_cls(test, cls_col_index, lookback)

    return X_train, y_train, c_train, X_val, y_val, c_val, X_test, y_test, c_test, train_t, val_t, test_t



def inverse_scale(prdt, X, scaler):
    data_sample = X[:,0,:]
    data_sample[:,0] = prdt
    sample_inverse = scaler.inverse_transform(data_sample)
    return sample_inverse


def inverse_train_test_predictions(trainPredict, X_train, y_train, testPredict, X_test, y_test, scaler):
    input_truth_train  = inverse_scale(y_train, X_train, scaler)[:,0]
    output_prdct_train = inverse_scale(trainPredict.flatten(), X_train, scaler)[:,0]
    
    input_truth_test  = inverse_scale(y_test, X_test, scaler)[:,0]
    output_prdct_test = inverse_scale(testPredict.flatten(), X_test, scaler)[:,0]

    return input_truth_train, output_prdct_train, input_truth_test, output_prdct_test


def format_result(train_t, test_t, input_truth_train, output_prdct_train, input_truth_test, output_prdct_test, dataset, df):
    time_index = train_t.tolist() + test_t.tolist()
    dfresample = df.reindex(dataset.index)
    cols = list(df)
    target = cols[0]
    # format all data
    input_truth  = input_truth_train.tolist()  + input_truth_test.tolist()
    output_prdct = output_prdct_train.tolist() + output_prdct_test.tolist()
    result_df = pd.DataFrame({
        'date':time_index,
        'truth':dfresample.loc[time_index, target],
        'input':np.array(input_truth),
        'prd':np.array(output_prdct),
    })

    result_df = result_df.set_index(['date'])
    result_df_dropna = result_df.dropna()

    prdt_df = result_df.loc[test_t, :]
    prdt_df_dropna = prdt_df.dropna()

    return result_df, result_df_dropna, prdt_df, prdt_df_dropna