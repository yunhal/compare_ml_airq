import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet, MultiTaskLasso, MultiTaskElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
try:
    from xgboost import XGBRegressor
except:
    print('xgboost is not installed')
    XGBRegressor = None



def linear_regression(*args):
    """linear"""
    name = 'linear'
    model = LinearRegression()
    return [model, name]


def sgd_regression(*args):
    """sgd"""
    name = 'sgd'
    model = SGDRegressor(*args)
    return [model, name]


def lasso_regression(*args):
    """lasso"""
    name = 'lasso'
    model = Lasso()
    return [model, name]


def elasticnet_regression(*args):
    """elasticnet"""
    name = 'elasticnet'
    model = ElasticNet()
    return [model, name]


def multitasklasso_regression(*args):
    """multitasklasso"""
    name = 'multitasklasso'
    model = MultiTaskLasso()
    return [model, name]


def multitaskelasticnet_regression(*args):
    """multitaskelasticnet"""
    name = 'multitaskelasticnet'
    model = MultiTaskElasticNet()
    return [model, name]


def decisiontree_regression(*args):
    """tree"""
    name = 'tree'
    model = DecisionTreeRegressor()
    return [model, name]


def randomforest_regression(*args):
    """randomforest"""
    name = 'randomforest'
    model = RandomForestRegressor()
    return [model, name]


def xgboost_regression(*args):
    """xgboost"""
    name = 'xgboost'
    model = XGBRegressor()
    return [model, name]


def knn_regression(*args):
    """knn"""
    name = 'knn'
    model = KNeighborsRegressor()
    return [model, name]


def mlp_regression(*args):
    """mlp"""
    name = 'mlp'
    model = MLPRegressor()
    return [model, name]


def svm_regression(*args):
    """svm"""
    name = 'svm'
    model = LinearSVR()
    return [model, name]


def twostage_randomforest(*args):
    """2randomforest"""
    name = '2randomforest'
    rf1 = RandomForestRegressor(n_estimators=100, max_depth=7,random_state=137)
    rf2 = RandomForestRegressor(n_estimators=200, max_depth=7,random_state=137)
    ln1 = LinearRegression(fit_intercept=False)
    ln2 = LinearRegression(fit_intercept=False)
    ln3 = LinearRegression(fit_intercept=False)
    return [[rf1,rf2,ln1,ln2,ln3], name]


def train_2randomforest(model, X_train, y_train, threshold=5):
    model[0][0].fit(X_train, y_train)
    d2_train = pd.DataFrame(X_train)
    d2_train['trth'] = y_train
    d2_train['pred'] = model[0][0].predict(X_train)
    d2_train['diff'] = abs(d2_train['pred'] - d2_train['trth'])
    d2_train = d2_train[d2_train['diff'] > threshold]
    X_trn = np.array(d2_train.drop(['trth','pred','diff'],axis=1))
    y_trn = y_train[d2_train.index]

    if (len(y_trn) > 0):
        model[0][1].fit(X_trn, y_trn)
    else:
        model[0][1] = model[0][0]
        print('train_2randomforest, RF1 is good enough')

    pred1 = model[0][0].predict(X_train)
    pred2 = model[0][1].predict(X_train)

    sep1 = np.percentile(pred1,33)
    sep2 = np.percentile(pred1,67)

    low_index = np.where(pred1<sep1)
    med_index = np.where((pred1<=sep2) & (pred1>=sep1))
    high_index = np.where(pred1>sep2)

    X_low = list(zip(pred1[low_index], pred2[low_index]))
    Y_low = y_train[low_index]
    model[0][2].fit(X_low, Y_low)
    low1 = model[0][2].coef_[0]
    low2 = model[0][2].coef_[1]

    X_med = list(zip(pred1[med_index], pred2[med_index]))
    Y_med = y_train[med_index]
    model[0][3].fit(X_med, Y_med)
    med1 = model[0][3].coef_[0]
    med2 = model[0][3].coef_[1]

    X_high = list(zip(pred1[high_index], pred2[high_index]))
    Y_high = y_train[high_index]
    model[0][4].fit(X_high, Y_high)
    high1 = model[0][4].coef_[0]
    high2 = model[0][4].coef_[1]

    model.append([sep1, sep2, low1, low2, med1, med2, high1, high2])

    return None



def predict_2randomforest(model, X_test):
    model_2rf, mname, model_coff = model
    rf1,rf2,ln1,ln2,ln3 = model_2rf
    sep1, sep2, low1, low2, med1, med2, high1, high2 = model_coff
    pred1 = rf1.predict(X_test)
    pred2 = rf2.predict(X_test)
    pred3 = pred2
    pred3[np.where(pred1<sep1)] = low1*pred1[np.where(pred1<sep1)]+low2*pred2[np.where(pred1<sep1)]
    pred3[np.where((pred1>=sep1)&(pred1<=sep2))] = med1*pred1[np.where((pred1>=sep1)&(pred1<=sep2))]+med2*pred2[np.where((pred1>=sep1)&(pred1<=sep2))]
    pred3[np.where(pred1>sep2)] = high1*pred1[np.where(pred1>sep2)]+high2*pred2[np.where(pred1>sep2)]
    pred3[np.where(pred3<0)] = pred1[np.where(pred3<0)]
    return pred3



def random_forest_embedding(X, D, n_estimators=5, random_state=0, max_depth=1):
    random_trees = RandomTreesEmbedding(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth).fit(X)
    X_sparse_embedding = random_trees.transform(D)
    X_sparse_embedding.toarray()
    return X_sparse_embedding



def get_feature_importance(model):
    importance = []
    if isinstance(model, list):
        mdl = model[0]
        name = model[1]
        #print(name, dir(mdl))
        if name == 'linear':
            importance = mdl.coef_
        elif name == 'sgd':
            importance = mdl.coef_
        elif name == 'lasso':
            importance = mdl.coef_
        elif name == 'elasticnet':
            importance = mdl.coef_
        elif name == 'knn':
            importance = None
        elif name == 'mlp':
            importance = mdl.coefs_[1].flatten()
        elif name == 'svm':
            importance = mdl.coef_
        elif name == 'tree':
            importance = mdl.feature_importances_
        elif name == 'randomforest':
            importance = mdl.feature_importances_
        elif name == 'xgboost':
            importance = mdl.feature_importances_
        else:
            importance = mdl.feature_importances_
        #print(name, importance)
    return importance
