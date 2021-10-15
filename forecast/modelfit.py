import numpy
import pandas
import sklearn.metrics as skmetrics

def train(model, X_train, y_train, epochs, batch_size, verbose, train_func=None, threshold=0.02):
    res = None
    if train_func is None:
        if isinstance(model, list):
            mname = model[1]
            ns = X_train.shape[0]
            nf = 1
            for i in X_train.shape[1:]:
                nf = nf * i
            X = X_train.reshape((ns, nf))
            if mname == 'nas':
                model[0].fit(X, y_train, epochs)
            else:
                model[0].fit(X, y_train)
        else:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
    else:
        if isinstance(model, list):
            mname = model[1]
            if mname == '2randomforest':
                res = train_func(model, X_train, y_train, threshold=0.02)
            else:
                res = train_func(model, X_train, y_train, epochs) # For DNN models
        else:
            res = train_func(model, X_train, y_train, epochs)

    return res





def predict(model, X_test, predict_func=None):
    if predict_func is None:
        if isinstance(model, list):
            ns = X_test.shape[0]
            nf = 1
            for i in X_test.shape[1:]:
                nf = nf * i
            X = X_test.reshape((ns, nf))
            testPredict = model[0].predict(X)
            return testPredict
        else:
            model.reset_states()
            testPredict = model.predict(X_test)
            return testPredict
    else:
        return predict_func(model, X_test)



def permuted_predictions_and_importance(model, x, y, index, feature_names, predict_func=None):
    prd = predict(model, x, predict_func=predict_func)
    if len(prd.shape) > 1:
        prd = prd.flatten()

    mae = skmetrics.mean_absolute_error(y, prd)

    prd_permuted = {'index':index}
    mae_permuted = {'train':mae}
    importance = {}

    for i,fn in enumerate(feature_names):
        xi_o = x[:, i]
        xi_p = x[:, i]
        numpy.random.shuffle(xi_p)
        x_p_i = x.copy()
        x_p_i[:, i] = xi_p
        prdi = predict(model, x_p_i, predict_func=predict_func)
        if len(prdi.shape) > 1:
            prdi = prdi.flatten()

        maei = skmetrics.mean_absolute_error(y, prdi)
        score = maei - mae

        prd_permuted[fn] = prdi
        mae_permuted[fn] = maei
        importance[fn] = score

    df_permute_prd = pandas.DataFrame(prd_permuted)
    df_permute_prd = df_permute_prd.set_index('index')

    return df_permute_prd, mae_permuted, importance
