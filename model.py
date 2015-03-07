from lasagne.easy import SimpleNeuralNet
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class EnsembleEstimators(object):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        print "fitting"
        self.model.fit(X, y)
        return self

    def transform(self, X):
        X = np.concatenate([e.predict(X)[:, np.newaxis] for e in self.model.estimators_], axis=1)
        return X.astype(np.float32)

def model(X_train, y_train, X_test):
    # replace Nans with -1
    X_train[np.isnan(X_train)] = -1
    X_test[np.isnan(X_test)] = -1
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)

    model = Pipeline( [('rf', EnsembleEstimators(RandomForestClassifier(n_estimators=200, max_depth=3, n_jobs=-1))),
                       ('neuralnet', SimpleNeuralNet(nb_hidden_list=[100], max_nb_epochs=20, batch_size=100, learning_rate=1.))
    ])
    # Fit and predict
    model.fit(X_train, y_train)
    print float(np.sum(model.predict(X_train)==y_train)) / len(X_train)
    return model.predict(X_test), model.decision_function(X_test)

if __name__ == "__main__":
    train = pd.read_csv("input/train.csv").values
    test = pd.read_csv("input/test.csv").values
    test = test[:, 1:]
    X = train[:, 1:]
    y = train[:, 0]

    targets, proba = model(X, y, test)
    #print targets.shape
    #print proba.shape
