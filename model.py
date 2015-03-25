from sklearn.base import BaseEstimator
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano


class Classifier(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
            ('imputer', Imputer()),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[50],
                                          max_nb_epochs=100,
                                          batch_size=100,
                                          learning_rate=0.9)),
        ])

    def __getattr__(self, attrname):
        return getattr(self.clf, attrname)

    def fit(self, X, y):
        X = X.astype(theano.config.floatX)
        y = y.astype(np.int32)
        return self.clf.fit(X, y)

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict_proba(X)
