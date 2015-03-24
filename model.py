from sklearn.base import BaseEstimator
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import theano

class Classifier(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
            ('imputer', Imputer()),
            ('rf_outputs', EnsembleEstimatorTransformer(RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1))),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[100, 100],
                                          max_nb_epochs=100,
                                          batch_size=100,
                                          learning_rate=0.8)),
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
        return self.clf.decision_function(X)


class EnsembleEstimatorTransformer(object):

    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model

    def fit(self, X, y):
        self.ensemble_model.fit(X, y)
        return self

    def transform(self, X):
        outputs = np.concatenate([e.predict(X)[:, np.newaxis]
                                  for e in self.ensemble_model.estimators_], axis=1)
        outputs = outputs.astype(theano.config.floatX)
        return outputs
