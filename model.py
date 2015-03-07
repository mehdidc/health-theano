from lasagne.easy import SimpleNeuralNet
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def model(X_train, y_train, X_test):
    # replace Nans with -1
    X_train[np.isnan(X_train)] = -1
    X_test[np.isnan(X_test)] = -1


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)

    # Fit and predict
    nnet = SimpleNeuralNet(nb_hidden_list=[100], max_nb_epochs=20, batch_size=100, learning_rate=1.)
    nnet.fit(X_train, y_train)
    print float(np.sum(nnet.predict(X_train)==y_train)) / len(X_train)
    return nnet.predict(X_test), nnet.decision_function(X_test)

if __name__ == "__main__":
    train = pd.read_csv("input/train.csv").values
    test = pd.read_csv("input/test.csv").values
    test = test[:, 1:]
    X = train[:, 1:]
    y = train[:, 0]

    targets, proba = model(X, y, test)

    #print targets.shape
    #print proba.shape
