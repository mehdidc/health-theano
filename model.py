from lasagne.easy import SimpleNeuralNet

def model(X_train, y_train, X_test):
    # replace Nans with -1
    X_train[np.isnan(X_train)] = -1
    X_test[np.isnan(X_test)] = -1

    # Fit and predict
    nnet = SimpleNeuralNet(nb_hidden_list=[100], max_nb_epochs=100, batch_size=100, learning_rate=1.)
    nnet.fit(X_train, y_train)
    return nnet.predict(X_test)
