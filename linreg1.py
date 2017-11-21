#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegressor():
    def __init__(self, kappa=0.01, lamb=0, max_iter=200, opt='batch'):
        self._kappa = kappa
        self._lamb = lamb
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        error = []
        if self._opt == 'sgd':
            error = self.__stochastic_gradient_descent(X, y)
        elif self._opt == 'batch':
            error = self.__batch_gradient_descent(X, y)
        else:
            print 'unknown opt'
        return error

    def predict(self, X):
        return np.dot(X, self._w)

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        cost_vector = np.zeros(self._max_iter)
        self._w = np.ones(X.shape[1])

        for i in range(self._max_iter):
            y_hat = X.dot(self._w)
            diff = y - y_hat
            gradient = (1.0/N) * np.transpose(X).dot(diff)
            self._w = self._w + (self._kappa * gradient)
            sse = 0
            for e in diff:
                sse += float(e ** 2) / N
            cost_vector[i] = sse

        plt.plot(cost_vector)
        plt.show()
        return cost_vector

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        cost_vector = np.zeros(N)
        self._w = np.ones(X.shape[1])

        for epoch in range(N):
            epoch_loss = np.zeros(self._max_iter)
            for i in range(self._max_iter):
                Xi = X[i % N]
                y_hat = Xi.dot(self._w)
                diff = y[i % N] - y_hat
                gradient = (1.0/N) * np.transpose(Xi).dot(diff)
                self._w = self._w + (self._kappa * gradient)
                epoch_loss[i] = diff ** 2
            cost_vector[epoch] = np.average(epoch_loss)

        plt.plot(cost_vector)
        plt.show()
        return cost_vector

    def __total_error(self, X, y, w):
        tl = 0.5 * np.sum((np.dot(X, w) - y)**2)/len(y)
        return tl

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    data = load_boston()
    X, y = data['data'], data['target']
    mylinreg = MyLinearRegressor(max_iter=100, opt='sgd')
    mylinreg.fit(X, y)
