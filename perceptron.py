#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sklearn.linear_model as sklinear
from sklearn.svm import LinearSVC

np.random.seed(666)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_boundary(clf, X_train, Y_train, xx, yy):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train,
                cmap=plt.cm.coolwarm,
                edgecolors='k')
    plt.show()


class Perceptron():

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = np.array([[5., 0., 5.]])

    def fit_epoch(self, X, Y, n_epochs):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        shuff = np.random.permutation(len(Y))
        X, Y = X[shuff], Y[shuff]
        N, M = X.shape
        Y=Y.reshape(N, 1)
        error = []
        for i in range(n_epochs):
            for j in range(N):
                deltax=(np.sign(np.dot(X[j],self.weights.T)) - Y[j])/2
                self.weights-= self.lr*deltax * X[j]
            y_pred=np.sign(np.dot(X,self.weights.T))
            delta=(y_pred - Y)/2
            err = np.sum(np.dot(delta.T, np.dot(X, self.weights.T)))
            error.append(err)
        return error

    def predict(self, X):
        asd = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sign(np.dot(asd, self.weights.T))


if __name__ == '__main__':
    N, M = 40, 2
    X_train = np.r_[np.random.randn(N, M) + [1, 1], np.random.randn(N, M) + [10, 10]]
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    Y_train = np.array([1]*N + [-1]*N)

    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])

    #3.1
    n_epochs = 25
    clf = Perceptron(learning_rate=0.001)
    error=clf.fit_epoch(X_train, Y_train, n_epochs)
    plot_boundary(clf, X_train, Y_train, xx, yy)
    iter_num=5
    plt.plot(np.arange(iter_num), error[0:iter_num],marker='o',linestyle='--', label ='learn_rate =0.001')
    plt.ylabel('Error')
    plt.xlabel('Number of Iterations')
    plt.show()

    #3.2
    learning_rate=[0.0001, 0.01, 0.1]
    for lr in learning_rate:
        clf=Perceptron(learning_rate=lr)
        error = clf.fit_epoch(X_train, Y_train, n_epochs)
        print(error)
        plot_boundary(clf, X_train, Y_train, xx, yy)
        plt.plot(np.arange(n_epochs), error, marker='o', linestyle='--', label='learn_rate ='+str(lr))
        plt.ylabel('Error')
        plt.xlabel('Number of Iterations')
        plt.show()

    #3.3
    clf_P=sklinear.Perceptron(max_iter=n_epochs)
    clf_P=clf_P.fit(X_train, Y_train)
    plot_boundary(clf_P, X_train, Y_train, xx, yy)
    clf_S=LinearSVC(max_iter=n_epochs)
    clf_S.fit(X_train, Y_train)
    plot_boundary(clf_S, X_train, Y_train, xx, yy)

    #3 boundaries
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ZP= clf_P.predict(np.c_[xx.ravel(), yy.ravel()])
    ZP = ZP.reshape(xx.shape)
    ZS = clf_S.predict(np.c_[xx.ravel(), yy.ravel()])
    ZS = ZS.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired, linestyles='dotted', label='0.1 learning rate boundary')
    plt.contour(xx, yy, ZP, cmap=plt.cm.Paired, linestyles='dashed', label='sklearn perceptron boundary')
    plt.contour(xx, yy, ZS, cmap=plt.cm.Paired, linestyles='solid', label='linearSVM boundary')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train,
                cmap=plt.cm.coolwarm,
                edgecolors='k')
    plt.show()


