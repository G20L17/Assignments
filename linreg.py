#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import random
from sklearn.utils import shuffle

errors ={}

class MyLinearRegressor():

    def __init__(self, kappa=0.01, lamb=0, max_iter=500, opt='batch'):
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
            print 'unknow opt'
        return error

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        self._w = np.ones(X.shape[1])
        for i in range(self._max_iter):
            # Find the Hypothesis
            hypothesis = np.dot(X,self._w)
            # calculate the Weights
            gradient = (np.dot(X.transpose(),(np.dot(X,self._w) - y)))/N
            # Update the Weights
            self._w = self._w - (self._kappa * gradient)
            
            # Compute the Error after update
            err = self.__total_error(X,y,self._w)
            error.append(err)
        return error

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        error = []
        #Choose an initial vector of weights
        self._w = np.ones(X.shape[1])
        for i in range(self._max_iter):
            # Iterate over each example
            X,y=shuffle(X,y,random_state=1234)
            for j in range(N):
                if self._lamb > 0:
                    self._w -= (self._kappa * ((np.dot(X[j],self._w) - y[j]) * X[j]) )
                    self._w[1:] -= self._kappa * self._lamb/N
                else:
                    self._w -= (self._kappa * ((np.dot(X[j],self._w) - y[j]) * X[j]) )
            #Compute the error
            err = self.__total_error(X,y,self._w)
            error.append(err)
        return error


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

with PdfPages('/Users/Dhanush/Desktop/BatchGradientDecent_Learning_rate.pdf') as pdf:
    mylinreg = MyLinearRegressor(0.001,0,500,'batch')
    error=mylinreg.fit(X, y)
    iter1 = [i for i in range(500)]
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='m',label ='learn_rate =0.001')
    mylinreg = MyLinearRegressor(0.01,0,500,'batch')
    error=mylinreg.fit(X, y)
    iter1 = [i for i in range(500)]
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='g',label ='learn_rate =0.01')
    mylinreg = MyLinearRegressor(0.1,0,500,'batch')
    error=mylinreg.fit(X, y)
    iter1 = [i for i in range(500)]
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='y',label ='learn_rate =0.1')
    mylinreg = MyLinearRegressor(0.0001,0,500,'batch')
    error=mylinreg.fit(X, y)
    iter1 = [i for i in range(500)]
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='r',label ='learn_rate = 0.0001')
    pl.ylabel('Error',color='r')
    pl.xlabel('Number of Iterations',color='r')
    pl.title('BatchGradientDecent_Error__vs_Learning_rate',color = 'b')
    pl.legend(bbox_to_anchor=(0.69, 0.70), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

with PdfPages('/Users/Dhanush/Desktop/SGD_Learning_rate.pdf') as pdf:
    pl.ylim(0,250)
    mylinreg = MyLinearRegressor(0.01,0,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)  
    pl.plot(iter1,error,marker='.',markersize =1,linewidth=2, linestyle='-', color='m',label ='learn_rate =0.01') 
    mylinreg = MyLinearRegressor(0.001,0,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)  
    pl.plot(iter1,error,marker='.',markersize = 1,linewidth=1, linestyle='-', color='g',label ='learn_rate =0.001') 
    mylinreg = MyLinearRegressor(0.0001,0,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)  
    pl.plot(iter1,error,marker='.',markersize = 1,linewidth=1, linestyle='-', color='r',label ='learn_rate =0.0001')
    pl.ylabel('Error',color='r')
    pl.xlabel('Number of Iterations',color='r')
    pl.title('stochasticGradientDecent_Error__vs_Learning_rate',color = 'b')
    pl.legend(bbox_to_anchor=(0.69, 0.70), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

with PdfPages('/Users/Dhanush/Desktop/SGD_Learning_rate_regularisation.pdf') as pdf:
    pl.ylim(0,250)
    mylinreg = MyLinearRegressor(0.01,10,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)   
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='m',label ='lambda =10')
    mylinreg = MyLinearRegressor(0.01,100,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)   
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='r',label ='lambda =100')
    mylinreg = MyLinearRegressor(0.01,1000,300,'sgd')
    iter1 = [i for i in range(300)]
    error=mylinreg.fit(X, y)   
    pl.plot(iter1,error,marker='.',markersize = 0.1,linewidth=1, linestyle='-', color='y',label ='lambda =1000')    
    pl.ylabel('Error',color='r')
    pl.xlabel('Number of iterations',color='r')
    pl.title('SGD_Error__vs_Iterations_with_Regularisation_study_kappa_0.01 ',color = 'b')
    pl.legend(bbox_to_anchor=(0.69, 0.70), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

"""
for i in range(1):
    pl.ylim(0,250)
    mylinreg = MyLinearRegressor(0.01,10,1,'sgd')
    iter1 = [i for i in range(X.shape[0])]
    error=mylinreg.fit(X, y)
    pl.plot(iter1,error,marker='.',markersize = 1,linewidth=0.4, linestyle='-', color='y',label ='lambda =10')
    mylinreg = MyLinearRegressor(0.01,100,1,'sgd')
    iter1 = [i for i in range(X.shape[0])]
    error=mylinreg.fit(X, y)
    pl.plot(iter1,error,marker='.',markersize = 1,linewidth=0.5, linestyle='-', color='g',label ='lambda =100') 
    mylinreg = MyLinearRegressor(0.01,1000,1,'sgd')
    iter1 = [i for i in range(X.shape[0])]
    error=mylinreg.fit(X, y)
    pl.plot(iter1,error,marker='.',markersize = 1,linewidth=0.4, linestyle='-', color='r',label ='lambda =1000')     
    pl.ylabel('Error',color='r')
    pl.xlabel('Number of Training Examples',color='r')
    pl.title('SGD_Error__vs_No of Training Examples with regularisation and rate =0.01',color = 'b')
    pl.legend(bbox_to_anchor=(0.69, 0.70), loc=2, borderaxespad=0.)
    pl.show()
"""
