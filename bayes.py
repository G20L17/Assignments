#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):
        # Your code goes here.
        alpha=self._smooth
        temp=[]
        temp.append(np.unique(y))
        self._Ncls.append(temp[0].size)
        self._Nfeat.append(X[0].size)
        dict_cls={}
        dict_cls_feat={}

        for i in range(y.size):
            if y[i] in dict_cls_feat:
                continue
            else:
                dict_cls_feat[y[i]]=[0 for j in range(X[i].size)]


        for i in range(y.size):
            if y[i] in dict_cls:
                dict_cls[y[i]]+=1
            else:
                dict_cls[y[i]]=1
            for j in range(X[i].size):
                dict_cls_feat[y[i]][j]+=X[i][j]

        for k in dict_cls_feat:
            Ni=dict_cls[k]+alpha
            N=y.size+self._Ncls[0]*alpha
            self._class_prob.append(Ni/float(N))
            ar=np.array([])
            for j in range(X[0].size):
                Nj=dict_cls_feat[k][j]+alpha
                Ni=dict_cls[k]+2*alpha
                ar=np.append(ar, (Nj/float(Ni)))
            self._feat_prob.append(ar)
        return 

    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        Y_pre=np.array([])
        for i in X:
            min_neg_prob=1e12
            for k in range(self._Ncls[0]):
                neg_log_prob=-np.log(self._class_prob[k])
                for j in range(self._Nfeat[0]):
                    if i[j]==0:
                        neg_log_prob-=np.log(1-self._feat_prob[k][j])
                    else:
                        neg_log_prob-=np.log(self._feat_prob[k][j])

                if neg_log_prob<min_neg_prob:
                    cls=k
                    min_neg_prob=neg_log_prob
            Y_pre=np.append(Y_pre, cls)

        return Y_pre

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        temp=[]
        temp.append(np.unique(y))
        self._Ncls.append(temp[0].size)
        self._Nfeat.append(X[0].size)
        dict_cls={}
        dict_cls_feat={}

        for i in range(y.size):
            if y[i] in dict_cls_feat:
                continue
            else:
                dict_cls_feat[y[i]]=[0 for j in range(X[i].size)]

        for i in range(y.size):
            if y[i] in dict_cls:
                dict_cls[y[i]]+=1
            else:
                dict_cls[y[i]]=1
            for j in range(X[i].size):
                dict_cls_feat[y[i]][j]+=X[i][j]

        self._class_prob.append(dict_cls)
        self._feat_prob.append(dict_cls_feat)
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        Y_pred=np.array([])
        N=0
        for i in self._class_prob[0]:
            N+=self._class_prob[0][i]

        for i in X:
            min_neg_log_prob = 1e12
            cls = 0

            for k in self._feat_prob[0]:
                Ny = sum(self._feat_prob[0][k])
                neg_log_prob = -np.log(
                    (self._class_prob[0][k] + 1) / float(N + (self._Ncls[0] * self._smooth)))
                for j in range(self._Nfeat[0]):
                    if (i[j]) == 0:
                        continue
                    for itere in range(i[j]):
                        num = (self._smooth + self._feat_prob[0][k][j])
                        din = (Ny + (self._Nfeat[0] * self._smooth))
                        neg_log_prob -= np.log(num / float(din))

                if neg_log_prob<min_neg_log_prob:
                    cls = k
                    min_neg_log_prob = neg_log_prob

            Y_pred = np.append(Y_pred, cls)
        return Y_pred
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
print('Q3.1')

vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

print ('For Bernoulli NB:')
ta = time()
alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
tb = time()
print ('alpha=%f, accuracy=%f, time= %f' %(alpha, np.mean((y_test-y_pred)==0), tb-ta))


print ('Q3.2')
acc = []
alp = []
for alpha in [float(j) / 100 for j in range(1, 101, 1)]:
    ta = time()
    clf =MyBayesClassifier(smooth=alpha)
    clf.train(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc.append(np.mean((y_test-y_pred)==0))
    alp.append(alpha)
    tb = time()
    print ('alpha=%f, accuracy=%f, time= %f' % (alpha, np.mean((y_test - y_pred) == 0), tb - ta))

with PdfPages('Q3.2 plot.pdf') as pdf:
    pl.plot(alp,acc,'go')
    pl.ylabel('Accuracy')
    pl.xlabel('Alpha')
    pl.title('Q3.2 Alpha Vs Accuracy (Bernoulli NB)')
    pdf.savefig()
    pl.close()
print ("The maximum accuracy For the bernoulli NB is: " + str(max(acc)))
print ("The corresponding value for alpha is: " + str(alp[(acc.index(max(acc)))]))

print ('Q3.3')
vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

print ('For Multinomial NB')
acc = []
alp = []
for alpha in [float(j) / 100 for j in range(1, 101, 1)]:
    ta = time()
    clf1 =MyMultinomialBayesClassifier(smooth=alpha)
    clf1.train(X_train,y_train)
    y_pred1 = clf1.predict(X_test)
    acc.append(np.mean((y_test-y_pred1)==0))
    alp.append(alpha)
    tb = time()
    print ('alpha=%f, accuracy=%f, time= %f' % (alpha, np.mean((y_test - y_pred1) == 0), tb - ta))

with PdfPages('Q3.3 plot.pdf') as pdf:
    pl.plot(alp,acc,'go')
    pl.ylabel('Accuracy')
    pl.xlabel('Alpha')
    pl.title('Alpha Vs Accuracy (Multinomial NB)')
    pdf.savefig()
    pl.close()
print ("The maximum accuracy For the Multinomial NB is: " + str(max(acc)))
print ("The corresponding value for alpha is: " + str(alp[(acc.index(max(acc)))]))
