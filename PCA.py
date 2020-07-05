#!/usr/bin/env python3

import sys
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

def load_object(file_name):
    with open(file_name, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def save_object(object,file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)

'''
input -> np.ndarray, np.ndarray, int
output -> np.ndarray, np.ndarray, np.ndarray, np.ndarray, int

it gets the matrix with all the samples, a matrix with the labeling and
the proportion that we want for training and test
and it returns the 2 partitions of samples and labels and the index that
separes the training and the test
'''


#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

if __name__ == "__main__":

    try:
        numcomp = int(sys.argv[1])
    except:
        numcomp = 1000
        print('need number of components after PCA')
    BD = load_object("BD_50x100")
    NXXLab = load_object("Names_X_XLab_NumPxls_50x100")
    Names = NXXLab[0]
    '''
    X and XLab must be converted to np.array float 32 because at NXXLab
    were saved as list
    '''
    #X = np.array(NXXLab[1], dtype = np.float32)
    XLab = np.array(NXXLab[2], dtype = np.float32)

    last2items = []
    print('starting splitting of the array')
    X = NXXLab[1]

    print('saving the last 2 items of the all the colums in X, actual length of X', len(X),len(X[0]))
    total = len(X)
    for i in range(0, len(X)):
        aux = []
        lastItem = list(X[i]).pop(len(X[i]) - 1)
        aux.append(np.array(lastItem, dtype = np.float32))
        lastItem = list(X[i]).pop(len(X[i]) - 2)
        aux.append(np.array(lastItem, dtype = np.float32))
        X[i] = X[i][0:len(X[i])-2]
        last2items.append(aux)
    print('Modified X', len(X), len(X[0]))
    print('Doing pca')
    X = np.array(NXXLab[1], dtype = np.float32)
    pca = decomposition.PCA(n_components='mle')
    pca.fit(X)
    X = pca.transform(X)
    print()
    print(X.shape)
    print()
    print('X with PCA', len(X), len(X[0]))

    print('Adding the 2 last items to all the points at the matrix')
    AuxList = []
    for i in range(0, len(X)):
        AuxList.append(np.append(X[i], last2items[i]))
    X = np.array(AuxList, dtype = np.float32)
    print('X with PCA and the last 2 items appended', len(X), len(X[0]))
    current = 0
    print('starting plotting')
    '''
    for i in range(0, len(X)):
        if XLab[i] == 1.0:
            plt.plot(X[i],'rs')
        if XLab[i] == 0.0:
            plt.plot(X[i],'b^')
        current = current + 1
        print(str(round(current*100/total,2)))
    print('saving png PCA', str(numcomp))
    name = 'PCA' + str(numcomp) + '.png'
    plt.savefig(name)
    print('plot ended')
    nameobj = "Names_X_XLab_NumPxls_50x100_PCA_" + str(numcomp)
    newtup = (NXXLab[0],X,NXXLab[2], NXXLab[3])
    print('saving pickle object')
    save_object(newtup,nameobj)
    '''
    