#!/usr/bin/env python3
'''IMPORTANT
#!! this kind of comment demarks things that can be implemented
    it isnt a real comment of the program
'''
import sys
import os
import cv2
import pickle
import numpy as np

def load_object(file_name):
    with open(file_name, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def save_object(object,file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)

if __name__ == "__main__":

    NXXLab = load_object("Names_X_XLab_NumPxls_50x100")
    X = NXXLab[1]


    for i in range(0,len(X)):

        tot = np.sum(X[i][0:len(X[i])-2])
        #print(len(X[i][0:len(X[i])-2]))
        val = int((tot/5000)/1.3)
        #bol = False
        if val > 130:
            val = val + 40
            #bol = True
        for j in range(0,len(X[i])):
            if j == len(X[i]) - 1:
                break
            if X[i][j] < val:
                X[i][j] = 255.0
            else:
                X[i][j] = 0.0

        #print(X[i])

    save_object(X, 'XstandarizedToBNWidth')


