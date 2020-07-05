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
from matplotlib import pyplot as plt

def load_object(file_name):
    with open(file_name, 'rb') as fh:
        obj = pickle.load(fh)
    return obj


'''
recreateImg()
input -> np.ndarray
return -> np.ndarray
This method convert to a 1xn matrix to a mxn matrix
where the 1xn matrix is the frame of the img on a line and the 
mxn matrix is the proper img to write it
'''
def recreateImg(obj):

    FrmX = list(obj)
    #remove 2 last items 

    a = FrmX.pop(len(obj) - 1)
    b = FrmX.pop(len(obj) - 2)
    print(a,b)
    NewObj = []
    for i in range(0,len(FrmX),100):
        to = i+100
        NewObj.append(FrmX[i : to])
    return np.array(NewObj, dtype = np.float32)

if __name__ == "__main__":

    All = load_object('Names_X_XLab_NumPxls_50x100')
    X = load_object('XstandarizedToBN')
    Names = All[0]
    ind = Names.index('43_44_18.png')
    cv2.imwrite('prueba.png', recreateImg(X[ind]))

    