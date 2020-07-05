#!/usr/bin/env python3

import sys
import os
import cv2
import pickle
import numpy as np
import time
import json
import matplotlib.pyplot as plt


#Those methods are used to create and load files which contains imgs
def load_object(file_name):
    with open(file_name, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def save_object(object,file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)


'''
BD is a dictionary that has all our images in the way it follows:

    {key:[value1, value2]} where:
    key -> Name of the Frame of a png in the format exposed at (1)
    value1 -> a np.narray of the img where each pos is the rgb value of the pixel
    value2 -> Layer, 0 for non splitted word, 1 if is a splitted word

    Exists a exception, there is a key that his key is "totalFrames" where the 
    value is the total frames of our bd

    Names is a list with the names of all the Frames
    X are the frames as np.array in 1 line in np.narray format and float32
    Y are the labels as np.array in float32

    The index of the three lists are the same for each lists
    for example if name_x_y.png is in index 100, is np.narray and is label is
    on index 100 in his respective list (X and XLab)
'''

BD = {}
Names = []
X = []
XLab = []
NumPxls = []

'''
oneLineMatrix()
input -> List[[]] mxn
return -> List[[]] 1x(mxn)

Puts all the rows of a matrix (m) in one line to get
a 1xn

This method agretates at the end of tbe matrix the relative height and width
of the frame(mat) at the properly image to know his respective position at it
'''
def oneLineMatrix(mat, name):
    name = name.split('.')
    name = name[0].split('_')
    h = int(name[1])
    w = int(name[2])
    oneLine = []
    for i in mat:
        oneLine.extend(i)
    oneLine.append(h)
    oneLine.append(w)
    return oneLine

'''
Layering()
input -> np.ndarray, np.ndarray, string
return -> none
rewrites the global variables BD, Names, X, XLab

This method gets a frame in color and a frame in black white,
with the frame in color, it sees the label, and asign the label
to the black white Frame
'''
#Green colours to identify the labels
UpperGreen = np.array([0, 255, 0],dtype="uint8")
LowerGreen = np.array([0, 200, 0],dtype="uint8")
'''
Testing for what range of colours we see, we create 2 green png 100x100

up = []
low = []
for i in range(100):
    aux1 = []
    aux2 = []
    for j in range(100):
        aux1.append(UpperGreen)
        aux2.append(LowerGreen)
    up.append(aux1)
    low.append(aux2)

up = np.array(up,dtype=np.float32)
low = np.array(low,dtype=np.float32)
cv2.imwrite('up.png', up)
cv2.imwrite('low.png', low)
'''
def Layering(imgEtiq, imgBN, name):
    
    
    #mask will have the number of pixels that contains green (the color of the layer)
    mask = cv2.inRange(imgEtiq, LowerGreen, UpperGreen)
    result = np.count_nonzero(mask)
    #!!findContours maybe it can be used

    #ALWAYS append the Frm on X
    X.append(np.array(oneLineMatrix(imgBN, name) ,dtype = np.float32))
    #!!We can make the window of the image smaller btw its a bad idea
    #150 is a best mark to use
    if result >= 70: #and name not in Lista
        #Lista.append(nameaux)
        BD[name] = [np.array(imgBN, dtype=np.float32),1]
        XLab.append(np.array(1 ,dtype = np.float32))
    else:
        BD[name] = [np.array(imgBN, dtype=np.float32),0]
        XLab.append(np.array(0 ,dtype = np.float32))
    NumPxls.append(result)
    Names.append(name)

    #print(name, " procesed")

'''
FillShortImg()
input -> List, int, int
return -> List
FillShortImg fills the Frames that are smaller than the normal size of
the frames, which is 50x100
It returns a list that is the img with the standard size
'''

def FillShortImg(mat, h, w):

    if len(mat) == h and len(mat[0]) == w:
        return mat
    mat = np.array(mat)
    summatory = np.sum(mat)
    mat = list(mat)
    try:
        thish = len(mat)
        thisw = len(mat[0])
    except:
        return
    newColor = int(summatory / (thish*thisw))
    toAppend = [newColor] * w
    toExtend = [newColor] * (w - thisw)
    if thish < h and thisw == w:
        #toAppend = [newColor] * w
        for i in range(h - thish):
            mat.append(toAppend)

    elif thish == h and thisw < w:
        #toExtend = [newColor] * (w - thisw)
        for i in range(h):
            mat[i] = list(mat[i])
            mat[i].extend(toExtend)

    else:
        #toExtend = [newColor] * (w - thisw)
        for i in range(thish):
            mat[i] = list(mat[i])
            mat[i].extend(toExtend)
        #toAppend = [newColor] * w
        for i in range(h - thish):
            mat.append(toAppend)
    
    return mat

def TravelingAndLayeringOfBd():
    print("running the creation of all the samples")
    timeIni = time.time()
    for Folder, _, Files in os.walk("../Data/bdImg"):

        TotalImg = len(Files)
        CurrImg = 0
        TotalFrames = 0 #The number of total frames of the imgs to know the % of correct classifications
        for png in Files:

            #Img to classify (White Black without labels)
            Img = cv2.imread(os.path.join(Folder, png))
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            #Img to train (Color with labels)
            ImgEtiq = cv2.imread(os.path.join("../Data/bdImgEtiq" , png))
            #Dimensions
            dimensions = Img.shape
            Height = Img.shape[0]
            Width = Img.shape[1]

            CurrH = 0   #start height 0 pixel -> right top corner
            CurrFil = 0 #This will be used to acceed to the pixels at the moment with the name
            
            
            while CurrH <= Height:

                CurrW = 0
                FrmH = CurrH + 50 #Heigth of the frame of the window
                if FrmH >= Height:
                    FrmH = Height #!!boundingRect can be used
                CurrCol = 0 #The same as CurrFil but with the current column
                while CurrW <= Width:
                    TotalFrames = TotalFrames + 1 #de 1 a N
                    '''
                    (1)
                    We save the frame as name_x_y.png where x is the Height of the frame 
                    and y is the Width of the frame, it will be usefull to know the place of the 
                    frame at the imaged known as name
                    '''
                    FrameName = png.split(".")
                    FrameName[0] = FrameName[0] + "_" + str(CurrFil) + "_" + str(CurrCol)
                    FrameName = ".".join(FrameName)
                    #FrameName = imgStr
                    FrmW = CurrW + 100
                    if FrmW >= Width:
                        ImgFrm = FillShortImg(Img[CurrH:FrmH, CurrW: Width], 50, 100) #quitted the conversion np.array at Img
                        Layering(ImgEtiq[CurrH:FrmH, CurrW: Width],ImgFrm, FrameName)
                        break
                    if FrmH == Height:
                        ImgFrm = FillShortImg(Img[CurrH:FrmH, CurrW: FrmW], 50, 100) #quitted the conversion np.array at Img
                        Layering(ImgEtiq[CurrH:FrmH, CurrW: FrmW], ImgFrm, FrameName)
                    else:
                        Layering(ImgEtiq[CurrH:FrmH, CurrW: FrmW], Img[CurrH:FrmH, CurrW: FrmW], FrameName)

                    CurrW = CurrW + 80
                    CurrCol = CurrCol + 1


                if FrmH == Height:
                    break
                CurrH = CurrH + 40
                CurrFil = CurrFil + 1

            CurrImg = CurrImg + 1
            print("Current image processed:", png,"Total images processed", str(round((CurrImg*100)/TotalImg,2)) + "%")
    
    BD["TotalFrames"] = TotalFrames
    timeEnd = time.time()

    print("Total time for ", TotalImg, timeEnd - timeIni,str(round(int((timeEnd - timeIni)/60),2)) + ' seconds')

if __name__ == "__main__":

    TravelingAndLayeringOfBd()
    tup = (Names, X, XLab, NumPxls)
    save_object(BD, 'BD_50x100')
    save_object(tup, 'Names_X_XLab_NumPxls_50x100')
    print('BD and Names_X_XLab_NumPxls Created')
