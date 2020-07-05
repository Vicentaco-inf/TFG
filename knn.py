#!/usr/bin/env python3

import sys
import os
import cv2
import pickle
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix #for evaluation of the method

def load_object(file_name):
    with open(file_name, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def save_object(object,file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(object, fh)

def partitioning(X, XLab, porc):
    used = int(porc*len(X)/100)
    return X[0:used] , XLab[0:used], X[used:len(X)], XLab[used:len(XLab)], used

def TotalEvaluation(pred, XLabt, Names, parc, Xtr):

    TotalSamples = len(Xt) #All the samples for test
    AccSamples = 0 #Correct samples labeled
    TotalSp = 0 #Total splitted words
    AccSp = 0 #Correct splitted words labeled   
    TotalNoSp = 0 #Total non splitted words
    AccNoSp = 0 #Correct non splitted words labeled
    NamesOfOnes = []
    NamesOfSuccess = []
    TotalOnes = []

    for i in range(0, len(pred)):
        #Predict a sample
        #clas = neigh.predict(Xt[i].reshape(1,-1)) #.reshape(1,-1)

        #PREDICTED 1.O
        if pred[i] == 1.0:
            TotalOnes.append(Names[len(Xtr) + i])
        #If the sample has the same label than the Labeled

        #ACCEPTED
        if pred[i] == XLabt[i]:
            AccSamples = AccSamples + 1

        #Lets get the success labeling the splitted words
        #REAL 1.0
        if XLabt[i] == 1.0:
            NamesOfOnes.append(Names[len(Xtr) + i])
            #plt.plot(Xt[i],'r')
            TotalSp = TotalSp + 1
            #ACCEPTED 1.0
            if pred[i] == 1.0:
                AccSp = AccSp + 1
                NamesOfSuccess.append(Names[len(Xtr) + i])
        
        #The same for non-splitted words
        #REAL 0.0
        elif XLabt[i] == 0.0:
            #plt.plot(Xt[i],'g')
            TotalNoSp = TotalNoSp + 1
            #ACCEPTED 0.0
            if pred[i] == 0.0:
                AccNoSp = AccNoSp + 1


    print('With', str(parc)+'%', 'of test, we get a success of:', str(round((AccSamples*100)/TotalSamples,2)) + "%")
    print('Total samples labeled:', len(XLabt), 'with', len(Xtr), 'for training')
    print('Total splitted words:', TotalSp, 'success labeling them:', str(round((AccSp*100)/TotalSp,2)) + "%")
    print('Total no splitted words:', TotalNoSp, 'success labeling them:', str(round((AccNoSp*100)/TotalNoSp,2)) + "%")
    print()
    #print('Total splitted words')
    #print(NamesOfOnes, len(NamesOfOnes))
    #print()
    #print('Splitted words found')
    #print(NamesOfSuccess, len(NamesOfSuccess))
    #print()
    #print('predicted 1.0 labels')
    #print(TotalOnes, len(TotalOnes))
    #print()
    #print('Wrong 1.0 labeled')
    #wrong = list(set(TotalOnes)-set(NamesOfOnes))
    #print(wrong,len(wrong))
    #return wrong

def posToJump(i, Names):

    for j in range(i, len(Names)):
        n = Names[j].split('.')[0]
        col = int(n.split('_')[2])
        if col == 1:
            return j

'''
This method gets the column of the 1.0 labeled and extends it to the next line
'''
def postprocessing(pred, Names ,Xtpuro , start):
    Xt = list(Xtpuro)
    for i in range(0, len(pred) - 1):
        #print(Xt[i])
        n = Names[i].split('.')[0]
        col = int(n.split('_')[2])
        if (col > 12 or col < 5) and pred[i] == 1.0 and pred[i + 1] == 0.0:
            pred[i + 1] = 1.0
        #if (col > 4 and col < 13) and pred[i] == 1.0:
        #    pred[i] = 0.0
        #if pred[i] == 1.0 and (255.0 not in Xt[i]):
        #    pred[i] = 0.0

    return pred

if __name__ == "__main__":

    #print('using 3 parameters: $./knn.py percentageOfTraining  dimensionsToRescalePCA  KneighboursToClassify')
    #print('If none parameters used, using 80 100 5')
    NXXLab = load_object("Names_X_XLab_NumPxls_50x100")
    Names = NXXLab[0]
    '''
    X and XLab must be converted to np.array float 32 because at NXXLab
    were saved as list
    '''
    X = load_object('XstandarizedToBN')
    #when XstandarizedToBNWidth is loaded, it needs to be converted to np.array
    X = np.array(X)
    #X = NXXLab[1]
    XLab = np.array(NXXLab[2], dtype = np.float32)
    Xpuro = X
    
    #Scaling to 0 mean 1 deviation
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    try:
        comp = int(sys.argv[2])
    except:
        comp = 100
    #VarianceThreshold
    
    #from sklearn.feature_selection import VarianceThreshold
    #sel = VarianceThreshold(threshold = (.2*(1-.2)))
    #X = sel.fit_transform(X)
    
    #Univariate feature selection
    #from sklearn.feature_selection import SelectKBest
    #from sklearn.feature_selection import f_classif
    #s = SelectKBest(f_classif, k=comp)
    #s.fit(X, XLab)

    #aplaying PCA to reduce dimensions
    
    pca = decomposition.PCA(n_components=comp)
    pca.fit(X)
    X = pca.transform(X)
    

    try:
        neig = int(sys.argv[3])
    except:
        neig = 5
    neigh = KNeighborsClassifier(n_neighbors=neig, weights='distance') # weights='distance'
    
    try:
        parc = int(sys.argv[1])
    except:
        parc = 80
    Xtr, XLabtr, Xt, XLabt, _ = partitioning(X, XLab, parc)
    _, _, Xtpuro, _, _ = partitioning(Xpuro, XLab, parc)
    neigh.fit(Xtr, XLabtr)

    pred = neigh.predict(Xt)
    try:
        print(sys.argv[1], sys.argv[2])
        print()
    except:
        pass
    print(confusion_matrix(XLabt, pred))
    print(classification_report(XLabt, pred))

    #This method only needs parc and Xtr to print it, needs pred and XLabt to work
    #w1 = TotalEvaluation(pred, XLabt, Names, parc, Xtr)
    TotalEvaluation(pred, XLabt, Names, parc, Xtr)

    '''
    #Post processing trying to solve some labels by imagination
    pred = postprocessing(pred, Names,Xtpuro, len(Xtr))


    #Reevaluating if postprocessing worked
    print()
    print('After postprocessing')
    print(confusion_matrix(XLabt, pred))
    print(classification_report(XLabt, pred))

    w2 = TotalEvaluation(pred, XLabt, Names, parc, Xtr)
    print()
    print('New Errors from classic to postprocessing')
    print(list(set(w2)-set(w1)), len(list(set(w2)-set(w1))))'''



