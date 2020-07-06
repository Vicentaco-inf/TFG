#!/usr/bin/env python3

import sys
import os

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
        if pred[i] == 1.0:
            TotalOnes.append(Names[len(Xtr) + i])
        #If the sample has the same label than the Labeled
        if pred[i] == XLabt[i]:
            AccSamples = AccSamples + 1

        #Lets get the success labeling the splitted words
        if XLabt[i] == 1.0:
            NamesOfOnes.append(Names[len(Xtr) + i])
            #plt.plot(Xt[i],'r')
            TotalSp = TotalSp + 1
            if pred[i] == 1.0:
                AccSp = AccSp + 1
                NamesOfSuccess.append(Names[len(Xtr) + i])
        
        #The same for non-splitted words
        elif XLabt[i] == 0.0:
            #plt.plot(Xt[i],'g')
            TotalNoSp = TotalNoSp + 1
            if pred[i] == 0.0:
                AccNoSp = AccNoSp + 1


    print('With', str(parc)+'%', 'of test, we get a success of:', str(round((AccSamples*100)/TotalSamples,2)) + "%")
    print('Total samples labeled:', len(XLabt), 'with', len(Xtr), 'for training')
    print('Total splitted words:', TotalSp, 'success labeling them:', str(round((AccSp*100)/TotalSp,2)) + "%")
    print('Total no splitted words:', TotalNoSp, 'success labeling them:', str(round((AccNoSp*100)/TotalNoSp,2)) + "%")
    print()
    print('Total splitted words')
    print(NamesOfOnes, len(NamesOfOnes))
    print()
    print('Splitted words found')
    print(NamesOfSuccess, len(NamesOfSuccess))
    print()
    print('predicted 1.0 labels')
    print(TotalOnes, len(TotalOnes))
    print()
    print('Wrong 1.0 labeled')
    wrong = list(set(TotalOnes)-set(NamesOfOnes))
    print(wrong,len(wrong))
    return wrong

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

#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

if __name__ == "__main__":

#Here loads the models to postProcess
    tolaunch = './knn.py ' + str(70) +' ' +  str(2) + ' ' + str(5)
    os.system(tolaunch)
    for i in range(10, 100, 10):
        print('TRAINING PARTITION', i)

        '''for j in range(2, 5102, 100):
            print('PCA PARTITION', j)

            tolaunch = './knn.py ' + str(i) +' ' +  str(j) + ' ' + str(5)
            print(tolaunch)
            os.system(tolaunch)'''
        tolaunch = './knn.py ' + str(70) +' ' +  str(i) + ' ' + str(5)
        os.system(tolaunch)
    #os.system('./pruebas.py')