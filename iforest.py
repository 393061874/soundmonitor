#encoding: utf8
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

#Isolation Forest


def Isolation_Forest(train_x):
    #train_x,label,xy= tsne.tsne_label(time_range,"csv", False, "sf")

    rng = np.random.RandomState(42)

    clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.1)
    clf.fit(train_x)
    y_pred_train = clf.predict(train_x)
    print y_pred_train
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            y_pred_train[i] = 0
    #palette = np.array(sns.color_palette("hls", 2))
    #plt.scatter(xy.T[0],xy.T[1],color = palette[y_pred_train])
    #plt.show()
    return y_pred_train

def test():
    #    train_x,label,xy= tsne.tsne_label(time_range,"csv", False, "sf")

    rng = np.random.RandomState(42)
    #train_x,label,xy= tsne.tsne_label(time_range,"csv", False, "sf")

    clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.1)
    clf.fit(train_x.drop(['tk'],axis=1))
    y_pred_train = clf.predict(train_x)
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            y_pred_train[i] = 0
    #plt.figure(1)
    #palette = np.array(sns.color_palette("hls", 2))
    #plt.scatter(xy.T[0],xy.T[1],color = palette[y_pred_train])

def test2():
    test_y = clf.predict(test_x)
    for i in range(len(test_y)):
        if test_y[i] == -1:
            test_y[i] = 0

    plt.figure(2)
    palette = np.array(sns.color_palette("hls", 2))
    plt.scatter(xy.T[0],xy.T[1],color = palette[test_y])
    plt.show()


def fast_detect():
    
    X = pd.read_csv("data.X", index_col=0)
    Y = pd.read_csv("data.Y", index_col=0)
    a1 = []
    a2 = []
    #print Y.columns 
    r = Isolation_Forest(X)
    print r
    for i in range(len(r)):
        
        if r[i] == 1:
            #a1.append(Y['file'][i])
            print ".",
            new_path = Y['file'][i].replace("debug","debug/1")
        else:
            #a2.append(Y['file'][i])    
            print "X",
            new_path = Y['file'][i].replace("debug","debug/2")
        os.rename(Y['file'][i],new_path)
    
    print a1
    print a2

fast_detect()