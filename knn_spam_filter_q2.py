import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def train_and_get_err (data, labels, test_data, test_labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(data, labels)

    pred = neigh.predict(test_data)
    err = np.count_nonzero(test_labels != pred)
    TP = 0
    FN = 0
    FP = 0
    for i in range(0, len(test_labels)):
        if test_labels[i] == 1 and pred[i] == 1:
            TP+=1
        if test_labels[i] == 1 and pred[i] == 0:
            FN+=1
        if test_labels[i] == 0 and pred[i] == 1:
            FP+=1
    print("recall =",TP/(TP+FN))
    print("precision =",TP/(TP+FP))
    return err

def form_data(df, T_START, T_END, TOTAL):
    df_train = df.drop('Email No.', axis=1)
    df_train = df_train.drop('Prediction', axis=1)
    if (T_START == 0): # 0 to 1000 is test set, then 1000:5000 is training set
        df_train = df_train[T_END:TOTAL]
    elif (T_END == TOTAL): # 4000 to 5000 is test set, then training set is 0:4000
        df_train = df_train[0:T_START]
    else:
        df_train1 = df_train[0:T_START] #first get 0 to T_START
        df_train2 = df_train[T_END:TOTAL]
        frames = [df_train1, df_train2]
        df_train = pd.concat(frames)

    #print("FORM_DATA")
    #print(df_train)

    df_test = df.drop('Email No.', axis=1)
    df_test = df_test.drop('Prediction', axis=1)
    df_test = df_test[T_START:T_END]
    #print("FORM_DATA")
    #print(df_test)


    df_label = df["Prediction"]
    df_label1 = df_label[0:T_START]
    if (T_END != TOTAL):
        df_label2 = df_label[T_END:TOTAL]
        frames = [df_label1, df_label2]
        df_label = pd.concat(frames)
    else:
        df_label = df_label1

    df_test_label = df["Prediction"]
    df_test_label = df_test_label[T_START:T_END]

    data = df_train.to_numpy()
    labels = df_label.to_numpy()
    test_data = df_test.to_numpy()
    test_labels = df_test_label.to_numpy()

    #print("FORM_DATA")
    #print(data)
    #print("FORM_DATA")
    #print(labels)

    return data, labels, test_data, test_labels

    

df = pd.read_csv("emails.csv")
#print(df)

#df_train = df.drop('Email No.', axis=1)
#df_train = df_train.drop('Prediction', axis=1)
#df_train = df_train[0:4000]
#print(df_train)
#
#df_test = df.drop('Email No.', axis=1)
#df_test = df_test.drop('Prediction', axis=1)
#df_test = df_test[4000:5000]
#print(df_test)
#
#
#df_label = df["Prediction"]
#df_label = df_label[0:4000]
#df_test_label = df["Prediction"]
#df_test_label = df_test_label[4000:5000]
#
#data = df_train.to_numpy()
#labels = df_label.to_numpy()
#test_data = df_test.to_numpy()
#test_labels = df_test_label.to_numpy()
#
#print(data)
#print(labels)

data1, labels1, test_data1, test_data2 = form_data(df, 0, 1000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_data2)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_data2 = form_data(df, 1000, 2000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_data2)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_data2 = form_data(df, 2000, 3000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_data2)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_data2 = form_data(df, 3000, 4000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_data2)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_data2 = form_data(df, 4000, 5000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_data2)
print("Accuracy = ", 1-(err/1000))


