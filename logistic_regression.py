import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_ITER = 5000
LR = 0.01

#np.set_printoptions(threshold=sys.maxsize)
def sigmoid(theta, x):
    z = np.dot(theta.T, x)
    sig = 1/(1+np.exp(-z))
    return sig

def loss_grad(theta, data, labels):
    total_samples = len(labels)
    total_samples = 10
    loss = np.zeros(3001)
    for i in range(0, total_samples):
        loss = loss + data[i]*(sigmoid(theta, data[i])-labels[i]);
    loss = loss/total_samples
    return loss
       
def train_and_get_err (data, labels, test_data, test_labels):

    theta_0 = np.zeros(3001)

    for i in range(0, NUM_ITER):
        theta_t = theta_0 - LR*loss_grad(theta_0, data, labels);
        #print(loss_grad(theta_0, data, labels))
        theta_0 = theta_t
    #print(theta_t)
    
    pred = []
    for i in range(0, 1000):
        prediction = sigmoid(theta_t, test_data[i])
        if (prediction >= 0.5):
            pred.append(1)
        else:
            pred.append(0)
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
    #print(err)
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

    df_test = df.drop('Email No.', axis=1)
    df_test = df_test.drop('Prediction', axis=1)
    df_test = df_test[T_START:T_END]


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
    #print("data before = ", data)
    #u = np.mean(data,axis=0)
    #print("u = ",u)
    #std = np.std(data, axis=0)
    #print("std = ", std)
    #data = (data-u)/std
    #print("data after = ", data)
    data = (data)
    labels = df_label.to_numpy()
    test_data = df_test.to_numpy()
    test_data = (test_data)
    test_labels = df_test_label.to_numpy()

    ones = np.ones(4000)
    ones = np.expand_dims(ones, axis=1)
    data = np.append(data, ones, axis=1)

    ones = np.ones(1000)
    ones = np.expand_dims(ones, axis=1)
    test_data = np.append(test_data, ones, axis=1)

    return data, labels, test_data, test_labels
    

#data = np.loadtxt("emails.csv", dtype = float, usecols=range(1,3001), skiprows=1, delimiter=",")
#labels = np.loadtxt("emails.csv", dtype = int, usecols=3001, skiprows=1, delimiter=",")
#ones = np.ones(5000)
#ones = np.expand_dims(ones, axis=1)
#
#print(data)
#print(ones)
#
#data = np.append(data, ones, axis=1)
#data_shape = data.shape
#labels_shape = labels.shape
#
#print(data_shape, labels_shape)
#print (data)
#print (labels)
#
#theta_0 = np.zeros(3001)
#
#for i in range(0, NUM_ITER):
#    theta_t = theta_0 - LR*loss_grad(theta_0, data, labels);
#    theta_0 = theta_t
#    #print(theta_t)
#
#
#print(theta_t)
#
#ones = np.ones(1)
#ones = np.expand_dims(ones, axis=1)
#prediction = sigmoid(theta_t, data[1])
#print(prediction)

df = pd.read_csv("emails.csv")

data1, labels1, test_data1, test_labels1 = form_data(df, 0, 1000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_labels1)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_labels1 = form_data(df, 1000, 2000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_labels1)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_labels1 = form_data(df, 2000, 3000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_labels1)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_labels1 = form_data(df, 3000, 4000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_labels1)
print("Accuracy = ", 1-(err/1000))

data1, labels1, test_data1, test_labels1 = form_data(df, 4000, 5000, 5000)
err = train_and_get_err(data1, labels1, test_data1, test_labels1)
print("Accuracy = ", 1-(err/1000))

