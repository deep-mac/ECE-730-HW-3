import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def train_and_get_err (data, labels, test_data, test_labels, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(data, labels)

    pred = neigh.predict(test_data)
    err = np.count_nonzero(test_labels != pred)
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
k_list = [1, 3, 5, 7, 10]
k_err = []
total_err = 0
for k in k_list:
    total_err = 0
    data1, labels1, test_data1, test_data2 = form_data(df, 0, 1000, 5000)
    err = train_and_get_err(data1, labels1, test_data1, test_data2, k)
    #print(err)
    total_err = total_err+err

    data1, labels1, test_data1, test_data2 = form_data(df, 1000, 2000, 5000)
    err = train_and_get_err(data1, labels1, test_data1, test_data2, k)
    #print(err)
    total_err = total_err+err

    data1, labels1, test_data1, test_data2 = form_data(df, 2000, 3000, 5000)
    err = train_and_get_err(data1, labels1, test_data1, test_data2, k)
    #print(err)
    total_err = total_err+err

    data1, labels1, test_data1, test_data2 = form_data(df, 3000, 4000, 5000)
    err = train_and_get_err(data1, labels1, test_data1, test_data2, k)
    #print(err)
    total_err = total_err+err

    data1, labels1, test_data1, test_data2 = form_data(df, 4000, 5000, 5000)
    err = train_and_get_err(data1, labels1, test_data1, test_data2, k)
    #print(err)
    total_err = total_err+err
    #print(k , 1-(total_err/5000))
    k_err.append(1-(total_err/5000))


print(k_err)

plt.plot(k_list, k_err)
plt.grid()
plt.savefig('knn_q4.pdf', format='pdf')



