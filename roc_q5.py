import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

NUM_ITER = 10000
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

def knn_train_and_get_err (data, labels, test_data, test_labels, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(data, labels)

    pred = neigh.predict(test_data)
    pred_confidence = neigh.predict_proba(test_data)
    err = np.count_nonzero(test_labels != pred)
    print("knn err = ", err)
    pred_conf_0, pred_conf_1 = np.split(pred_confidence, 2, axis=1)
    print("knn pred, pred_conf_0, pred_conf_1 = ", pred[0], pred_conf_0[0], pred_conf_1[0])
    return pred_conf_1.flatten()

       
def logistic_train_and_get_err (data, labels, test_data, test_labels):

    theta_0 = np.zeros(3001)

    for i in range(0, NUM_ITER):
        theta_t = theta_0 - LR*loss_grad(theta_0, data, labels);
        theta_0 = theta_t
    print(theta_t)
    
    pred = []
    pred_confidence = []
    for i in range(0, 1000):
        prediction = sigmoid(theta_t, test_data[i])
        if (prediction >= 0.5):
            pred.append(1)
        else:
            pred.append(0)
        pred_confidence.append(prediction)
    err = np.count_nonzero(test_labels != pred)
    print("logistic error = ", err)
    return pred_confidence

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
    labels = df_label.to_numpy()

    test_data = df_test.to_numpy()
    test_labels = df_test_label.to_numpy()

    ones = np.ones(4000)
    ones = np.expand_dims(ones, axis=1)
    data = np.append(data, ones, axis=1)

    ones = np.ones(1000)
    ones = np.expand_dims(ones, axis=1)
    test_data = np.append(test_data, ones, axis=1)

    return data, labels, test_data, test_labels
    
df = pd.read_csv("emails.csv")

data1, labels1, test_data1, test_labels1 = form_data(df, 0, 1000, 5000)
log_pred = logistic_train_and_get_err(data1, labels1, test_data1, test_labels1)
knn_pred = knn_train_and_get_err(data1, labels1, test_data1, test_labels1, 5)

df_ones = df[0:1000]
df_ones = df_ones[df_ones["Prediction"] == 1]
print(df_ones.shape[0])

test_labels1 = np.expand_dims(test_labels1, axis=1)
knn_pred = np.expand_dims(knn_pred, axis=1)
knn_pred = np.append(knn_pred, test_labels1, axis=1)
#print(knn_pred)
knn_pred = knn_pred[knn_pred[:, 0].argsort()[::-1]]
knn_pred, knn_labels = np.split(knn_pred, 2, axis=1)
print(knn_pred)

log_pred = np.expand_dims(log_pred, axis=1)
log_pred = np.append(log_pred, test_labels1, axis=1)
#print(log_pred)
log_pred = log_pred[log_pred[:, 0].argsort()[::-1]]
log_pred, log_labels = np.split(log_pred, 2, axis=1)
print(log_pred)

#log_pred = np.sort(log_pred)
#log_pred = log_pred[::-1]
#knn_pred = np.sort(knn_pred)
#knn_pred = knn_pred[::-1]
#print(log_pred)
#print(knn_pred)

num_neg = 1000 - df_ones.shape[0]
num_pos = df_ones.shape[0]
TP = 0
FP = 0
last_TP = 0

knn_FPR = []
knn_TPR = []
for i in range(1, 1000):
    if i > 1 and knn_pred[i] != knn_pred[i-1] and knn_labels[i] == 0 and TP > last_TP:
        FPR = FP/num_neg
        TPR = TP/num_pos
        knn_FPR.append(FPR)
        knn_TPR.append(TPR)
        last_TP = TP
    if knn_labels[i] == 1:
        TP += 1
    else:
        FP += 1
        FPR = FP/num_neg
        TPR = TP/num_pos
        knn_FPR.append(FPR)
        knn_TPR.append(TPR)


TP = 0
FP = 0
last_TP = 0
log_FPR = []
log_TPR = []
for i in range(1, 1000):
    if i > 1 and log_pred[i] != log_pred[i-1] and log_labels[i] == 0 and TP > last_TP:
        FPR = FP/num_neg
        TPR = TP/num_pos
        log_FPR.append(FPR)
        log_TPR.append(TPR)
        last_TP = TP
    if log_labels[i] == 1:
        TP += 1
    else:
        FP += 1
        FPR = FP/num_neg
        TPR = TP/num_pos
        log_FPR.append(FPR)
        log_TPR.append(TPR)



plt.plot(knn_FPR, knn_TPR, label='knn')
plt.plot(log_FPR, log_TPR, label='Logistic regression')
plt.grid()
plt.legend(loc='lower right')
plt.savefig('roc_q5.pdf', format='pdf')
