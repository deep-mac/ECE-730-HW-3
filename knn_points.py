import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


data = np.loadtxt("D2z.txt", dtype = float, usecols=range(0,2))
labels = np.loadtxt("D2z.txt", dtype = int, usecols=2)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(data, labels)

bp1 = np.arange(0.0, 2.1, 0.1)
print(bp1)
bp2 = np.arange(-2.0, 0, 0.1)
print(bp2)

y = np.concatenate((bp2, bp1), axis=0)
print(y)
all_rows = [[0, 0]]
x = -2
for i in range(0, 41):
    x_arr = np.full((41), x) 
    row = np.column_stack((x_arr, y))
    all_rows = np.concatenate((all_rows, row), axis=0)
    #print(row)
    x = x+0.1

print(all_rows)
pred = neigh.predict(all_rows)

df_x = pd.DataFrame(all_rows, columns=['x', 'y'])
print(df_x)
df = pd.DataFrame(list(zip(df_x["x"].to_list(), df_x["y"].to_list(), pred)), columns =['x', 'y',  'label'])
print(df)

lab0 = df[df["label"] == 0]
x0 = lab0["x"].values.tolist()
y0 = lab0["y"].values.tolist()
lab1 = df[df["label"] == 1]
x1 = lab1["x"].values.tolist()
y1 = lab1["y"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x0, y0, c = "blue")
plt.scatter(x1, y1, c = "red")

pred = neigh.predict(data)
df_x = pd.DataFrame(data, columns=['x', 'y'])
print(df_x)
df = pd.DataFrame(list(zip(df_x["x"].to_list(), df_x["y"].to_list(), pred)), columns =['x', 'y',  'label'])
print(df)
lab0 = df[df["label"] == 0]
x0 = lab0["x"].values.tolist()
y0 = lab0["y"].values.tolist()
lab1 = df[df["label"] == 1]
x1 = lab1["x"].values.tolist()
y1 = lab1["y"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x0, y0, c = "black", marker='x' )
plt.scatter(x1, y1, c = "black", marker='+')
plt.savefig('d2z_boundary.pdf', format="pdf")

