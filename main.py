from minisom import MiniSom
import numpy as np
import pickle
import pandas as pd
from pylab import bone, pcolor, colorbar, plot, show, savefig
from sklearn.preprocessing import MinMaxScaler

# load the som
ms = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
with open('som/som.p', 'rb') as infile:
    ms = pickle.load(infile)

# import the dataset
dataset = pd.read_csv('data/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# find the frauds
mappings = ms.win_map(X)
frauds = np.concatenate((mappings[(7, 4)], mappings[(6, 6)], mappings[(9, 9)]), axis=0)

# create fraud column array to be appended to X
fraud_column = np.empty([0])
for i in range(len(X)):
    is_fraud = False
    for ii in range(len(frauds)):
        if X[i][0] == frauds[ii][0]:
            is_fraud = True
    if is_fraud:
        fraud_column = np.append(fraud_column, np.array([1]), axis=0)
    else:
        fraud_column = np.append(fraud_column, np.array([0]), axis=0)
fraud_column = np.reshape(fraud_column, (690, 1))

# combine X and new fraud column
X_with_fraud = dataset.iloc[:, :].values
X_with_fraud = np.append(X_with_fraud, fraud_column, axis=1)

# save new array to csv for the ANN
np.savetxt('data/Credit_Card_Applications_Fraud.csv', X_with_fraud, fmt='%.8g', delimiter=',')
