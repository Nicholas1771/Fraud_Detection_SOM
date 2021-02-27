# import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dense, Dropout
from keras.models import Sequential

# import the dataset
dataset = pd.read_csv('data/Credit_Card_Applications_Fraud.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# preprocess the data
sc = StandardScaler()
X = sc.fit_transform(X)

# build the ann
ann = Sequential()
ann.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))
ann.add(Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the ann
ann.fit(X, y, batch_size=1, epochs=2)

# make predictions
y_pred = ann.predict(X)

# sort fraud customers
customers = dataset.iloc[:, 0].values
customers = np.reshape(customers, ([689, 1]))
results = np.concatenate((customers, y_pred), axis=1)
results = results[results[:, 1].argsort()]


