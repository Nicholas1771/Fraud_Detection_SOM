import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show, savefig
import pickle

# import the dataset
dataset = pd.read_csv('data/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# preprocess the data
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# train the som
ms = MiniSom(x=11, y=11, input_len=15, sigma=1.0, learning_rate=0.5)
ms.random_weights_init(X)
ms.train_random(data=X, num_iteration=100)

# visualize results and save image
bone()
pcolor(ms.distance_map().T)
colorbar()
markers = ['x', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = ms.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markersize=10,
         markeredgewidth=2,
         markeredgecolor=colors[y[i]],
         markerfacecolor='None')
show()

# save the SOM model and figure
savefig('som/som_fig')
with open('som/som.p', 'wb') as outfile:
    pickle.dump(ms, outfile)
