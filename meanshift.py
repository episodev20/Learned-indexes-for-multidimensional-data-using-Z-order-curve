import pandas
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import copy
import morton
from sklearn.cluster import MeanShift

df = pandas.read_csv('z_values_world_map_sorted.csv')

df = df.sample(frac=0.001, random_state=1)
df = np.array(df)

clustering = MeanShift(2000).fit(df[:, [1, 2]])
clustering.labels_

#len(np.unique(clustering.labels_))

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=9, affinity='euclidean', linkage='ward')
a = cluster.fit_predict(df[:, [1, 2]])

df = pandas.DataFrame(df)
plt.scatter(df[1][clustering.labels_ == 0], df[2][clustering.labels_ == 0], color = "red")
plt.scatter(df[1][clustering.labels_ == 1], df[2][clustering.labels_ == 1], color = "blue")
plt.scatter(df[1][clustering.labels_ == 2], df[2][clustering.labels_ == 2], color = "green")
plt.scatter(df[1][clustering.labels_ == 3], df[2][clustering.labels_ == 3], color = "black")
plt.scatter(df[1][clustering.labels_ == 4], df[2][clustering.labels_ == 4], color = "yellow")
plt.scatter(df[1][clustering.labels_ == 5], df[2][clustering.labels_ == 5], color = "magenta")
plt.scatter(df[1][clustering.labels_ == 6], df[2][clustering.labels_ == 6], color = "brown")
plt.scatter(df[1][clustering.labels_ == 7], df[2][clustering.labels_ == 7], color = "teal")
plt.scatter(df[1][clustering.labels_ == 8], df[2][clustering.labels_ == 8], color = "Lime")
plt.scatter(df[1][clustering.labels_ == 9], df[2][clustering.labels_ == 9], color = "chocolate")
plt.scatter(df[1][clustering.labels_ == 10], df[2][clustering.labels_ == 10], color = "orange")
plt.scatter(df[1][clustering.labels_ == 11], df[2][clustering.labels_ == 11], color = "cyan")