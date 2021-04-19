import sys
sys.modules[__name__].__dict__.clear()

import pandas
import time
import copy
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = pandas.read_csv('z_values_gaussian_sorted.csv')
df = df.rename(columns={"Unnamed: 0": "ID"})
df.ID = range(0,len(df))

#sampling
learn_sample = df.sample(frac=0.001, random_state=1).reset_index()
    #need more sorting, cause sampling
    learn_sample = learn_sample.sort_values(['z_value'], ascending=[True]).reset_index()
test_sample = df[~df.ID.isin(learn_sample.ID)].reset_index()

diff_absolute = np.diff(learn_sample.z_value).reshape(-1, 1)
diff_percent = np.diff(learn_sample.z_value)/learn_sample.z_value[1:]*100


plt.plot(diff_absolute, color = "green")
plt.plot(diff_percent, color = "red")


threshhold_values = diff_percent[diff_percent>0.5]
indexes_as_list = threshhold_values.index.to_list()
threshhold_percent_changes = np.diff(indexes_as_list)/len(learn_sample)*100
threshhold_percent_changes = pandas.Series(threshhold_percent_changes)

segment_boundaries_indexes_in_list = pandas.Series(threshhold_percent_changes.index.to_list())[threshhold_percent_changes > 1]
segment_boundaries = pandas.Series(indexes_as_list)[segment_boundaries_indexes_in_list].reset_index()

learn_sample.z_value[0:segment_boundaries.iloc[0][0]].plot()
learn_sample.z_value[segment_boundaries.iloc[0][0]+1:segment_boundaries.iloc[1][0]].plot()
learn_sample.z_value[segment_boundaries.iloc[1][0]+1:segment_boundaries.iloc[2][0]].plot()
learn_sample.z_value[segment_boundaries.iloc[2][0]+1:segment_boundaries.iloc[3][0]].plot()
learn_sample.z_value[segment_boundaries.iloc[3][0]+1:len(learn_sample)].plot()








#### meanshift

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#z_value = learn_sample.z_value.to_list()

X = np.array(learn_sample.z_value).reshape(-1, 1)
#bandwidth = estimate_bandwidth(X, quantile=0.1)
ms = MeanShift(bandwidth=0.00001)
ms.fit(X)
labels = ms.labels_
np.unique(labels)
cluster_centers = ms.cluster_centers_
learn_sample.z_value[ms.labels_ == 0].plot()
learn_sample.z_value[ms.labels_ == 1].plot()
learn_sample.z_value[ms.labels_ == 2].plot()


labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

for k in range(n_clusters_):
    my_members = labels == k
    print "cluster {0}: {1}".format(k, X[my_members, 0])

    # pieceweise linear approx.
    # z_value difference soll nicht x-x(-1)/x sein. weil mit dem zuwachs von zahlen sinkt prozent