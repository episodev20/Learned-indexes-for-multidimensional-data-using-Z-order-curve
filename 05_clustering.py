import pandas
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import copy
import morton
from sklearn import preprocessing

df = pandas.read_csv('z_values_world_map_sorted.csv')

#df = pandas.read_csv('z_values_gaussian_sorted.csv')

df = df.sample(frac=0.001, random_state=1)
df = df.sort_values("z_value", ascending=[True])
df = df.reset_index()
df = df.drop(columns="index")
df = np.array(df)

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=9, affinity='euclidean', linkage='ward')
a = cluster.fit_predict(df[:, [1, 2]])


df = pandas.DataFrame(df)
df["cluster"] = a
#df = df.sort_values([3], ascending=[True])
#df = df.reset_index()
#df = df.drop(columns="index")


# x-y plot
plt.scatter(df[1][df.cluster == 0], df[2][df.cluster == 0], color = "red")
plt.scatter(df[1][df.cluster == 1], df[2][df.cluster == 1], color = "blue")
plt.scatter(df[1][df.cluster == 2], df[2][df.cluster == 2], color = "green")
plt.scatter(df[1][df.cluster == 3], df[2][df.cluster == 3], color = "black")
plt.scatter(df[1][df.cluster == 4], df[2][df.cluster == 4], color = "yellow")
plt.scatter(df[1][df.cluster == 5], df[2][df.cluster == 5], color = "magenta")
plt.scatter(df[1][df.cluster == 6], df[2][df.cluster == 6], color = "brown")
plt.scatter(df[1][df.cluster == 7], df[2][df.cluster == 7], color = "teal")
plt.scatter(df[1][df.cluster == 8], df[2][df.cluster == 8], color = "Lime")



#z-value plot
df = df.reset_index()
plt.scatter(df.index[df.cluster == 0], df[3][df.cluster == 0], color = "red")
plt.scatter(df.index[df.cluster == 1], df[3][df.cluster == 1], color = "blue")
plt.scatter(df.index[df.cluster == 2], df[3][df.cluster == 2], color = "green")
plt.scatter(df.index[df.cluster == 3], df[3][df.cluster == 3], color = "black")
plt.scatter(df.index[df.cluster == 4], df[3][df.cluster == 4], color = "yellow")
plt.scatter(df.index[df.cluster == 5], df[3][df.cluster == 5], color = "magenta")
plt.scatter(df.index[df.cluster == 6], df[3][df.cluster == 6], color = "brown")
plt.scatter(df.index[df.cluster == 7], df[3][df.cluster == 7], color = "teal")
plt.scatter(df.index[df.cluster == 8], df[3][df.cluster == 8], color = "Lime")



from matplotlib import cm
plt.scatter(x = df.index , y = df[3], c=cm.hot(np.abs(df.cluster)), edgecolor='none')













b = df[3][a == 0]
b = b.sort_values(ascending =[True])
c = pandas.DataFrame()
c["b"] = b
c["index"] = range(c.count()[0])
c = c.set_index("index")
figure = c.plot(legend = False)
figure.set_ylabel("z_value")
figure = c[70:500].plot(legend = False)
figure.set_ylabel("z_value")

b = df[a == 0]
b = b.sort_values([3], ascending=[True])
c = pandas.DataFrame()
c = b
c["index"] = range(c.count()[0])
c = c.set_index("index")
c = c.rename(columns={0: "Database_index", 1: "x", 2: "y", 3: "z_value"})

c.plot()
c[70:500].plot()





x_temp = df[1][a == 0]
x_temp = x_temp.sort_values(ascending=[True])
x = pandas.DataFrame()
x["x_temp"] = x_temp
x["index"] = range(x.count()[0])
x = x.set_index("index")[0:700]

y_temp = df[2][a == 7]
y_temp = y_temp.sort_values(ascending=[True])
y = pandas.DataFrame()
y["y_temp"] = y_temp
y["index"] = range(y.count()[0])
y = y.set_index("index")[0:700]


df_table = pandas.DataFrame(index=range(0,len(x)), columns=range(0,len(x)))
m = morton.Morton(dimensions=2, bits=32)
for x_i in range(0, len(x)):
    for y_i in range(0, len(y)):
        df_table[x_i][y_i] = m.pack(int(x.x_temp[x_i]), int(y.y_temp[y_i]))

normalized_table = df_table.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(normalized_table)
#in x scaled we can clearly see the kink that data makes in z-curve. i think due to jump in values

#### dendogram
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

df = np.array(df)

linked = linkage(df, 'single')

dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


#cluster data
# select left top and right down edges of each cluster.
# if the new point is not in the square (it is assignet to initially blank space (e.g. ocean) than use linear interpolation of 2 nearest points z values. (these values are precomputed)
#  if it is in the squeare. use cluster. and then linear reg or interpolation. depending on the subclass...
    # subclass fill be dependent on the z-curve order of the x-y so that the jumps are neglected