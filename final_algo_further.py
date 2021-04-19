import pandas
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import copy
import morton
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# if lin reg is r2 = 0.99. good we have universal. if not, lets go for clustering.

### var 1 - world map ###
df = pandas.read_csv('z_values_world_map_sorted.csv')

### or ###

### var 2 - gaussian ###
#df = pandas.read_csv('z_values_gaussian_sorted.csv')

### go on ###

sampled_df = df.sample(frac=0.001, random_state=1)
sampled_df = sampled_df.sort_values("z_value", ascending=[True])
sampled_df = sampled_df.reset_index()
sampled_df = sampled_df.drop(columns={"index"})
sampled_df = np.array(sampled_df)

from sklearn.cluster import AgglomerativeClustering



#### dendogram ###
# ### decide number of clusters ###
# from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import pyplot as plt
#
# df = np.array(df)
#
# linked = linkage(df, 'single')
#
# dendrogram(linked,
#             orientation='top',
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()
# ### end dendogram ###



cluster = AgglomerativeClustering(n_clusters=100, affinity='euclidean', linkage='ward')
a = cluster.fit_predict(sampled_df[:, [1, 2]])



sampled_df = pandas.DataFrame(sampled_df)
sampled_df = sampled_df.rename(columns = {1: "x", 2: "y", 3 : "z_value"})
sampled_df["cluster"] = a


# k neirest neighbors clustering
X = np.array(sampled_df[{"x", "y"}])
Y = np.array(sampled_df.cluster)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)

#predicted = neigh.predict(np.array(sampled_df[{"x", "y"}]))
#sampled_df["cluster2"] = predicted


predicted = neigh.predict(np.array(df[{"x", "y"}]))
df["cluster"] = predicted




# create list where to store final clusters porameters
final_clusters = pandas.DataFrame(columns= ["clustername", "hierarhical_cluster", "n", "k", "min_y", "max_y", "min_z", "max_z", "delete", "r2"])



######### functions to use later ####


#checking if what R2 for a cluster is
def check_r2(hierarhical_cluster, n, k, min_cluster_z_value):

    #retrieving boundaries of a cluster
    clustername = "subclass_" + str(hierarhical_cluster) + "_" + str(n) + "_" + str(k) + "_" + str(min_cluster_z_value)
    cluster_data = final_clusters[final_clusters.clustername == clustername]

    min_y = int(cluster_data.min_y)
    max_y = int(cluster_data.max_y)

    if int(cluster_data.max_y) != 0:
        min_z = int(cluster_data.min_z)
        max_z = int(cluster_data.max_z)


    #using same variable but for retrievien actual data for the cluster
    cluster_data = df[df.cluster == hierarhical_cluster]
    cluster_data = cluster_data[cluster_data.y > min_y]
    cluster_data = cluster_data[cluster_data.y <= max_y]

    if max_z != 0:
        cluster_data = cluster_data[cluster_data.z_value > min_z]
        cluster_data = cluster_data[cluster_data.z_value <= max_z]



    if len(cluster_data) == 0:
        return "delete"

    #building ling reg
    lin_reg_z_value = np.array(cluster_data.z_value).reshape(-1, 1)
    lin_reg_original_database_ID = np.array(cluster_data.index)
    model = linear_model.LinearRegression(fit_intercept=False).fit(lin_reg_z_value, lin_reg_original_database_ID)

    lin_reg_pred = model.predict(np.array(cluster_data.z_value).reshape(-1, 1))
    lin_reg_pred = pandas.DataFrame(lin_reg_pred.round(), index=cluster_data.index, columns=['Predictions'])

    cluster_data["prediction"] = copy.copy(lin_reg_pred)

    r2 = r2_score(cluster_data.index, cluster_data.prediction)
    return r2

# loop for separating current cluster into subclusters
def finer_clusters(n, max_y, min_y, current_cluster, hierarhical_cluster, min_cluster_z_value):
    # set number of multiplications to 1
    k = 1
    while True:
        # break loop if multiplicated number is more than max_y in this cluster.
        if len(final_clusters) == 0:
            index_to_write_to = 0
        else:
            index_to_write_to = max(final_clusters.index)+1

        # for cases if we overshoot, then include all that was before the overshoot
        if 2 ** n * k > max_y:
            #in case if the last legit number was the boundry for previous cluster, then skip this cluster and continue loop
            if 2 ** n * (k - 1) == max_y:
                break
            else:
                clustername = "subclass_" + str(hierarhical_cluster) + "_" + str(n) + "_" + str(k) + "_" + str(min_cluster_z_value)
                min_cluster_y = 2 ** n * (k - 1) + 1
                max_cluster_y = max_y
                final_clusters.loc[index_to_write_to] = [clustername, hierarhical_cluster, n, k, min_cluster_y, max_cluster_y, 0, 0, 0, 0]
                break

        # naming convention is: first number is hierarchical clustering cluste
        # second number is the potenz
        # third number is the multyplyier

        if 2 ** n * k < min_y:
            k = k + 1
            continue

        clustername = "subclass_" + str(hierarhical_cluster) + "_" + str(n) + "_" + str(k) + "_" + str(min_cluster_z_value)

        # definin min_cluster_y
        if k == 1:
            min_cluster_y = min(current_cluster.y)
            #min_cluster_y = 0  # actually should be min(current_cluster.y), but i am not sure number las cant get here.
        else:
            min_cluster_y = 2 ** n * (k - 1) + 1
            if min(current_cluster.y) > min_cluster_y:
                min_cluster_y = min(current_cluster.y)
        max_cluster_y = 2 ** n * k

        final_clusters.loc[index_to_write_to] = [clustername, hierarhical_cluster, n, k, min_cluster_y, max_cluster_y, 0, 0, 0, 0]
        k = k + 1



# separate clusters
def separate_clusters(hierarhical_cluster, n = 1, min_y = 0, max_y = 0, min_cluster_z_value = 0):
    # select one cluster from hierarchical clusterin
    current_cluster = df[df.cluster == hierarhical_cluster]
    # select maximal y in this cluster

    #calculate min and max only if they are not given. (basically an indicator for first run agains following ones
    if max_y == 0:
        max_y = max(current_cluster.y)
    if min_y == 0:
        min_y = min(current_cluster.y)

    # set potenz to 1


    if n == 1:
        while True:
            # compute current number we are checking
            current_y = 2**n
            # increase potenz for next level
            n = n + 1
            # if current number is mor than our max number, than previous one was right n.
            if current_y > max_y:
                break
        #break loop and define n one less than, which was to much and one more less for which was increaset after too much.
        n = n - 2

    finer_clusters(n, max_y, min_y, current_cluster, hierarhical_cluster, min_cluster_z_value)

def separate_clusters_by_z_value(hierarhical_cluster, n, k, min_cluster_y, max_cluster_y, min_cluster_z_value):
    clustername = "subclass_" + str(hierarhical_cluster) + "_" + str(n) + "_" + str(k) + "_" + str(min_cluster_z_value)
    cluster_data = final_clusters[final_clusters.clustername == clustername]

    min_y = int(cluster_data.min_y)
    max_y = int(cluster_data.max_y)

    if int(cluster_data.max_z) != 0:
        min_z = int(cluster_data.min_z)
        max_z = int(cluster_data.max_z)

    # using same variable but for retrievien actual data for the cluster
    cluster_data = df[df.cluster == hierarhical_cluster]
    cluster_data = cluster_data[cluster_data.y > min_y]
    cluster_data = cluster_data[cluster_data.y <= max_y]

    if max_z != 0:
        cluster_data = cluster_data[cluster_data.z_value > min_z]
        cluster_data = cluster_data[cluster_data.z_value <= max_z]



    max_z_value = max(cluster_data.z_value)
    min_z_value = min(cluster_data.z_value)
    threshhold = (max_z_value - min_z_value) * 0.1

    diff_absolute = np.diff(cluster_data.z_value).reshape(-1, 1)
    test = np.where(diff_absolute > threshhold)

    if len(test[0]) == 0:
        return "no"

    for i2 in range(0, len(test[0])):

        if i2 == 0:
            min_cluster_z_value = min_z_value
            max_cluster_z_value = int(cluster_data[test[0][i2]:test[0][i2]+1].z_value)

        if (i2 != 0 & i2 != len(test)):
            min_cluster_z_value = int(cluster_data[test[0][i2-1]:test[0][i2-1]+1].z_value) + 1
            max_cluster_z_value = int(cluster_data[test[0][i2]:test[0][i2]+1].z_value)

        if i2 == len(test):
            min_cluster_z_value = int(cluster_data[test[0][i2-1]:test[0][i2-1]+1].z_value) + 1
            max_cluster_z_value = max_z_value


        clustername = "subclass_" + str(hierarhical_cluster) + "_" + str(n) + "_" + str(k) + "_" + str(min_cluster_z_value)
        index_to_write_to = max(final_clusters.index) + 1
        final_clusters.loc[index_to_write_to] = [clustername,
                                                 hierarhical_cluster,
                                                 n,
                                                 k,
                                                 min_cluster_y,
                                                 max_cluster_y,
                                                 min_cluster_z_value,
                                                 max_cluster_z_value,
                                                 0,
                                                 0]

    # and adjust other clustreting function so that it takes clusters in with respect to z_values boundries.

#starting loop
for i in range(0, len(np.unique(a))):
    separate_clusters(i)


#
# plt.scatter(sampled_df.x[sampled_df.cluster == 0], sampled_df.y[sampled_df.cluster == 0], color = "red")
# plt.scatter(sampled_df.x[sampled_df.cluster == 1], sampled_df.y[sampled_df.cluster == 1], color = "blue")
# plt.scatter(sampled_df.x[sampled_df.cluster == 2], sampled_df.y[sampled_df.cluster == 2], color = "green")
# plt.scatter(sampled_df.x[sampled_df.cluster == 3], sampled_df.y[sampled_df.cluster == 3], color = "black")
# plt.scatter(sampled_df.x[sampled_df.cluster == 4], sampled_df.y[sampled_df.cluster == 4], color = "yellow")
# plt.scatter(sampled_df.x[sampled_df.cluster == 5], sampled_df.y[sampled_df.cluster == 5], color = "magenta")
# plt.scatter(sampled_df.x[sampled_df.cluster == 6], sampled_df.y[sampled_df.cluster == 6], color = "brown")
# plt.scatter(sampled_df.x[sampled_df.cluster == 7], sampled_df.y[sampled_df.cluster == 7], color = "teal")
# plt.scatter(sampled_df.x[sampled_df.cluster == 8], sampled_df.y[sampled_df.cluster == 8], color = "Lime")
#
# plt.scatter(df.x[df.cluster == 0], df.y[df.cluster == 0], color = "red")
# plt.scatter(df.x[df.cluster == 1], df.y[df.cluster == 1], color = "blue")
# plt.scatter(df.x[df.cluster == 2], df.y[df.cluster == 2], color = "green")
# plt.scatter(df.x[df.cluster == 3], df.y[df.cluster == 3], color = "black")
# plt.scatter(df.x[df.cluster == 4], df.y[df.cluster == 4], color = "yellow")
# plt.scatter(df.x[df.cluster == 5], df.y[df.cluster == 5], color = "magenta")
# plt.scatter(df.x[df.cluster == 6], df.y[df.cluster == 6], color = "brown")
# plt.scatter(df.x[df.cluster == 7], df.y[df.cluster == 7], color = "teal")
# plt.scatter(df.x[df.cluster == 8], df.y[df.cluster == 8], color = "Lime")







# check R2 of linreg for each cluster
# if r2 is mora than threshhold (0.99) we are good to let that cluster be.


i = 0
#this for loop is for testing
for i in range(0,5):
# while i < max(final_clusters.index):


#for i in range(0, len(final_clusters)-1):
    # remove rows if there is only 1 point in a cluster. its not really usefull
    # also due to how code is written it can be that min_y > max_y in the table. which is obviously not true
    # so we have to delete this record.
    if final_clusters.min_y[i] >= final_clusters.max_y[i]:
        final_clusters = final_clusters.drop(i)
        i = i + 1
        continue

    # in case we become a cluster boundaries, inbetween our given data. so that we have data under and over it but no in
    # theese boundaries. than delete this class
    r2 = check_r2(hierarhical_cluster=final_clusters.hierarhical_cluster[i],
                n=final_clusters.n[i],
                k=final_clusters.k[i],
                min_cluster_z_value = final_clusters.min_z[i])

    if r2 == "delete":
        final_clusters = final_clusters.drop(i)
        i = i + 1
        continue

    if r2 < 0.50:
        # if not 99, than remove clustering from final_clusters for this n and add one for one n less.
        ## delete this cluster and perform another finer_clustering
        final_clusters.delete[i] = 1

        #if r2 less than threshhold, try finding jump in Z_value and cluster it that way. if not found. use x values as separators.
        if separate_clusters_by_z_value(final_clusters.hierarhical_cluster[i],
                                        final_clusters.n[i],
                                        final_clusters.k[i],
                                        final_clusters.min_y[i],
                                        final_clusters.max_y[i],
                                        final_clusters.min_z[i]) == "no":
            separate_clusters(final_clusters.hierarhical_cluster[i],
                              final_clusters.n[i] - 1,
                              final_clusters.min_y[i],
                              final_clusters.max_y[i])

    final_clusters.r2[i] = r2

    print(i, max(final_clusters.index), i/max(final_clusters.index), final_clusters.r2[i])

    i = i + 1



# reset index for further correct calculations
final_clusters["index"] = range(0, len(final_clusters))
final_clusters = final_clusters.set_index("index")


df.loc[(df["y"].isin(range(0,4096))) & (df.cluster == 4)].z_value.plot()
df.loc[(df["y"].isin(range(16385,16942)))].z_value.plot()

df.loc[(df["y"].isin(range(8193,13532))) & (df.cluster == 1)].reset_index().plot.scatter(x = "index", y = "z_value", alpha=0.5)

#try using more chirarcical clusters
# try separating by 10% change of x in the cluster.