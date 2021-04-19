import pandas
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import copy
import morton
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import time

### importing data to work with.
### var 1 - world map ###
#df = pandas.read_csv('z_values_world_map_sorted.csv')


### or ###

### var 2 - gaussian ###
df = pandas.read_csv('z_values_gaussian_sorted.csv')

### go on ###

sampled_df = df.sample(n = 10000, random_state=1)
#sampled_df = df.sample(frac=0.001, random_state=1)

# sampled_df = df[df.x > 500]
# sampled_df = sampled_df[sampled_df.x < 1000]
# sampled_df = sampled_df[sampled_df.y > 500]
# sampled_df = sampled_df[sampled_df.y < 100000]

sampled_df = sampled_df.sort_values("z_value", ascending=[True])
sampled_df = sampled_df.reset_index()
sampled_df = sampled_df.drop(columns={"index"})


start = time.time()
### assign cluster name for the data
## load model
#path = "C:\\Users\\AP\Dropbox\\basismodul\\world_data_models\\"
path = "C:\\Users\\AP\Dropbox\\basismodul\\gaussian_data_models\\"
neigh = pickle.load(open(path + "main_clustering.sav",'rb'))

predicted = neigh.predict(np.array(sampled_df[{"x", "y"}]))
sampled_df["cluster"] = predicted
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


clusters_table = pickle.load(open(path + "clusters_table.sav",'rb'))


final_table = pandas.DataFrame()
for i in range(0, len(clusters_table)):
    filename = clusters_table.clustername[i]
    current_model = pickle.load(open(path + filename + ".sav", 'rb'))

    selected_cluster = sampled_df[sampled_df.cluster == clusters_table.hierarhical_cluster[i]]
    selected_cluster = selected_cluster[selected_cluster.y >= clusters_table.min_y[i]]
    selected_cluster = selected_cluster[selected_cluster.y <= clusters_table.max_y[i]]

    if len(selected_cluster) == 0:
        continue

    lin_reg_pred = current_model.predict(np.array(selected_cluster.z_value).reshape(-1, 1))
    lin_reg_pred = pandas.DataFrame(lin_reg_pred.round(), index=selected_cluster.index, columns=['Predictions'])
    selected_cluster["prediction"] = copy.copy(lin_reg_pred)
    final_table = final_table.append(selected_cluster)

    #r2 = r2_score(selected_cluster.level_0, selected_cluster.prediction)

end = time.time()
end - start

a = abs(final_table.level_0 - final_table.prediction)
np.average(a)
b = a/max(df.index)
np.average(b)

