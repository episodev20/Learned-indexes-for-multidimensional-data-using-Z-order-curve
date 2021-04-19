import sys
sys.modules[__name__].__dict__.clear()

import pandas
import morton
import time
import copy
import os

############
# recoding #
############

# Uniform #

df = pandas.read_csv('ds_uniform.csv', header=None)
z_values = []
m = morton.Morton(dimensions=2, bits=32)

for i in range(df.count()[0]):
    z_values.append(m.pack(df[0][i], df[1][i]))

df_z_values = pandas.DataFrame(z_values, columns= ["z_value"])
df_z_values.to_csv("z_values_uniform.csv", index_label="index")

del(df)
del(df_z_values)


# Gaussian #
df = pandas.read_csv('ds_gaussian.csv', header=None)
z_values = []
m = morton.Morton(dimensions=2, bits=32)

for i in range(df.count()[0]):
    z_values.append(m.pack(df[0][i], df[1][i]))

df_z_values = pandas.DataFrame(z_values, columns= ["z_value"])
df_z_values.to_csv("z_values_gaussian.csv", index_label="index")

del(df)
del(df_z_values)


# World map #

# data2.csv is to large to fit on my dropbox, so different paths due to different computers used,
# width different data2.csv save locations
df = pandas.read_csv(os.path.join(os.environ["HOMEPATH"], "Desktop/world_map.csv"), delimiter=";", header=None)
#df = pandas.read_csv("../../data2.csv", delimiter=";", header=None)

#reduce dataset size in order to match other datasets
df = df.sample(n=10000000)
df = df.reset_index()
df = df.drop(columns={"index"})

#unfortunatelly this data is not saved as integers. So morton.pack will complain about it later
df = df.astype('int32')

# y axes should be flipped. because for unknown reason it was flipped while rastering.
df[1] = max(df[1]) - df[1]

#save it for further work
df.to_csv("world_map.csv", index_label="index")

z_values = []
m = morton.Morton(dimensions=2, bits=32)

for i in range(df.count()[0]):
    z_values.append(m.pack(df[0][i], df[1][i]))

df_z_values = pandas.DataFrame(z_values, columns= ["z_value"])
df_z_values.to_csv("z_values_world_map.csv", index_label="index")

del(df)
del(df_z_values)
del(z_values)
del(m)

############
## Sorting #
############

# Uniform #
# concat the DFs and assign proper column names
df = pandas.read_csv('ds_uniform.csv', header=None)
df = df.rename(columns={0: "x", 1: "y"})
df_z_values = pandas.read_csv('z_values_uniform.csv', index_col = "index")
df["z_value"] = df_z_values
del(df_z_values)

# sorting by z_value
df = df.sort_values(['z_value'], ascending=[True])
df = df.reset_index()
df = df.drop(columns=["index"])
df.to_csv("z_values_uniform_sorted.csv", index_label="index")




# Gaussian #
# concat the DFs and assign proper column names
df = pandas.read_csv('ds_gaussian.csv', header=None)
df = df.rename(columns={0: "x", 1: "y"})
df_z_values = pandas.read_csv('z_values_gaussian.csv', index_col = "index")
df["z_value"] = df_z_values
del(df_z_values)

# sorting by z_value
df = df.sort_values(['z_value'], ascending=[True])
df = df.reset_index()
df = df.drop(columns=["index"])
df.to_csv("z_values_gaussian_sorted.csv", index_label="index")



# World_map #
# concat the DFs and assign proper column names

    df = pandas.read_csv("world_map.csv", index_col = "index")
    df = df.rename(columns={"0": "x", "1": "y"})
    df_z_values = pandas.read_csv('z_values_world_map.csv', index_col = "index")
    df["z_value"] = df_z_values
    del(df_z_values)
    
    # sorting by z_value
    df = df.sort_values(['z_value'], ascending=[True])
    df = df.reset_index()
    df = df.drop(columns=["index"])
    df.to_csv("z_values_world_map_sorted.csv", index_label="index")




####################
#### visualization #
####################

## uniform #
df = pandas.read_csv('z_values_uniform_sorted.csv', index_col= "index")
df.plot.scatter(x="x", y="y", alpha=0.5)
figure = df.plot(y="z_value", legend = False)
figure.set_xlabel("index")
figure.set_ylabel("z value")


# Gaussian #
df = pandas.read_csv('z_values_gaussian_sorted.csv',  index_col = "index")
df.plot.scatter(x="x", y="y", alpha=0.5)
figure = df.plot(y="z_value", legend = False)
figure.set_xlabel("index")
figure.set_ylabel("z value")


# World map #
df = pandas.read_csv('z_values_world_map_sorted.csv', index_col= "index")
df.plot.scatter(x="x", y="y", alpha=0.5)
figure = df.plot(y="z_value", legend = False)
figure.set_xlabel("index")
figure.set_ylabel("z value")

