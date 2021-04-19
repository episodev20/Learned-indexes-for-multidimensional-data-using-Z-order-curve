import sys
sys.modules[__name__].__dict__.clear()

import pandas
import morton
import time
import copy
import os

###########
# recoding
###########

# Uniform
df = pandas.read_csv('ds_uniform.csv', header=None)
z_values = []
m = morton.Morton(dimensions=2, bits=32)

start_time = time.time()
for i in range(df.count()[0]):
    z_values.append(m.pack(df.iloc[i][0], df.iloc[i][1]))
print("--- %s seconds ---" % (time.time() - start_time))

df_z_values = pandas.DataFrame(z_values)
df_z_values.to_csv("z_values_uniform.csv", header=False)



# Gaussian
df = pandas.read_csv('ds_gaussian.csv', header=None)
z_values = []
m = morton.Morton(dimensions=2, bits=32)

start_time = time.time()
for i in range(df.count()[0]):
    z_values.append(m.pack(df.iloc[i][0], df.iloc[i][1]))
print("--- %s seconds ---" % (time.time() - start_time))

df_z_values = pandas.DataFrame(z_values)
df_z_values.to_csv("z_values_gaussian.csv", header=False)

# World map
df = pandas.read_csv(os.path.join(os.environ["HOMEPATH"], "Desktop/data2.csv"), delimiter=";", header=None)
df = pandas.read_csv("../../data2.csv", delimiter=";", header=None)
z_values = []
m = morton.Morton(dimensions=2, bits=32)

start_time = time.time()
for i in range(df.count()[0]):
    z_values.append(m.pack(int(df.iloc[i][0]), int(df.iloc[i][1])))
print("--- %s seconds ---" % (time.time() - start_time))

df_z_values = pandas.DataFrame(z_values)
df_z_values.to_csv("z_values_data2.csv", header=False)

###########
## Sorting
###########

# Uniform
# concat the DFs and assign proper column names
df_original = pandas.read_csv('ds_uniform.csv', header=None)
df_z_values = pandas.read_csv('z_values_uniform.csv', header=None)
df = copy.copy(df_original)
df = df.rename(columns={0: "x", 1: "y"})
df["z_value"] = df_z_values[1]

# sorting by x and y var1
# df = df.sort_values(['x', 'y'], ascending=[True, True])
# df.to_csv("z_values_gaussian_sorted.csv", header=False)

# sorting by z_value
df = df.sort_values(['z_value'], ascending=[True])
df = df.reindex(range(df.count()[0]))
df.to_csv("z_values_uniform_sorted.csv", header=True)

# Gaussian
# concat the DFs and assign proper column names
df_original = pandas.read_csv('ds_gaussian.csv', header=None)
df_z_values = pandas.read_csv('z_values_gaussian.csv', header=None)
df = copy.copy(df_original)
df = df.rename(columns={0: "x", 1: "y"})
df["z_value"] = df_z_values[1]

# sorting by x and y var1
# df = df.sort_values(['x', 'y'], ascending=[True, True])
# df.to_csv("z_values_gaussian_sorted.csv", header=False)

# sorting by z_value
df = df.sort_values(['z_value'], ascending=[True])
df = df.reindex(range(df.count()[0]))
df.to_csv("z_values_gaussian_sorted.csv", header=True)



# World_map
# concat the DFs and assign proper column names
df_original = pandas.read_csv(os.path.join(os.environ["HOMEPATH"], "Desktop/data2.csv"), delimiter=";", header=None)
df_original = pandas.read_csv("../../data2.csv", delimiter=";", header=None)
df_z_values = pandas.read_csv('z_values_data2.csv', header=None)
df = copy.copy(df_original)
df = df.rename(columns={0: "x", 1: "y"})
df["z_value"] = df_z_values[1]

# sorting by x and y var1
# df = df.sort_values(['x', 'y'], ascending=[True, True])
# df.to_csv("z_values_gaussian_sorted.csv", header=False)

# sorting by z_value
df = df.sort_values(['z_value'], ascending=[True])
df["index"] = range(df.count()[0])
df = df.set_index("index")
df.to_csv("z_values_data2_sorted.csv", header=True)


#### visualization
df = pandas.read_csv('z_values_gaussian_sorted.csv')
df = df.rename(columns={0: "ID", 1: "x", 2: "y", 3: "z"})
df.plot.scatter(x="x", y="y")
df["z_value"].plot()

df = pandas.read_csv('z_values_uniform_sorted.csv')
df = df.rename(columns={0: "ID", 1: "x", 2: "y", 3: "z"})
df.plot.scatter(x="x", y="y")
df["z_value"].plot()

df = pandas.read_csv(os.path.join(os.environ["HOMEPATH"], "Desktop/z_values_data2_sorted.csv"))
df = pandas.read_csv("../../z_values_data2_sorted.csv")
df["z_value"].plot()


df_sample = df.sort_values(['z_value'], ascending=[True]).reset_index()
df_sample.to_csv("z_values_gaussian_sorted_100k_sample.csv", header=True)