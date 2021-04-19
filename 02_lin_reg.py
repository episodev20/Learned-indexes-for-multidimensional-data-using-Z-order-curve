    import sys
    sys.modules[__name__].__dict__.clear()

    import pandas
    import time
    import copy
    import numpy as np
    from sklearn import datasets, linear_model
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score


# Uniform
# data prep
df = pandas.read_csv('z_values_uniform_sorted.csv')
df = df.rename(columns={"index": "original_database_ID"})
learn_sample = df.sample(frac=0.6, random_state=1)
test_sample = df[~df.original_database_ID.isin(learn_sample.original_database_ID)].reset_index()

#actually it would be more precise to reset indexes for learned and test samples. i guess... shut think on that.

#lin_reg_x = np.array(learn_sample[{"x","y"}])
lin_reg_z_value = np.array(learn_sample.z_value).reshape(-1, 1)
lin_reg_original_database_ID = np.array(learn_sample.original_database_ID)

# create model
model = linear_model.LinearRegression(fit_intercept=False).fit(lin_reg_z_value, lin_reg_original_database_ID)

# predictions
#lin_reg_pred = model.predict(np.array(learn_sample.z_value).reshape(-1, 1))
lin_reg_pred = model.predict(np.array(test_sample.z_value).reshape(-1, 1))
lin_reg_pred = pandas.DataFrame(lin_reg_pred.round(), columns = ['Predictions'])

#get it into same df
test_sample["prediction"] = copy.copy(lin_reg_pred)

#prediction visualization
#figure = test_sample[{"original_database_ID", "prediction"}].plot(color =  {"green", "red"}, alpha = 0.5)
#figure.set_xlabel("Index")
#figure.set_ylabel("Database ID")

figure = plt.plot(np.array(test_sample.original_database_ID), np.array(test_sample.z_value), '-',
                  np.array(test_sample.prediction), np.array(test_sample.z_value), "-", alpha = 0.5)
#figure = test_sample[{"z_value", "interpolated"}].plot(color =  {"green", "red"}, alpha = 0.5)
plt.ylabel("Z_value")
plt.xlabel("Index")
plt.legend(['Original', 'Prediction']);

# model info
model.coef_
model.intercept_
mean_squared_error(test_sample.Database_ID, test_sample.prediction)
r2_score(test_sample.Database_ID, test_sample.prediction)

#Gaussian
# data prep
df = pandas.read_csv('z_values_gaussian_sorted.csv')
df = df.rename(columns={"index": "original_database_ID"})
learn_sample = df.sample(frac=0.6, random_state=1)
    #need more sorting, cause sampling
    learn_sample = learn_sample.sort_values(['z_value'], ascending=[True]).reset_index()
test_sample = df[~df.original_database_ID.isin(learn_sample.original_database_ID)].reset_index()

#lin_reg_x = np.array(learn_sample[{"x","y"}])
lin_reg_z_value = np.array(learn_sample.z_value).reshape(-1, 1)
lin_reg_original_database_ID = np.array(learn_sample.original_database_ID)

# create model
model = linear_model.LinearRegression(fit_intercept=False).fit(lin_reg_z_value, lin_reg_original_database_ID)

# predictions
#lin_reg_pred = model.predict(np.array(learn_sample.z_value).reshape(-1, 1))
lin_reg_pred = model.predict(np.array(test_sample.z_value).reshape(-1, 1))
lin_reg_pred = pandas.DataFrame(lin_reg_pred.round(), columns = ['Predictions'])

#get it into same df
test_sample["prediction"] = copy.copy(lin_reg_pred)

#prediction visualization
#figure = test_sample[{"original_database_ID", "prediction"}].plot(color =  {"green", "red"}, alpha = 0.5)
#figure.set_xlabel("Index")
#figure.set_ylabel("Database ID")

figure = plt.plot(np.array(test_sample.original_database_ID), np.array(test_sample.z_value), '-',
                  np.array(test_sample.prediction), np.array(test_sample.z_value), "-", alpha = 0.5)
    #figure = test_sample[{"z_value", "interpolated"}].plot(color =  {"green", "red"}, alpha = 0.5)
plt.ylabel("Z_value")
plt.xlabel("Index")
plt.legend(['Original', 'Prediction'])

# model info
model.coef_
model.intercept_
mean_squared_error(test_sample.original_database_ID, test_sample.prediction)
r2_score(test_sample.original_database_ID, test_sample.prediction)



# World map #

df = pandas.read_csv('z_values_world_map_sorted.csv')
df = df.rename(columns={"index": "original_database_ID"})
learn_sample = df.sample(frac=0.6, random_state=1)
    #need more sorting, cause sampling
    learn_sample = learn_sample.sort_values(['z_value'], ascending=[True]).reset_index()
test_sample = df[~df.original_database_ID.isin(learn_sample.original_database_ID)].reset_index()

#lin_reg_x = np.array(learn_sample[{"x","y"}])
lin_reg_z_value = np.array(learn_sample.z_value).reshape(-1, 1)
lin_reg_original_database_ID = np.array(learn_sample.original_database_ID)

# create model
model = linear_model.LinearRegression(fit_intercept=False).fit(lin_reg_z_value, lin_reg_original_database_ID)

# predictions
#lin_reg_pred = model.predict(np.array(learn_sample.z_value).reshape(-1, 1))
lin_reg_pred = model.predict(np.array(test_sample.z_value).reshape(-1, 1))
lin_reg_pred = pandas.DataFrame(lin_reg_pred.round(), columns = ['Predictions'])

#get it into same df
test_sample["prediction"] = copy.copy(lin_reg_pred)

#prediction visualization
#figure = test_sample[{"original_database_ID", "prediction"}].plot(color =  {"green", "red"}, alpha = 0.5)
#figure.set_xlabel("Index")
#figure.set_ylabel("Database ID")



figure = plt.plot(np.array(test_sample.original_database_ID), np.array(test_sample.z_value), '-',
                  np.array(test_sample.prediction), np.array(test_sample.z_value), "-", alpha = 0.5)
    #figure = test_sample[{"z_value", "interpolated"}].plot(color =  {"green", "red"}, alpha = 0.5)
plt.ylabel("Z_value")
plt.xlabel("Index")
plt.legend(['Original', 'Prediction'])


# model info
model.coef_
model.intercept_
mean_squared_error(test_sample.original_database_ID, test_sample.prediction)
r2_score(test_sample.original_database_ID, test_sample.prediction)





