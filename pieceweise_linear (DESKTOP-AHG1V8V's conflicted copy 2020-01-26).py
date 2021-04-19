from scipy.interpolate import interp1d
import numpy as np

x = np.linspace(0, 10, 10)
y = np.cos(-x**2/8.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 10, 40)
import matplotlib.pyplot as plt
plt.plot(x,y,'o',xnew,f(xnew))

plt.plot(x,y,'o',xnew,f(xnew),'-', xnew, f2(xnew),'--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

import pandas
df = pandas.read_csv('z_values_gaussian_sorted.csv')
df = df.rename(columns={"Unnamed: 0": "ID"})
df.ID = range(0,len(df))
learn_sample = df.sample(frac=0.6, random_state=1)
    #need more sorting, cause sampling
    learn_sample = learn_sample.sort_values(['z_value'], ascending=[True]).reset_index()
test_sample = df[~df.ID.isin(learn_sample.ID)].reset_index()

x = np.array(learn_sample.z_value)
y = np.array(learn_sample.ID)
f = interp1d(x, y)

interpolated = np.append([0], f(np.array(test_sample.z_value[1:])))
test_sample["interpolated"] = interpolated

f2 = interp1d(x, y, kind='cubic')

plt.plot(x, y, 'o', np.array(test_sample.z_value), interpolated, "-")





df = pandas.read_csv('z_values_data2_sorted.csv')
df = df.rename(columns={"index": "ID"})
learn_sample = df.sample(frac=0.6, random_state=1)
    #need more sorting, cause sampling
    learn_sample = learn_sample.sort_values(['z_value'], ascending=[True]).reset_index()
test_sample = df[~df.index.isin(learn_sample.index)].reset_index()

x = np.array(learn_sample.z_value)
y = np.array(learn_sample.index)
f = interp1d(x, y)

interpolated = np.append([0], f(np.array(test_sample.z_value[1:])))
test_sample["interpolated"] = interpolated

f2 = interp1d(x, y, kind='cubic')




###########

import imageio
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 529170000

#im = imageio.imread('world.png')
#print(im.shape)

#len(im)
#im == [[255, 255, 255, 255]]


from PIL import Image
import numpy as np
import sys
import os
import csv

    img_file = Image.open('world.png')
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    #value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    #value = value.flatten()
    #print(value)
    #with open("D:\img_pixels.csv", 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(value)

    #value = value[:10000]


world_map = list()

for i_width in range(0, width):
    for i_height in range(0, height):
        if img_grey.getpixel((i_width,i_height)) != 255:
            world_map.append([i_width, i_height])


import pandas
#(pandas.DataFrame(world_map)).to_csv("world_mapsss.csv")
np.savetxt("data2.csv", world_map, delimiter=";")


df = pandas.read_csv('data2.csv', delimiter=";", header=None)
learn_sample = df.sample(frac=0.0001, random_state=1).reset_index()
learn_sample.plot.scatter(x=0, y=1)