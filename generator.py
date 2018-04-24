import numpy as np

import h5py

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from skimage import color, measure


f = h5py.File('DecorColorImages.h5', 'r')

keys = list(f.keys())
print(keys)

countries = np.array(f[keys[0]])
decors = np.array(f[keys[1]])
images = np.array(f[keys[2]])
types = np.array(f[keys[3]])

print ('Country shape:', countries.shape)
print ('Decor shape', decors.shape)
print ('Image shape:', images.shape)
print ('Type shape', types.shape)
images = images.astype('float32')/255

pattern_number = 106
plt.figure(figsize=(5,5))
plt.imshow(images[pattern_number])
print(images[pattern_number])
print(images[pattern_number].shape)
print(images[pattern_number][0])
plt.plot()
plt.show()

# One-hot encode the targets, started from the zero label
cat_countries = to_categorical(np.array(countries-1), 4)
cat_decors = to_categorical(np.array(decors-1), 7)
cat_types = to_categorical(np.array(types-1), 2)



