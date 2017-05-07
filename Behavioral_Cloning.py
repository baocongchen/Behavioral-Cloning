
# coding: utf-8

# In[1]:

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import json
from skimage.exposure import adjust_gamma
from drive import rgb2gray
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Convolution2D, Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.misc import imresize


# Get steering angles for controlled driving
angles = pd.read_csv('./new_data/driving_log.csv', header=0)
angles.columns = ('Center Image','Left Image','Right Image','Steering Angle','Throttle','Brake','Speed')
angles = np.array(angles['Steering Angle'])

# Create arrays for center images of controlled driving
images = np.asarray(os.listdir('./new_data/IMG/'))
center = np.ndarray(shape=(len(angles), 20, 64, 3))

# Create controlled driving datasets
# Images are resized to x64 to increase training speeds
# The top 12 pixels are cropped off because they contain irrelevant information for training behavior
# The final image size to be used in training is 20 x 64 x 1. 
count = 0
angles_num = len(angles)
for image in images:
    image_file = os.path.join('./new_data/IMG', image)
    if image.startswith('center'):
        image_data = ndimage.imread(image_file).astype(np.float32)
        center[count % angles_num] = imresize(image_data, (32,64,3))[12:,:,:]
    count += 1


X_train = center
y_train = angles

# Create a mirror image of the images in the dataset to prevent bias
mirror = [X_train[0]]
mirror_angles = [y_train[0]]
for i in range(1, len(X_train)):
    angle = y_train[i]
    mirror_angles = np.append(mirror_angles, [angle * -1], axis=0)
    mirror = np.append(mirror, [np.fliplr(X_train[i])], axis=0)
print(mirror.shape)

# Combine regular features/labels with mirror features/labels
X_train = np.concatenate((X_train, mirror), axis=0)
y_train = np.concatenate((y_train, mirror_angles),axis=0)

# Perform train/test split to a create validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.1)


# In[2]:

# Build model architecture
# The image data is cropped and normalized to achieve the best prediction accuracy.
model = Sequential()
model.add(Cropping2D(((0,6),(0,0)), input_shape=(20, 64, 3)))
model.add(Lambda(lambda x: (x / 127.5) - .5))
model.add(Convolution2D(12, 4, 4, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(24, 2, 2, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 2, 2, border_mode='same', subsample=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.summary()


# In[3]:

# Compile model with adam optimizer and learning rate of .0001
adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['accuracy'])

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = './checkpoints/chk.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True, monitor='val_loss')

# Stop training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

# Train model for 25 epochs and a batch size of 45
model.fit(X_train,
        y_train,
        nb_epoch=25,
        verbose=0,
        batch_size=45,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, callback])

json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
model.save('model.h5')    
print("Model Saved")


# In[ ]:



