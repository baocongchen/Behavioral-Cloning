
# coding: utf-8

# In[18]:

import csv
import cv2
import numpy as np

samples = []
with open('../data/1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)
print(len(train_samples), len(validation_samples))
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            if (num_samples - 1) >= (offset + batch_size):
                batch_samples = samples[offset : offset + batch_size]
            else:
                batch_samples = samples[offset : num_samples]
            images, angles, throttles, brakes, speeds = [], [], [], [], []
            
            for batch_sample in batch_samples:
                for i in range(1):
                    image_name = '../data/1/IMG/' + batch_sample[i].split('\\')[-1]
                    image = cv2.imread(image_name)
                    images.append(image)
                    angles.append(float(batch_sample[3]))
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle * -1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)
            
# compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, height, width = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def resize_images(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (45,160))

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(height,width,ch)))
model.add(Lambda(resize_images))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(16, 7, 7, subsample=(4,4), activation='relu', border_mode="same"))
model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu', border_mode="same"))
model.add(Convolution2D(64, 5, 5, subsample=(2,2), activation='relu', border_mode="same"))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')

