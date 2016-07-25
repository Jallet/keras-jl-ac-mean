'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
# np.random.seed(1337)  # for reproducibility
import sys
sys.path.insert(0, "/home/liangjiang/code/keras-jl/")

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l1l2

from PIL import Image

batch_size = 128
nb_classes = 10
nb_epoch = 1000

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(8, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols),
                        # W_regularizer = l1l2ld(l1 = 0., l2 = 0., ld = 0.), 
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.), 
                        activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(16, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols),
                        # W_regularizer = l1l2ld(l1 = 0., l2 = 0., ld = 0.), 
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(32, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols),
                        # W_regularizer = l1l2ld(l1 = 0., l2 = 0., ld = 0.), 
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
# model.add(AveragePooling2D(pool_size=(5, 5)))
# model.add(Dropout(0.25))

model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes, W_regularizer = l2(l = 0.), b_regularizer = l2(l = 0.)))
model.add(Activation('softmax'))

plot(model, to_file = "./mnist.png", show_shapes = True)
# sys.exit()

for i in range(len(model.layers)):
    print("i: ", i) 
    print(model.layers[i].get_config())

# sys.exit()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5)],
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
weights = model.layers[0].get_weights()
print(weights)
weights = weights[0]
print ("shape of weights: ", weights.shape)
reshaped_weights = weights.reshape((weights.shape[0]), weights.size / weights.shape[0])
reshaped_weights = np.asmatrix(reshaped_weights)
print ("shape of weights: ", reshaped_weights.shape)
# print("shape of reshaped_weights: ", reshaped_weights.shape)
# print ("weights of conv layer:", weights)
# print ("reshaped_weights of conv layer: \n", reshaped_weights)
centered_weights = (reshaped_weights - np.mean(reshaped_weights, axis = 1, keepdims = True)) / np.std(reshaped_weights, axis = 1, keepdims = True)
# print("centered_weights:")
# print(centered_weights)
covariance = centered_weights * centered_weights.transpose() / (centered_weights.shape[1])
# print("covariance")
# print(covariance)
np.savetxt("covariance", covariance, fmt = "%f")
weights = model.layers[7].get_weights()
weights = weights[0]
weights = weights.transpose()
print("shape of weights: ", weights.shape)
reshaped_weights = weights.reshape((weights.shape[0]), weights.size / weights.shape[0])
reshaped_weights = np.asmatrix(reshaped_weights)
# print("shape of reshaped_weights: ", reshaped_weights.shape)
# print ("weights of conv layer:", weights)
# print ("reshaped_weights of conv layer: \n", reshaped_weights)
centered_weights = (reshaped_weights - np.mean(reshaped_weights, axis = 1, keepdims = True)) / np.std(reshaped_weights, axis = 1, keepdims = True)
# print("centered_weights:")
# print(centered_weights)
covariance = centered_weights * centered_weights.transpose() / (weights.shape[1])
# print("covariance")
# print(covariance)
np.savetxt("dense_covariance", covariance, fmt = "%f")
