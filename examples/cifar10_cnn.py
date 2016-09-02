'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''
from __future__ import print_function
import sys
sys.path.insert(0, "/home/liangjiang/code/keras-jl-ac-mean/")
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2, activity_l1l2

import numpy as np

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols),
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
                        # activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3,
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
                        # activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same',
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
                        # activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
                        # activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.), 
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activity_regularizer = activity_l1l2(l1 = 0., l2 = 0.), 
                        W_regularizer = l2(l = 0.), 
                        b_regularizer = l2(l = 0.)))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

def lr_schedule(epoch):
    if 0 <= epoch and epoch < 100:
        return 0.01
    elif 100 <= epoch and epoch <= 150:
        return 0.001
    else:
        return 0.0001

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              callbacks = [LearningRateScheduler(lr_schedule)],
              # verbose = 2, 
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen = ImageDataGenerator(featurewise_std_normalization = True,
            width_shift_range = 0.125, 
            height_shift_range = 0.125, 
            horizontal_flip = True)
    test_datagen = ImageDataGenerator(featurewise_std_normalization = True,
            width_shift_range = 0.125, 
            height_shift_range = 0.125, 
            horizontal_flip = True)
    datagen.fit(X_train)
    test_datagen.fit(X_test)
    history = model.fit_generator(datagen.flow(X_train, Y_train ,batch_size = batch_size),
            samples_per_epoch = X_train.shape[0], nb_epoch = nb_epoch,
            callbacks = [LearningRateScheduler(lr_schedule)],  
            validation_data=test_datagen.flow(X_test, Y_test, batch_size = batch_size),
            nb_val_samples = 10000)

    # fit the model on the batches generated by datagen.flow()
    # history = model.fit_generator(datagen.flow(X_train, Y_train,
    #                     batch_size=batch_size),
    #                     samples_per_epoch=X_train.shape[0],
    #                     nb_epoch=nb_epoch,
    #                     callbacks = [LearningRateScheduler(lr_schedule)],
    #                     validation_data=(X_test, Y_test))
# weights = model.layers[0].get_weights()
# weights = weights[0]
# weights = weights.reshape(weights.shape[0], weights.size / weights.shape[0])
# weights = np.asmatrix(weights)
# centered_weights = weights - np.mean(weights, axis = 1, keepdims = True)
# covariance = centered_weights * centered_weights.transpose()
# np.savetxt("cifar_covariance", covariance, fmt = "%f")
loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc = history.history["acc"]
val_acc = history.history["val_acc"]
np.savetxt("loss", loss, fmt = "%f")
np.savetxt("val_loss", val_loss, fmt = "%f")
np.savetxt("acc", acc, fmt = "%f")
np.savetxt("val_acc", val_acc, fmt = "%f")
model.save_weights("./weight/weight", overwrite = True)
