#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.insert(0, "/home/liangjiang/code/keras-jl-mean/")
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2, activity_l1l2
from keras import backend as K
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_path", action = 'store',
            help = "Path of learned weight")
    parser.add_argument("--layer", "-l", action = 'store', type = int, default = 1,
            dest = 'layer', help = "Layer to be visualized")
    return parser
def random_crop(X_train, size = (3, 3), times = 10):
    num_samples = times * X_train.shape[0]
    print("num_samples: ", num_samples)
    row = X_train.shape[2]
    col = X_train.shape[3]
    crop_row = size[0]
    crop_col  = size[1]
    random_sample = np.random.randint(0, X_train.shape[0], size = num_samples)
    print("random_sample: ", random_sample)
    random_col_index = np.random.randint(0, row - crop_row + 1, size = num_samples) 
    print("random_col_index: ", random_col_index)
    random_row_index = np.random.randint(0, col - crop_col, size = num_samples) 
    print("random_row_index: ", random_row_index)
    # cropped_x_cols = cropped_x.shape[2]
    # cropped_x_rows = cropped_x.shape[3]
    crop_x = np.zeros((num_samples, X_train.shape[1], crop_row, crop_col))
    for i in range(num_samples):
        crop_x[i, :, :, :] = X_train[random_sample[i], :, 
                random_row_index[i] : random_row_index[i] + crop_row,
                random_col_index[i] : random_col_index[i] + crop_col]
    # print("crop_x[0]: ", crop_x[0, :, :, :])
    return crop_x

def main():
    parser = argparser()
    args = parser.parse_args()
    weight_path = args.weight_path
    layer = args.layer

    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3
    batch_size = 32
    nb_classes = 10


    model = Sequential()
    
    print("Making model")

    model.add(Convolution2D(32, 3, 3, border_mode='same', 
                            input_shape=(img_channels, img_rows, img_cols),
                            W_regularizer = l2(l = 0.), 
                            b_regularizer = l2(l = 0.)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3,
                            W_regularizer = l2(l = 0.), 
                            b_regularizer = l2(l = 0.)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            W_regularizer = l2(l = 0.), 
                            b_regularizer = l2(l = 0.)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3,
                            W_regularizer = l2(l = 0.), 
                            b_regularizer = l2(l = 0.)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer = l2(l = 0.), b_regularizer = l2(l = 0.)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, W_regularizer = l2(l = 0.), b_regularizer = l2(l = 0.)))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    print("Compiling model")
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print("Going to visualize layer ", layer)
    print(model.layers[layer].get_config())

    # load learned weight
    print("Loading weight")
    model.load_weights(weight_path)
    weight = model.layers[0].get_weights()
    print("shape of weight: ", weight[0].shape)
    # generate function to get output at layer to be visualized
    for i in range(len(model.layers)):
        print(i)
    input = model.layers[0].input
    output = model.layers[layer].output
    func = K.function([K.learning_phase()] + [input], output)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # im = X_train[100, :, :, :]
    # im = np.swapaxes(im, 0, 2)
    # im = np.swapaxes(im, 0, 1)
    # plt.figure(1)
    # plt.imshow(im)
    # plt.show()
    # sys.exit()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_test.shape[0], 'test samples')
    crop_x = X_test
    # crop_x = random_crop(X_test, size = (9, 9), times = 10)
    print("shape of crop_x: ", crop_x.shape)
    im = crop_x[0, :, :, :] 
    # print("crop_x[0]", im)
    im = im * 255
    im = im.astype(np.uint8)
    # print("im of uint8: ", im)
    fig = plt.figure()
    # plt.imshow(im)
    # plt.show()
    # sys.exit()

    # get output from layer to be visualized
    # print(X_test[50][1])

    activation = func([0] + [crop_x])
    print("shape of activation: ", activation.shape)
    # max_sample_index = np.argmax(activation, axis = 0)
    # max_sample_index = max_sample_index.squeeze()
    # np.savetxt("max_sample_index", max_sample_index, fmt = "%d")
    # print("shape of max_sample_index: ", max_sample_index.shape)
    # # print("max_29", activation[:, 29, :, :])
    # for i in range(32):
    #     ax = fig.add_subplot(8, 4, i + 1, frameon=False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.xaxis.set_ticks_position('none')
    #     ax.yaxis.set_ticks_position('none')
    #     im = crop_x[max_sample_index[i], :, :, :]
    #     im = np.swapaxes(im, 0, 2)
    #     im = np.swapaxes(im, 1, 0)
    #     # print("shape of im: ", im.shape)
    #     im = im * 255
    #     im = im.astype(np.uint8)
    #     ax.imshow(im)
    # plt.show()
        
    if activation.ndim == 4:
        num = activation.shape[0]
        print("num: ", num)
        col = activation.shape[1]
        print("col: ", col)
        map_size = activation.shape[2] * activation.shape[3]
        print("map_size: ", map_size)
        # temp = np.mean(activation, axis = -1)
        # matrix_activation = np.mean(temp, axis = -1)
        flatten_activation = np.reshape(activation, (num, col * map_size))
        print("shape of flatten_activation: ", flatten_activation.shape)
        trans_activation = flatten_activation.transpose()
        print("shape of trans_activation: ", trans_activation.shape)
        reshape_activation = np.reshape(trans_activation, (col, num * map_size))
        print("shape of reshape_activation: ", reshape_activation.shape)
        matrix_activation = reshape_activation.transpose()
        print("shape of matrix_activation: ", matrix_activation.shape)

        mean = np.mean(matrix_activation, axis = 0, keepdims = True)
        # mean_p = T.printing.Print('mean')(mean)
        std = np.std(matrix_activation, axis = 0, keepdims = True)
        normalized_output = (matrix_activation - mean) / std
        covariance = np.dot(np.transpose(normalized_output), normalized_output) / num  / map_size
    else:
        num = activation.shape[0]
        mean = np.mean(activation, axis = 0, keepdims = True)
        # mean_p = T.printing.Print('mean')(mean)
        std = np.std(activation, axis = 0, keepdims = True)
        normalized_output = (activation - mean) / std
        covariance = np.dot(np.transpose(normalized_output), normalized_output) / num

    np.savetxt("mean", mean, fmt = "%f")
    np.savetxt("std", std, fmt = "%f")
    np.savetxt("covariance", covariance, fmt = "%f")

    
if "__main__" == __name__:
    main()
