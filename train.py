# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
#from keras import initializations
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from image import extract_data, resize_with_pad, IMAGE_SIZE, IMAGE_CHANNELS


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None
        self.nb_classes = 0

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE):
        images, labels = extract_data('./data/')
        nb_classes = len(set(labels))
        labels = np.reshape(labels, [-1])
        X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2, random_state=1234)
        _, X_test, _, y_test = train_test_split(images, labels, test_size=0.5, random_state=5678)
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], IMAGE_CHANNELS, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], IMAGE_CHANNELS, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], IMAGE_CHANNELS, img_rows, img_cols)
            input_shape = (IMAGE_CHANNELS, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, IMAGE_CHANNELS)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, IMAGE_CHANNELS)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, IMAGE_CHANNELS)
            input_shape = (img_rows, img_cols, IMAGE_CHANNELS)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')
        print('nb_classes:', nb_classes)

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test
        self.nb_classes = nb_classes


class Model(object):

    FILE_PATH = './store/model.h5'

    def __init__(self):
        self.model = None


    def build_model_cnn(self, dataset):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(dataset.nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=40, data_augmentation=True):
        optimizer = Adam()
        #optimizer = 'sgd'
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.X_valid, dataset.Y_valid),
                           shuffle=True)
        else:
            print('Using real-time data augmentation.')

            datagen = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=10,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)
                shear_range=0.01,
                zoom_range=0.1,
                horizontal_flip=False,                # randomly flip images
                vertical_flip=False)                  # randomly flip images

            datagen.fit(dataset.X_train)

            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.X_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.X_valid, dataset.Y_valid))

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        if K.image_dim_ordering() == 'th':
            if image.shape != (1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
                image = resize_with_pad(image)
                image = image.reshape((1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        else:
            if image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
                image = resize_with_pad(image)
                image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image, verbose=0)
        print(result)

        return result

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()

    model = Model()
    model.build_model_cnn(dataset)

    model.train(dataset, batch_size=128, nb_epoch=20, data_augmentation=True)

    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)

