import scipy.io
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import mne
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
from keras import regularizers


def create_raw_model(nchan, nclasses, trial_length=960, l1=0):
    """
    CNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model

def create_raw_model2(nchan, nclasses, trial_length=960, l1=0, full_output=False):
    """
    CRNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((5, 1), strides=(5, 1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(40, activation="sigmoid", dropout=0.25, return_sequences=full_output))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model



def load_data(subject):
    '''
    This function returns training and testing EEG data, and training labels.
    Parameter: Subject number
    Return:
        - X_train: training set EEG data
        - X_test: testing set EEG data
        - y_train: training set labels
    '''
    seperator = ''  # specify the seperating term when joining strings
    # load train, test, label .mat files
    X_train = scipy.io.loadmat(seperator.join(['D:/Josie/23spring/signal_process/bandpass_filter_only/', subject, '_train.mat']));
    X_test = scipy.io.loadmat(seperator.join(['D:/Josie/23spring/signal_process/bandpass_filter_only/', subject, '_test.mat']));
    y_train = scipy.io.loadmat(seperator.join(['D:/Josie/23spring/signal_process/bandpass_filter_only/', subject, '_label.mat']));
    # The loaded files are in the data structure numpy.void, findind the array in void that contains EEG data
    X_train = X_train['EEG'][0][0][15]
    X_test = X_test['EEG'][0][0][15]
    y_train = y_train['train_label']

    return X_train, X_test, y_train