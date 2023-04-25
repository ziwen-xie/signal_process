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





def load_data(subject):
    '''
    This function returns training and testing EEG data, and training labels.
    Parameter: Subject number
    Return:
        - X_train: training set EEG data
        - X_test: testing set EEG data
        - y_train: training set labels
    '''
    filepath = '/Users/josie/PycharmProjects/signal_process/bandpass_60_ica/'
    seperator = ''  # specify the seperating term when joining strings
    # load train, test, label .mat files
    X_train = scipy.io.loadmat(seperator.join([filepath, subject, '_train.mat']));
    X_test = scipy.io.loadmat(seperator.join([filepath, subject, '_test.mat']));
    y_train = scipy.io.loadmat(seperator.join([filepath, subject, '_label.mat']));
    # The loaded files are in the data structure numpy.void, findind the array in void that contains EEG data
    X_train = X_train['EEG'][0][0][15]
    X_test = X_test['EEG'][0][0][15]
    y_train = y_train['train_label']

    return X_train, X_test, y_train
