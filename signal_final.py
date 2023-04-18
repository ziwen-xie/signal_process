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
from keras.layers.pooling import AveragePooling1D
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

subject1_train, subject1_test, subject1_label = load_data('Subject_1')
subject1_train = np.swapaxes(subject1_train,0,2)
subject1_test = np.swapaxes(subject1_test,0,2)


subject2_train, subject2_test, subject2_label = load_data('Subject_2')
subject2_train = np.swapaxes(subject2_train,0,2)
subject2_test = np.swapaxes(subject2_test,0,2)

traindata = np.concatenate((subject1_train,subject2_train))
label = np.concatenate((subject1_label,subject2_label))

subject3_train, subject3_test, subject3_label = load_data('Subject_3')
subject3_train = np.swapaxes(subject3_train,0,2)
subject3_test = np.swapaxes(subject3_test,0,2)

traindata = np.concatenate((traindata,subject3_train))
label = np.concatenate((label,subject3_label))

subject4_train, subject4_test, subject4_label = load_data('Subject_4')
subject4_train = np.swapaxes(subject4_train,0,2)
subject4_test = np.swapaxes(subject4_test,0,2)

traindata = np.concatenate((traindata,subject4_train))
label = np.concatenate((label,subject4_label))

subject5_train, subject5_test, subject5_label = load_data('Subject_5')
subject5_train = np.swapaxes(subject5_train,0,2)
subject5_test = np.swapaxes(subject5_test,0,2)

traindata = np.concatenate((traindata,subject5_train))
label = np.concatenate((label,subject5_label))


subject7_train, subject7_test, subject7_label = load_data('Subject_7')
subject7_train = np.swapaxes(subject7_train,0,2)
subject7_test = np.swapaxes(subject7_test,0,2)

traindata = np.concatenate((traindata,subject7_train))
label = np.concatenate((label,subject7_label))



subject8_train, subject8_test, subject8_label = load_data('Subject_8')
subject8_train = np.swapaxes(subject8_train,0,2)
subject8_test = np.swapaxes(subject8_test,0,2)
print(subject8_train.shape, subject8_test.shape, subject8_label.shape)

input_shape = (1200,60)
l1 = 0
model = models.Sequential()
model.add(Conv1D(60, 3, activation='relu',input_shape=input_shape))
model.add(Dense(16, activation="relu"))
model.add(AveragePooling1D(3))
model.add(layers.Flatten())
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.summary()

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(traindata, label, batch_size=16,epochs=100)

plt.plot(history.history['acc'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1])
plt.legend(loc='lower right')
plt.show()

acc = model.evaluate(traindata, label)
print("Loss:", acc[0], " Accuracy:", acc[1])




