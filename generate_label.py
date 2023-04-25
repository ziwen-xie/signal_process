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
from utils import load_data
from scipy.io import savemat
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split

subject1_train, subject1_test, subject1_label = load_data('Subject_1')  # import data
subject1_train = np.swapaxes(subject1_train,0,2)   # swap to make the shape (trail, time_stamp, feature)
subject1_test = np.swapaxes(subject1_test,0,2)

# X_train, X_test, y_train, y_test = train_test_split(subject1_train,
#                                                     subject1_label,
#                                                     test_size=0.2,
#                                                     random_state=42)

subject2_train, subject2_test, subject2_label = load_data('Subject_2')
subject2_train = np.swapaxes(subject2_train,0,2)
subject2_test = np.swapaxes(subject2_test,0,2)

# X2_train, X_val, y2_train, y_val = train_test_split(subject2_train,
#                                                     subject2_label,
#                                                     test_size=0.2,
#                                                     random_state=44)

traindata = np.concatenate((subject1_train,subject2_train))  # combine to train data
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

subject6_train, subject6_test, subject6_label = load_data('Subject_6')
subject6_train = np.swapaxes(subject6_train,0,2)
subject6_test = np.swapaxes(subject6_test,0,2)
#
# traindata = np.concatenate((traindata,subject6_train))
# label = np.concatenate((label,subject6_label))


subject7_train, subject7_test, subject7_label = load_data('Subject_7')
subject7_train = np.swapaxes(subject7_train,0,2)
subject7_test = np.swapaxes(subject7_test,0,2)

traindata = np.concatenate((traindata,subject7_train))
label = np.concatenate((label,subject7_label))

 # create test & validation data
subject8_train, subject8_test, subject8_label = load_data('Subject_8')
subject8_train = np.swapaxes(subject8_train,0,2)
subject8_test = np.swapaxes(subject8_test,0,2)
print(subject8_train.shape, subject8_test.shape, subject8_label.shape)

traindata = np.concatenate((traindata,subject8_train))
label = np.concatenate((label,subject8_label))

traindata = np.swapaxes(traindata,2,1)
pca = UnsupervisedSpatialFilter(PCA(20), average=False)
pca_data = pca.fit_transform(traindata)
pca_data = np.swapaxes(pca_data,1,2)

pca_data2 = pca_data[:,0:1200,:]
X_train, X_test, y_train, y_test = train_test_split(pca_data2,
                                                    label,
                                                    test_size=0.1,
                                                    random_state=47)


def create_model():
    input_shape = (1200,20) # declare input shape
    model = models.Sequential()

    model.add(Conv1D(60, 3, activation='relu',input_shape=input_shape))   #conv1D_1
    model.add(MaxPooling1D(3))  # maxpooling

    model.add(Conv1D(60, 1, activation='relu'))
    model.add(Dropout(0.5))     # dropout 0.5
    model.add(AveragePooling1D(3))

    model.add(Conv1D(120, 2, activation='relu'))
    model.add(AveragePooling1D(3))

    model.add(Conv1D(20, 3, activation='relu'))
    model.add(AveragePooling1D(2))

    model.add(Conv1D(60, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(AveragePooling1D(3))
    #
    # model.add(Conv1D(60, 3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(AveragePooling1D(3))

    model.add(layers.Flatten())   # flatten
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))   # sigmoid dense


    return model

model = create_model()
model.summary()  # print model summary


model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])  # compile model


checkpoint_path = '/Users/josie/PycharmProjects/signal_process/t_0.84_norm/cp-{epoch:04d}.ckpt'  # declare checkpoint
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)



temp_label = np.zeros((8, 19  ))

def predict_result(num1,num2,temp_label):
    predict = model.predict(num1, verbose=0)
    for i in range(19):
        if i < len(num1):
            if predict[i] < 0.5:
                temp_label[num2][i] = 0
            else:
                temp_label[num2][i] = 1
        else:
            temp_label[num2][i] = np.nan
    return 0

def perform_PCA(testdata):
    sub1 = np.swapaxes(testdata, 2, 1)
    pca = UnsupervisedSpatialFilter(PCA(20), average=False)
    pca_data = pca.fit_transform(sub1)
    pca_data1 = np.swapaxes(pca_data, 1, 2)
    return pca_data1

sub1 = perform_PCA(subject1_test)
predict_result(sub1,0,temp_label)

sub2 = perform_PCA(subject2_test)
predict_result(sub2,1,temp_label)

sub3 = perform_PCA(subject3_test)
predict_result(sub3,2,temp_label)

sub4 = perform_PCA(subject4_test)
predict_result(sub4,3,temp_label)

sub5 = perform_PCA(subject5_test)
predict_result(sub5,4,temp_label)

sub6 = perform_PCA(subject6_test)
predict_result(sub6,5,temp_label)

sub7 = perform_PCA(subject7_test)
predict_result(sub7,6,temp_label)

sub8 = perform_PCA(subject8_test)
predict_result(sub8,7,temp_label)

mdic = {"subj": temp_label}
savemat("matlab_matrix.mat", mdic)


