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
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split



##TODO: PCA


################## import data#######################################

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

X_train, X_test, y_train, y_test = train_test_split(traindata,
                                                    label,
                                                    test_size=0.15,
                                                    random_state=50)


# traindata = np.swapaxes(traindata,2,1)
# pca = UnsupervisedSpatialFilter(PCA(427), average=False)
# pca_data = pca.fit_transform(traindata)
# pca_data = np.swapaxes(pca_data,1,2)

######################building model####################

def create_model():
    input_shape = (1200,60) # declare input shape
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

def create_model2():
    input_shape = (1200,60) # declare input shape
    l1 = 0
    model = models.Sequential()

    model.add(Conv1D(60, 3, activation='relu',input_shape=input_shape))   #conv1D_1
    model.add(MaxPooling1D(3))  # maxpooling
    model.add(Dropout(0.2))  # dropout 0.5

    model.add(Conv1D(60, 1, activation='relu'))
    model.add(AveragePooling1D(3))
    model.add(Dropout(0.2))  # dropout 0.5

    model.add(Conv1D(120, 2, activation='relu'))
    model.add(AveragePooling1D(3))
    model.add(Dropout(0.2))  # dropout 0.5


    # model.add(Conv1D(60, 3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(AveragePooling1D(3))
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
BATCH_SIZE = 24
# STEPS_PER_EPOCH = labels.size / BATCH_SIZE
SAVE_PERIOD = 4
checkpoint_path = 'D:/Josie/23spring/signal_process/training_1/cp.ckpt'  # declare checkpoint
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])  # compile model


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    #save_freq= int(SAVE_PERIOD * STEPS_PER_EPOCH),
    save_freq=4)
# TODO: learnning rate
# TODO: early stop, cp_callback
# callbacks=[cp_callback],

input_data = X_train
y_train = y_train
history = model.fit(input_data, y_train,callbacks=[cp_callback],batch_size=16,epochs=80, validation_data=(X_test, y_test))



###### plot accuracy #####################
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1])
plt.legend(loc='lower right')
plt.show()

acc = model.evaluate(traindata, label)
print("Loss:", acc[0], " Accuracy:", acc[1])

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

predict = model.predict(subject1_test, verbose=0)


