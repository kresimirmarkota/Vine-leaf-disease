import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import tensorflow as tf
import time
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense,Flatten




NAME = "Bolesti_vinove_loze_model-{}".format(int(time.time()))
tensBrd = tf.keras.callbacks.TensorBoard(log_dir="logs\{}".format(NAME), histogram_freq=1)

X = pickle.load(open("X_googleNet.pickle","rb"))
y = pickle.load(open("y_googleNet.pickle","rb"))
IMG_SIZE=100
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X=X/255.0

dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

n=0
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME1="{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

            model = tf.keras.Sequential()

            model.add(tf.keras.layers.Conv2D((layer_size),(3,3),padding="same",input_shape=X.shape[1:]))
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.BatchNormalization(-1))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
            model.add(tf.keras.layers.Dropout(0.25))

            for l in range(conv_layer-1):
                model.add(tf.keras.layers.Conv2D((layer_size),(3,3),padding="same"))
                model.add(tf.keras.layers.Activation("relu"))
                model.add(tf.keras.layers.BatchNormalization(-1))
                model.add(tf.keras.layers.Conv2D((layer_size),(3,3),padding="same"))
                model.add(tf.keras.layers.Activation("relu"))
                model.add(tf.keras.layers.BatchNormalization(-1))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
                model.add(tf.keras.layers.Dropout(0.25))


            model.add(tf.keras.layers.Flatten())

            for l in range(dense_layer):
                model.add(tf.keras.layers.Dense(layer_size))
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(0.25))

            model.add(tf.keras.layers.Dense(4))
            model.add(tf.keras.layers.Activation("softmax"))


            earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
            mcp_save = ModelCheckpoint('{}-conv-{}-nodes-{}-dense-{}-.mdl_wts.hdf5'.format(conv_layer,layer_size,dense_layer,int(time.time())), save_best_only=True, monitor='val_loss', mode='min')
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
            model.compile(loss="sparse_categorical_crossentropy",optimizer="adam", metrics=['accuracy'])

            history=model.fit(X,y,batch_size=32, epochs=25,validation_split=0.3,callbacks=[tensBrd,earlyStopping,mcp_save,reduce_lr_loss],shuffle=True)
            model.save("{}-conv-{}-nodes-{}-dense-{}-my_model.model".format(conv_layer,layer_size,dense_layer,int(time.time())))

            with open('trainHistoryDict-{}-conv-{}-nodes-{}-dense-{}.pickle'.format(conv_layer,layer_size,dense_layer,int(time.time())), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(acc) + 1)
            n=n+1
            plt.figure(n)
            plt.plot(epochs, acc, 'r', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.savefig('training-validation-accuracy--{}-conv-{}-nodes-{}-dense-{}.png'.format(conv_layer,layer_size,dense_layer,int(time.time())))
            print("-----------------------------------------------------------------")
            n=n+1
            plt.figure(n)
            plt.plot(epochs, loss, 'g', label='Training loss')
            plt.plot(epochs, val_loss, 'y', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.savefig('training-validation-loss--{}-conv-{}-nodes-{}-dense-{}.png'.format(conv_layer,layer_size,dense_layer,int(time.time())))