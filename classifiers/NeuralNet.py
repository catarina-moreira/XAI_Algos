from classifiers.ClassifierWrapper import ClassifierWrapper

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class NeuralNetwork( ClassifierWrapper ):
  
  def __init__(self, load_model : bool, dataset_name : str, dataset : pd.DataFrame , class_var : str):
    super().__init__(load_model, "Neural Network", dataset_name, dataset, class_var)

    if load_model:
      self.clf = keras.models.load_model(os.path.join(".","models", "NN_" + self.dataset_name + ".json"))

  def classify(self,save_model = False, act_fn="tanh", batch_size=32, epochs=50, learning_rate=0.01):
    
    # define model
    self.clf = self.create_nn_arch(act_fn=act_fn)

    # train
    self.clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    self.history = self.clf.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.Y_val), verbose=1)

    # evaluate model
    self.evaluate_model( )

    # save model
    if save_model:
      self.clf.save(os.path.join(".","models", "NN_" + self.dataset_name + ".json"))

  def create_nn_arch(self, act_fn):
    nn = tf.keras.Sequential()
    nn.add(layers.Dense(7, activation=act_fn, input_shape=(self.X_train.shape[-1],) ))
    nn.add(layers.Dense(5, activation=act_fn ))
    nn.add(layers.Dense(3, activation=act_fn ))
    nn.add(layers.Dense(2, activation="softmax"))

    return nn

  def plotLearningHistory(self):
    logs = pd.DataFrame(self.history.history)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(logs.loc[5:,"loss"], lw=2, label='training loss')
    plt.plot(logs.loc[5:,"val_loss"], lw=2, label='validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(logs.loc[5:,"accuracy"], lw=2, label='training accuracy score')
    plt.plot(logs.loc[5:,"val_accuracy"], lw=2, label='validation accuracy score')
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()

  def getClassVar(self):
    return super().getClassVar()
