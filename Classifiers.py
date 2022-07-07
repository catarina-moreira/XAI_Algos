
import sys
import os
import numpy as np
import pandas as pd

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from pyAgrum.skbn import BNClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# for binary classification only
class Classifier:

    SUPPORTED_CLFS = ["XGBoost", "Neural Network", "Bayesian Network"]

    def __init__(self, load_model : bool, clf_name : str, dataset_name : str, dataset : pd.DataFrame , class_var : str):

        self.clf_name = clf_name
        self.clf = None       
        self.predictions = None 
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_var = class_var
        self.clf_results = {}
        self.history = None

        # extract feature names
        self.feature_names = dataset.columns.to_list()
        self.feature_names.remove(class_var)

        # generate class label
        self.class_labels = ["Y_0", "Y_1"]

        # genetate training files
        X_train, X_test, Y_train, Y_test, X_val, Y_val = self.generateTrainTestValSets()

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val

        # load a pretrained model
        if load_model:
            self.clf = self.load_pretrained_model()

    def load_pretrained_model( ):
        

    def applyClassifer(self, save_model = True, learning_rate=0.01, max_depth=5, 
                        n_estimators=200, min_child_weight = 10, subsample = 0.8, early_stopping = 10, 
                        learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4,
                        discretizationStrategy="kmeans",usePR=False, act_fn = "tanh", batch_size=32, epochs=50):
        
        clf = None
        self.errorClassifierNotFOund() if self.clf_name not in self.SUPPORTED_CLFS else ""

        if self.clf_name == "XGBoost":
           clf = self.applyXGBoost( save_model = save_model, learning_rate=learning_rate, max_depth=max_depth, 
                                    n_estimators=n_estimators,  min_child_weight = min_child_weight, 
                                    subsample = subsample, early_stopping = early_stopping )
        
        if self.clf_name == "Bayesian Network":
            clf = self.applyBN(save_model=save_model, learningMethod=learningMethod, prior=prior, 
                                priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, 
                                discretizationStrategy=discretizationStrategy,usePR=usePR)
        
        if self.clf_name == "Neural Network":
            clf = self.applyNN(save_model=save_model, act_fn=act_fn, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
        
        return clf
    

    def applyNN(self,save_model, act_fn, batch_size, epochs, learning_rate):
        
        # define model
        self.clf = self.create_nn_arch(act_fn)

        # train
        self.clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
        self.history = self.clf.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.Y_val), verbose=1)

        # evaluate model
        self.evaluate_model( )

        # save model
        if save_model:
            self.clf.save(os.path.join(".","models", "NN_" + self.dataset_name + ".json"))
        
        return self.clf

    def create_nn_arch(self, act_fn):
        nn = tf.keras.Sequential()
        nn.add(layers.Dense(7, activation=act_fn, input_shape=(self.X_train.shape[-1],) ))
        nn.add(layers.Dense(5, activation=act_fn ))
        nn.add(layers.Dense(3, activation=act_fn ))
        nn.add(layers.Dense(2, activation="softmax"))

        return nn


    ########################################################
    # BAYESIAN NETWORK CLASSIFER
    #
    def applyBN(self, save_model=True, learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="kmeans",usePR=False ):
        
        self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
        self.clf.fit(self.X_train, self.Y_train)
        
        # evaluate model
        self.evaluate_model( )

        if save_model:
            gum.saveBN(self.clf.bn, os.path.join(".","models", "BNC_" + self.dataset_name + ".net"))
        
        return self.clf

    ########################################################
    # XGBOOST CLASSIFER
    #
    def applyXGBoost( self, save_model = True, learning_rate=0.01, max_depth=5, n_estimators=200, 
                      min_child_weight = 10, subsample = 0.8, early_stopping = 10 ):

        # apply classifier
        self.clf = XGBClassifier(objective ='multi:softmax', num_class = 2, learning_rate = learning_rate, max_depth = max_depth,
                            n_estimators = n_estimators, min_child_weight = min_child_weight, subsample = subsample,
                            early_stopping_rounds = early_stopping)
        self.clf.fit(self.X_train, self.Y_train, eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], verbose=True)

        # evaluate the model
        self.evaluate_model( )

        # save model
        if save_model:
            self.clf.save_model(os.path.join(".","models", "XGB_" + self.dataset_name + ".json"))

        return self.clf

    def evaluate_model(self):
        
        Y_pred = self.clf.predict(self.X_test)

        if self.clf_name == "Neural Network":
            Y_pred = list(map( np.argmax, Y_pred ))
            self.Y_test = list(map( np.argmax, self.Y_test))

        self.clf_results["predictions"] = Y_pred

        self.clf_results["accuracy"] = accuracy_score(self.Y_test, Y_pred) 
        self.clf_results["precision"] = precision_score(self.Y_test, Y_pred)
        self.clf_results["recall"] = recall_score(self.Y_test, Y_pred)
        self.clf_results["f1"] = f1_score(self.Y_test, Y_pred)
  
        # show perfomance
        print("Overall Performace: ")
        print("\tClassifier: " + self.clf_name)
        print("\tAccuracy: " + str(self.clf_results["accuracy"]))
        print("\tPrecision: " + str(self.clf_results["precision"]))
        print("\tRecall: " + str(self.clf_results["recall"]))
        print("\tF1 Score: " + str(self.clf_results["f1"]))

    def generateTrainTestValSets(self):
        
        X = self.dataset[self.feature_names]
        Y = self.dataset[self.class_var]

        # need to convert to categorical representation, because ...
        if self.clf_name == "Neural Network":
            Y = pd.DataFrame(to_categorical( Y ), columns=self.class_labels) # LIME works with prediction probabilities which are only supported by softmax act. function in NNs

        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=515)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=515)

        return X_train, X_test, Y_train, Y_test, X_val, Y_val
        

    def errorClassifierNotFOund(self):
        print("[ERROR] Classifier " + self.clf_name + {" is not supported. Please try one of the following:"})
        print(self.SUPPORTED_CLFS)
        return sys.exit(-1)


    ##################################################
    # GETTERS
    #################################################
    
    def getPredictions(self):
        return self.predictions

    def getFeatureNames(self):
        return self.feature_names

    def getClassVar(self):
        return self.class_var

    def getHistory(self):
        return self.history

    def getResults(self):
        return self.clf_results

    def getTrainingData(self):
        return self.X_train, self.Y_train
    
    def getTestData(self):
        return self.X_test, self.Y_test

    def getValidationData(self):
        return self.X_val, self.Y_val

    def getClassifier(self):
        return self.clf

    def getClfName(self):
        return self.clf_name
    
    def getDataset(self):
        return self.dataset

    def getClassVar(self):
        return self.class_var

    ##################################################
    # sETTERS
    #################################################
    def setClfName(self, newClfName):
        self.clf_name = newClfName
    
    def setDataset(self, newDataset):
        self.dataset = newDataset
