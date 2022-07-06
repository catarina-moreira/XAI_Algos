
import sys
import os
import pandas as pd

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from pyAgrum.skbn import BNClassifier

from keras.utils.np_utils import to_categorical

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# for binary classification only
class Classifier:

    SUPPORTED_CLFS = ["XGBoost", "Neural Network", "Bayesian Network"]

    def __init__(self, clf_name : str, dataset_name : str, dataset : pd.DataFrame , class_var : str):

        self.clf_name = clf_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_var = class_var
        self.clf_results = {}

        # extract feature names
        self.feature_names = dataset.columns.to_list()
        self.feature_names.remove(class_var)

        # generate class label
        self.class_label = ["Y_0", "Y_1"]

        # genetate training files
        X_train, X_test, Y_train, Y_test, X_val, Y_val = self.generateTrainTestValSets()

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val

    def applyClassifer(self, save_model = True, learning_rate=0.01, max_depth=5, 
                        n_estimators=200, min_child_weight = 10, subsample = 0.8, early_stopping = 10, 
                        learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4,
                        discretizationStrategy="kmeans",usePR=False):
        
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
            pass
        
        
        return clf
            
    def applyBN(self, save_model=True, learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="kmeans",usePR=False ):
        
        bnc = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
        bnc.fit(self.X_train, self.Y_train)

        # evaluate model
        self.evaluate_model(bnc)

        if save_model:
            gum.saveBN(bnc.bn, os.path.join(".","models", "BNC_" + self.dataset_name + ".net"))
        
        return bnc

    def applyXGBoost( self, save_model = True, learning_rate=0.01, max_depth=5, n_estimators=200, 
                      min_child_weight = 10, subsample = 0.8, early_stopping = 10 ):

        # apply classifier
        xgb = XGBClassifier(objective ='multi:softmax', num_class = 2, learning_rate = learning_rate, max_depth = max_depth,
                            n_estimators = n_estimators, min_child_weight = min_child_weight, subsample = subsample,
                            early_stopping_rounds = early_stopping)
        xgb.fit(self.X_train, self.Y_train, eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], verbose=True)

        # evaluate the model
        self.evaluate_model(xgb)

        # save model
        if save_model:
            xgb.save_model(os.path.join(".","models", "XGB_" + self.dataset_name + ".json"))

        return xgb

    def evaluate_model(self, clf):

        Y_pred = clf.predict(self.X_test)
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
            Y = to_categorical( Y ) # LIME works with prediction probabilities which are only supported by softmax act. function in NNs

        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=515)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=515)

        return X_train, X_test, Y_train, Y_test, X_val, Y_val
        

    def errorClassifierNotFOund(self):
        print("[ERROR] Classifier " + self.clf_name + {" is not supported. Please try one of the following:"})
        print(self.SUPPORTED_CLFS)
        return sys.exit(-1)


    def getResults(self):
        return self.clf_results

    def getTrainingData(self):
        return self.X_train, self.Y_train
    
    def getTestData(self):
        return self.X_test, self.Y_test

    def getValidationData(self):
        return self.X_val, self.Y_val

    def getClfName(self):
        return self.clf_name
    
    def getDataset(self):
        return self.dataset

    def getClassVar(self):
        return self.class_var

    def setClfName(self, newClfName):
        self.clf_name = newClfName
    
    def setDataset(self, newDataset):
        self.dataset = newDataset
