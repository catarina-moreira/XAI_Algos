from classifiers.ClassifierWrapper import ClassifierWrapper

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from xgboost import XGBClassifier


class XGBoost( ClassifierWrapper ):

  def __init__(self, load_model : bool, dataset_name : str, dataset : pd.DataFrame , class_var : str):
    super().__init__(load_model, "XGBoost", dataset_name, dataset, class_var)

  def classify(self, save_model = False, learning_rate=0.01, max_depth=5, n_estimators=200, 
                           min_child_weight = 10, subsample = 0.8, early_stopping = 10):
    
    # train model
    self.clf = XGBClassifier(objective ='multi:softmax', num_class = 2, learning_rate = learning_rate, 
                             max_depth = max_depth, n_estimators = n_estimators, min_child_weight = min_child_weight, 
                             subsample = subsample, early_stopping_rounds = early_stopping)
    self.clf.fit(self.X_train, self.Y_train, eval_set=[(self.X_train, self.Y_train), (self.X_val, self.Y_val)], verbose=True)

    # evaluate model
    self.evaluate_model( )

    # save model
    if save_model:
      self.clf.save_model(os.path.join(".","models", "XGB_" + self.dataset_name + ".json"))

    return self.clf
  


  






