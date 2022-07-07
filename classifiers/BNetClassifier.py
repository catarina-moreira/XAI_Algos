from classifiers.ClassifierWrapper import ClassifierWrapper

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import pyAgrum as gum
from pyAgrum.skbn import BNClassifier
import pyAgrum.lib.notebook as gnb
from pyAgrum.lib.bn2roc import showROC, showPR, showROC_PR

class BayesNet( ClassifierWrapper ):

  def __init__(self, load_model : bool, dataset_name : str, dataset : pd.DataFrame , class_var : str):
    super().__init__(load_model, "Bayesian Network", dataset_name, dataset, class_var)

  def classify(self, save_model=False, learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="kmeans",usePR=False):
    # learningMethod: A string designating which type of learning we want to use. Possible values are: 
    # Chow-Liu, NaiveBayes, TAN, MIIC + (MDL ou NML), 
    # GHC, 3off2 + (MDL ou NML), Tabu. 
    self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
    self.clf.fit(self.X_train, self.Y_train)

  def classifyNaiveBayes(self, save_model=False, learningMethod='NaiveBayes', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="kmeans",usePR=False):
    self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
    self.clf.fit(self.X_train, self.Y_train)
    
    # evaluate model
    self.evaluate_model( )
    
    if save_model:
      gum.saveBN(self.clf.bn, os.path.join(".","models", "BNC_" + self.dataset_name + ".net"))
    
    return self.clf

  def plot_feature_importances(self):
    n_features = len(self.feature_names)
    plt.barh(range(n_features), self.clf.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), self.X)
    plt.xlabel("importance")
    plt.ylabel("features")
    plt.show()

def show_bn(self):
  gnb.showBN( self.clf.bn )
        

