from classifiers.ClassifierWrapper import ClassifierWrapper

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import pyAgrum as gum
from pyAgrum.skbn import BNClassifier
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.explain as expl
from pyAgrum.lib.bn2graph import BN2dot
from pyAgrum.lib.bn2roc import showROC, showPR, showROC_PR

class BayesNet( ClassifierWrapper ):

  def __init__(self, load_model : bool, dataset_name : str, dataset : pd.DataFrame , class_var : str):
    super().__init__(load_model, "Bayesian Network", dataset_name, dataset, class_var)

  def classify(self, save_model=False, learningMethod='MIIC', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="uniform",usePR=False):
    # learningMethod: A string designating which type of learning we want to use. Possible values are: 
    # Chow-Liu, NaiveBayes, TAN, MIIC + (MDL ou NML), 
    # GHC, 3off2 + (MDL ou NML), Tabu. 
    self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
    self.clf.fit(self.X_train, self.Y_train)

    # evaluate model
    self.evaluate_model( )
    
    if save_model:
      gum.saveBN(self.clf.bn, os.path.join(".","models", "BNC_" + self.dataset_name + ".net"))
    

  def classifyNaiveBayes(self, save_model=False, learningMethod='NaiveBayes', prior='Smoothing', priorWeight=1, discretizationNbBins=4, discretizationStrategy="uniform",usePR=False):
    self.clf_name = "Naive Bayes"
    self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy,usePR=usePR)
    self.clf.fit(self.X_train, self.Y_train)
    
    # evaluate model
    self.evaluate_model( )
    
    if save_model:
      gum.saveBN(self.clf.bn, os.path.join(".","models", self.clf_name.replace(" ", "") + "_" + self.dataset_name + ".net"))
    
    return self.clf
    
  def show_bn(self):
    gnb.showBN( self.clf.bn )

  def show_query_mode(self):
    gnb.showInference( self.clf.bn )

  def getBN(self):
    return super().clf.bn

  def getClassVar(self):
    return super().getClassVar()

  # MODEL EXPLANATIONS

  # 1 - INDEPENDENCE LISTS
  
  # Given a model, it may be interesting to investigate the conditional 
  # independences of the class Y created by this very model with respect 
  # to the class variable
  def explainCondInd(self, data_path):
    expl.independenceListForPairs(self.clf.bn,data_path, target=self.class_var)

  # 2 - SHAP VALUES
  # The ShapValue class implements the calculation of Shap values 
  # in Bayesian networks. It is necessary to specify a target and 
  # to provide a Bayesian network whose parameters are known and will 
  # be used later in the different calculation methods.
  # The result is returned as a dictionary, the keys are the names of 
  # the features and the associated value is the absolute value of the average of the calculated shap.
  def explainShapCondInd(self, isPlot=True, isPlot_importance=True, isPercentage=False ):
    gumshap = expl.ShapValues(self.clf.bn, super().getClassVar())
    resultat = gumshap.conditional(self.X_train, plot=isPlot, plot_importance=isPlot_importance, percentage=isPercentage)
    return resultat

  # This method is similar to the previous one, except the formula of computation. 
  # It computes the causal shap value as described in the paper of Heskes Causal Shapley Values: 
  # Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models .
  def explainCausalShap(self, isPlot=True, isPlot_importance=True, isPercentage=False):
    gumshap = expl.ShapValues(self.clf.bn, super().getClassVar())
    causal = gumshap.causal(self.X_train, plot=isPlot, plot_importance=isPlot_importance, percentage=isPercentage)
    return causal
  
  # same as the first method, but takes into consideration the vars in the Markov Blanket
  def explainMarginalShap(self, isPlot=True, isPlot_importance=True, isPercentage=False):
    gumshap = expl.ShapValues(self.clf.bn, super().getClassVar())
    marginal = gumshap.marginal(self.X_train, sample_size=10, plot=isPlot, plot_importance=isPlot_importance, percentage=isPercentage)
    return marginal

  # This method returns a coloured graph that makes it easier to 
  # understand which variable is important and where it is located in the graph.
  def explainVizShapValues(self, shap_type="conditional", isPlot=True, isPlot_importance=True, isPercentage=False):
    g = None
    if( shap_type == "conditional"):
      g = self.explainShapCondInd(isPlot=isPlot, isPlot_importance=isPlot_importance, isPercentage=isPercentage )

    if( shap_type == "causal"):
      g = self.explainCausalShap(isPlot=isPlot, isPlot_importance=isPlot_importance, isPercentage=isPercentage )

    if( shap_type == "marginal"):
      g = self.explainMarginalShap(isPlot=isPlot, isPlot_importance=isPlot_importance, isPercentage=isPercentage )
    
    gnb.showGraph(g)

  def explainVizEntropy(self):
    expl.showInformation(self.clf.bn)
    





  
  

    


        

