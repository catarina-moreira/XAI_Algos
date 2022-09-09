
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod
from data.DatasetLoader import Dataset

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Learner(ABC):
    
    def __init__(self, loadModel, clf_name : str, data : Dataset):
        """
		Args:
			loadModel (bool) : 
			clf_name (str) : 
			data (Dataset) :  
        """
        self.loadSavedModel = loadModel
        self.clf_name = clf_name
        self.data : Dataset = data
        
        self.clf = None       
        self.predictions = None 
        self.clf_results = {}
        
        self.history = None
        self.Xtrain, self.Ytrain = None, None 
        self.Xtest, self.Ytest, self.Ypred = None, None, None
        self.Xval, self.Yval = None, None
        self.modelPath = None
        self.decisionBoundary = None
        self.debug = None

    def plot_decision_boundary(self, flag=False, figsize = (5,4), dpi = 150, size = 40, step=0.1, cmap = plt.cm.RdYlBu ): 
        x_min, x_max = self.data.X.iloc[:, 0].min() - step, self.data.X.iloc[:,0].max() + step
        y_min, y_max = self.data.X.iloc[:, 1].min() - step, self.data.X.iloc[:, 1].max() + step

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        x_in = np.c_[xx.ravel(), yy.ravel()]

        if flag:
            col = np.zeros( (len(x_in) ,1))
            x_in = np.concatenate([x_in, col], axis=1)

        self.debug = x_in
        y_pred = self.clf.predict_proba(x_in)[:,1]
        y_pred = np.round(y_pred).reshape(xx.shape)

        fig = Figure(figsize=figsize, dpi=dpi) 
        ax = (fig.subplots(1, 1, sharex=True, sharey=True))
        ax.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.7 )
        ax.scatter(self.Xtest[:,0], self.Xtest[:, 1], c=self.Ytest, s=size,alpha=0.6, cmap=cmap, marker=".")
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        ax.set_xlabel(self.data.feature_names[0])
        ax.set_ylabel(self.data.feature_names[1])
        ax.set_title( self.clf_name + " | Dataset: " + self.data.dataset_name + "\nPrecision: " + str(self.clf_results["precision"]) + " | Recall: " + str(self.clf_results["recall"]))

        self.decisionBoundary = os.path.join(".", "tmp", "networks", self.clf_name + "_" + self.data.dataset_name + ".png") 
        fig.savefig(self.decisionBoundary, dpi=dpi )

        return fig
        
   
    @abstractmethod
    def evaluate(self) -> None:
        pass
    
    @abstractmethod
    def generateTrainingData():
        pass
    
    @abstractmethod
    def applyClassifier(self) -> None:
        pass
    
    @abstractmethod
    def loadModel(self):
        pass
    
    @abstractmethod
    def saveModel(self) -> None:
        pass

	# Getters and Setters -------------------------------------------
    def setLoadSavedModel(self, loadSavedModel : bool):
        self.loadSavedModel = loadSavedModel
    
    def getLoadSavedModel(self) -> bool:
        return self.loadSavedModel
    
    def setClf_name(self, clf_name : str):
        self.clf_name = clf_name
        
    def getClf_name(self) -> str:
        return self.clf_name


