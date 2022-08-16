from data.DatasetLoader import Dataset
from learning.Learner import Learner

from sklearn.model_selection import train_test_split

import pickle
import os

import numpy as np

import pyAgrum as gum
from pyAgrum.skbn import BNClassifier
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.explain as expl
from pyAgrum.lib.bn2graph import BN2dot
from pyAgrum.lib.bn2roc import showROC, showPR, showROC_PR

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_auc_score

class BayesianNet(Learner):

	def __init__(self, isLoadModel, dataset : Dataset):
		super().__init__(isLoadModel, "Bayesian Network", dataset)
		self.modelPath = os.path.join("saved_models", self.clf_name.replace(" ", "") + "_" + self.data.dataset_name + ".net")
		self.resultsPath = os.path.join("results", self.clf_name.replace(" ", "")+"_RES_" + self.data.dataset_name + ".pkl")

		self.generateTrainingData()

		self.loadModel() if isLoadModel else None

	# APPLYCLASSIFIER
	def applyClassifier(self, saveModel = False, learningMethod = "MIIC", prior = "Smoothing", priorWeight=1, discretizationNbBins=4, discretizationStrategy="uniform",usePR=False) -> None:
		"""
		learningMethod: A string designating which type of learning we want to use. 
		Possible values are: 
    \tChow-Liu, NaiveBayes, TAN, MIIC + (MDL ou NML), GHC, 3off2 + (MDL ou NML), Tabu. 
		"""
		self.clf = BNClassifier(learningMethod=learningMethod, prior=prior, priorWeight=priorWeight, 
														discretizationNbBins=discretizationNbBins, discretizationStrategy=discretizationStrategy, usePR=usePR)
		self.clf.fit(self.Xtrain, self.Ytrain)
		self.evaluate( )

		self.clf_name = self.clf_name + "_" + learningMethod

		if saveModel:
			self.modelPath = self.modelPath.replace(".net", "_" + learningMethod + ".net")
			self.saveModel()

	# GENERATETRAININGDATA
	def generateTrainingData(self, test_size = 0.3) -> None:
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.data.X.values, self.data.Y.values, test_size=test_size, random_state=515)
		self.Xtest, self.Xval, self.Ytest, self.Yval = train_test_split(self.Xtest, self.Ytest, test_size=0.5, random_state=515)

	#EVALUATE
	def evaluate(self) -> None:
		self.Ypred = self.clf.predict(self.Xtest)

		self.clf_results["accuracy"] = np.round(accuracy_score(self.Ytest, self.Ypred),4)
		self.clf_results["precision"] = np.round(precision_score(self.Ytest, self.Ypred),4)
		self.clf_results["recall"] = np.round(recall_score(self.Ytest, self.Ypred),4)
		self.clf_results["f1"] = np.round(f1_score(self.Ytest, self.Ypred),4)
		self.clf_results["roc_auc_score"] = np.round(roc_auc_score(self.Ytest, self.Ypred),4)
		self.clf_results["confusion_matrix"] = np.round(confusion_matrix(self.Ytest, self.Ypred),4)
		self.clf_results["prediction"] = np.round(self.Ypred,4)

	# SAVE MODEL
	def saveModel(self) -> None:
		print("Writing model to " + self.modelPath)
		gum.saveBN(self.clf.bn, self.modelPath)

		# save results
		with open(self.resultsPath, 'wb') as f:
			pickle.dump(self.clf_results, f)

	# LOAD MODEL
	def loadModel(self) -> None:
		#self.clf = gum.loadBN(self.modelPath)
		self.clf = gum.loadBN( pickle.load(self.modelPath.replace(".net", ".pickle")) )
		self.clf_results = pickle.load(open(self.resultsPath, 'rb'))
		
	# SHOW BN
	def showBN(self):
		gnb.showBN( self.clf.bn )
	
	#SHOW QUERY MODE
	def showQueryMode(self):
		gnb.showInference( self.clf.bn )
	


