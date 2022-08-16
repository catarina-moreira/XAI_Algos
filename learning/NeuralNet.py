from abc import ABCMeta, abstractmethod
from data import DatasetLoader
from learning.Learner import Learner
from typing import List

class NeuralNet(Learner):

	__metaclass__ = ABCMeta
	
	@classmethod
	def __init__(self, loadModel : bool, dataset : DatasetLoader):
		super().__init__(loadModel, "Neural Network", dataset)

	@abstractmethod
	def applyClassifier(self) -> None:
		pass

	@abstractmethod
	def loadModel(self):
		pass

	@abstractmethod
	def saveModel(self) -> None:
		pass