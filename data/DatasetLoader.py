
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

import seaborn as sns

class Dataset():

	def __init__(self, aDataset_name : str, aDataset_path : str, aData_gen_eq : str, aClass_var : str):	
		self.dataset_name : str = aDataset_name
		self.dataset_path : str = aDataset_path
		self.class_var : str = aClass_var

		# a string in latex form that describes the equations that generated the data 
		self.data_gen_eq : str = aData_gen_eq

		# extract information from dataset
		self.process_data()

	# PROCESSDATA
	def process_data(self, aNorm_feat : bool = False, aEnc_class : bool = False) -> None:

		self.X = None
		self.Y = None
	
		self.dataset = pd.read_csv( self.dataset_path )
		self.class_labels = self.dataset[self.class_var].unique().tolist()
		self.numClasses = len(self.class_labels)

		self.X = self.dataset.drop( self.class_var, axis=1)
		self.Y = self.dataset[self.class_var]

		self.feature_names = self.X.columns.tolist()
		self.numFeatures = len(self.feature_names)
	
	# VISUALIZE DATA
	def visualize_data2D(self, figsize=(4,4), scatter_size = 2, savefig=True) -> None:
	
		plt.figure(figsize=figsize)
		colors = cm.RdYlBu(np.linspace(0, 1, self.numClasses))
		for c in range(self.numClasses):
			data = self.dataset[ self.dataset[self.class_var] == self.class_labels[c]].values
			plt.scatter(data[:, 0], data[:, 1], label="Class " + str(self.class_labels[c]), s=scatter_size, color=colors[c])
			
		plt.title(self.dataset_name)
		plt.ylabel(self.feature_names[1])
		plt.xlabel(self.feature_names[0])
		plt.legend()
		plt.grid(False)

		if savefig:
			plt.savefig( os.path.join("tmp", "data", self.dataset_name + "_dataset.png"), dpi = 150) 

	def correlation_matrix(self, figsize = (4,4), cmap = 'RdBu', savefig=True, text_size = 8):
		corr_matrix = self.dataset.corr()
		mask = np.zeros_like(corr_matrix, dtype=np.bool)
		mask[np.triu_indices_from(mask)]= True
		
		f, ax = plt.subplots(figsize=figsize) 
		sns.set_style("white")
		heatmap = sns.heatmap(corr_matrix, mask = mask, square = True, linewidths = .5,cbar_kws={"shrink": .9},
                      cmap = cmap, vmin = -1,  vmax = 1, annot = True, annot_kws = {"size": text_size})

		ax.set_yticklabels(corr_matrix.columns, rotation = 0)
		ax.set_xticklabels(corr_matrix.columns)
		sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

		if savefig:
			plt.savefig( os.path.join("tmp", "corr", self.dataset_name + "_corr.png"), dpi = 150) 

		return corr_matrix, heatmap
		

	# GETTERS and SETTERS --------------------------------------
	def setDataset_name(self, aDataset_name : str):
		self.dataset_name = aDataset_name

	def getDataset_name(self) -> str:
		return self.dataset_name

	def setDataset_path(self, aDataset_path : str):
		self.dataset_path = aDataset_path

	def getDataset_path(self) -> str:
		return self.dataset_path

	def setData_gen_eq(self, aData_gen_eq : str):
		self.data_gen_eq = aData_gen_eq

	def getData_gen_eq(self) -> str:
		return self.data_gen_eq

	def setClass_var(self, aClass_var : str):
		self.class_var = aClass_var

	def getClass_var(self) -> str:
		return self.class_var

	def setClass_labels(self, aClass_labels):
		self.class_labels = aClass_labels

	def getClass_labels(self):
		return self.class_labels

	def setFeature_names(self, aFeature_names):
		self.feature_names = aFeature_names

	def getFeature_names(self):
		return self.feature_names

	def setX(self, aX):
		self.X = aX

	def getX(self):
		return self.X

	def setY(self, aY):
		self.Y = aY

	def getY(self):
		return self.Y

