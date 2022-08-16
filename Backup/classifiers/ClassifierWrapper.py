
import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_auc_score
from keras.utils.np_utils import to_categorical

# for binary classification only
class ClassifierWrapper:

    SUPPORTED_CLFS = ["XGBoost", "Neural Network", "Bayesian Network"]

    def __init__(self, load_model : bool, clf_name : str, dataset_name : str, dataset : pd.DataFrame , class_var : str):

        self.clf_name = clf_name
        self.clf = None       
        self.predictions = None 
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_var = class_var
        self.clf_results = {}
        
        self.history,self.X, self.y,self.Y = None, None, None, None

        # extract feature names
        self.feature_names = dataset.columns.to_list()
        self.feature_names.remove(class_var)

        # generate class label
        self.class_labels = ["Y_0", "Y_1"]

        # genetate training files
        self.generateTrainTestValSets()

        
    def applyClassifer(self):
        pass

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
        self.clf_results["roc_auc_score"] = roc_auc_score(self.Y_test, Y_pred)
        self.clf_results["confusion_matrix"] = confusion_matrix(self.Y_test, Y_pred)

        if self.clf_name == "Neural Network":
            self.clf_results["history"] = self.history
        
        # show perfomance
        print("Overall Performace: ")
        print("\tClassifier: " + self.clf_name)
        print("\tAccuracy: " + str(self.clf_results["accuracy"]))
        print("\tPrecision: " + str(self.clf_results["precision"]))
        print("\tRecall: " + str(self.clf_results["recall"]))
        print("\tF1 Score: " + str(self.clf_results["f1"]))
        print("\tROC AUC Score: " + str(self.clf_results["roc_auc_score"]) )

        # save results
        with open(os.path.join("results", self.clf_name.replace(" ", "")+"_RES_" + self.dataset_name + ".pkl"), 'wb') as f:
            pickle.dump(self.clf_results, f)

    def generateTrainTestValSets(self):
        
        self.X = self.dataset[self.feature_names]
        self.y = self.dataset[self.class_var] # keeping a copy of the original labels. it will be handy for later
        self.Y = self.dataset[self.class_var]

        # need to convert to categorical representation, because ...
        if self.clf_name == "Neural Network":
            self.Y = pd.DataFrame(to_categorical( self.Y ), columns=self.class_labels) # LIME works with prediction probabilities which are only supported by softmax act. function in NNs

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X.values, self.Y.values, test_size=0.3, random_state=515)
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(self.X_test, self.Y_test, test_size=0.5, random_state=515)
        

    def plot_decision_boundary(self, figsize=list([5, 5]), colormap = plt.cm.RdBu):

        h = .02  # step size in the mesh
        fs=6
        
        data_0 = self.dataset[self.class_var == 0]
        data_1 = self.dataset[self.class_var == 0]

        plt.figure(figsize=(4, 4))
        plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
        plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
        plt.title("Dataset samples")
        plt.ylabel(r"$x_2$")
        plt.xlabel(r"$x_1$")
        plt.legend()

        #x_min, x_max = self.X.iloc[:, 0].values.min() - .1, (self.X.iloc[:, 0].values).max() + .1
        #y_min, y_max = self.X.iloc[:, 1].values.min() - .1, self.X.iloc[:, 1].values.max() + .1
        #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        #cm = colormap
        #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplots(1, 1, figsize=figsize )

        #if(self.clf_name == "Neural Network"):
        #    Z = np.array( list( map(np.argmax, self.clf.predict(np.c_[xx.ravel(), yy.ravel()]))))
        #else:
        #    Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        #ax.contourf(xx, yy, Z, cmap=cm, alpha=.7)

        # Plot the training points
        #if self.clf_name == "Neural Network":
        #    ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=np.array(list(map(np.argmax, self.Y_train))), cmap=cm_bright, edgecolors='k', alpha=0.2,marker='.')
        #    ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=np.array(list(map(np.argmax, self.Y_test))), cmap=cm_bright, edgecolors='k',marker='.')
        #else:
        #    ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.Y_train, cmap=cm_bright, edgecolors='k', alpha=0.2,marker='.')
        #    ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.Y_test, cmap=cm_bright, edgecolors='k',marker='.')

        #ax.set_xlim(xx.min(), xx.max())
        #ax.set_ylim(yy.min(), yy.max())
        #ax.set_xticks(())
        #ax.set_yticks(())

        #plt.tight_layout()
        #plt.show()





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
