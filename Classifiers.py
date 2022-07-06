
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from xgboost import XGBClassifier

# for binary classification only
class Classifier:

    SUPPORTED_CLFS = ["XGBoost", "Neural Network", "Bayesian Network"]

    def __init__(self, clf_name : str, dataset : pd.DataFrame , class_var : str):

        self.clf_name = clf_name
        self.dataset = dataset
        self.class_var = class_var

        # extract feature names
        self.feature_names = dataset.columns.to_list()
        self.feature_names.remove(class_var)

        # generate class label
        self.class_label = ["Y_0", "Y_1"]

    def applyClassifer(self):

        self.errorClassifierNotFOund() if self.clf_name not in self.SUPPORTED_CLFS else ""

        if self.clf_name == "XGBoost":
            self.applyXGBoost(  )
            
    
    # https://pyagrum.readthedocs.io/en/1.2.0/notebooks/53-Classifier_CompareClassifiersWithSklearn.html
    def applyXGBoost(self, X_train, X_test, Y_train, Y_test, X_val, Y_val ):

        # genetate training files
        X_train, X_test, Y_train, Y_test, X_val, Y_val = self.generateTrainTestValSets()


        



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
