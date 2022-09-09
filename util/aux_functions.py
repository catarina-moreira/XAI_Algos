import numpy as np

from data.DatasetLoader import Dataset
from learning.BayesianNet import BayesianNet

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def generate_BNClassifiers(data : Dataset, learn_methods, cmap = plt.cm.RdBu, flag=True):
  BNclfs = []
  BNclfs_bn = []
  
  for method in learn_methods:
    BNclf = BayesianNet( isLoadModel = False, dataset = data)
    BNclf.applyClassifier(saveModel=False, learningMethod=method)
    BNclf.plot_decision_boundary( flag=flag, size=10, cmap=cmap, step=0.05)
    BNclfs.append( BNclf )
    BNclfs_bn.append( BNclf.clf.bn)
  return BNclfs, BNclfs_bn


def visualize_BNDecisionBoundary(BNclfs : list) -> None:
  f, axarr = plt.subplots(2,2, figsize=(10, 8))
  axs = axarr.flatten()
  for i, ax in zip(range(0,len(BNclfs)), axs):
    ax.imshow(mpimg.imread(BNclfs[i].decisionBoundary))
    ax.grid(False)
    ax.set_axis_off()
  plt.tight_layout()