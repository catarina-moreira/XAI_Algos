import pandas as pd
import numpy as np
import lime
import os
import numpy as np

from data.DatasetLoader import Dataset
from Constants import Constants
from learning import Learner, BayesianNet

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from IPython.display import display, HTML, Latex

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

SMALL_SIZE = 6
MEDIUM_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rcParams['figure.dpi']=110

sns.set()







