"""File which contains functions to calculate mutual informations between data and plot them
"""

# Import modules
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

# Functions
def getMIScores(X:pd.DataFrame,y:pd.Series,discreateFeatures:pd.Series) -> pd.Series:
    raise NotImplementedError

def plotMIBar(miScores:pd.Series,miScoresIndex:list):
    raise NotImplementedError