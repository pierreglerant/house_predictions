"""File which contains functions to calculate mutual informations between data and plot them
"""

# Import modules
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

# Functions
def getMIScores(X:pd.DataFrame,y:pd.Series,discreateFeatures:pd.Series) -> pd.Series:
    """ Get the mutual information scores between y and every columns of X

    Args:
        -   X              (pd.DataFrame) : feature matrix
        -   y                 (pd.Series) : target variable
        -   discreateFeatures (pd.Series) : series with X's columns as index composed of booleans
                                           (X.columnName.dtype == int)
    
    Returns:
        pd.Series : mutual information scores between y and every columns of X
    """
    raise NotImplementedError

def plotMIBar(miScores:pd.Series,miScoresIndex:list):
    raise NotImplementedError