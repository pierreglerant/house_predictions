"""File which contains functions to calculate mutual informations between data and plot them
"""

# Import modules
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np

# Functions
def getMIScores(X:pd.DataFrame,y:pd.Series,discreateFeatures:pd.Series) -> pd.Series:
    """ Get the mutual information scores between y and every columns of X

    Args :
        -   X              (pd.DataFrame) : feature matrix
        -   y                 (pd.Series) : target variable
        -   discreateFeatures (pd.Series) : series with X's columns as index composed of booleans
                                           (X.columnName.dtype == int)
    
    Returns:
        pd.Series : mutual information scores between y and every columns of X
    """
    
    # Get a list of mutual information scores between y and every columns of y 
    miScores = mutual_info_regression(X,y,discrete_features=discreateFeatures)

    # Transform the list into a pd serie
    miScores = pd.Series(miScores,name= 'MI Scores',index=X.columns)

    # Return miScores sorted (desc)
    return miScores.sort_values(ascending=False)

def plotMIBar(miScores:pd.Series,miScoresIndex:list=None):
    """ Make a bar plot with mutual information scores

    Args :
        -   miScores (pd.Series) : mutual information scores sorted (desc) between x and every columns of X
        -   miScoresIndex (list) : index of ploted mutual information (optional)
    """

    # Select the MI scores indicated by miScoresIndex (if the list has been defined)
    if miScoresIndex != None:
        miScores = miScores[miScoresIndex]

    # Define a figure
    plt.figure()

    # Create an array of values from 0 to the length of miScores - 1
    # This will serve as the positions for the bars on the y-axis
    width = np.arange(len(miScores))

    # Convert the index of the miScores DataFrame/Series into a list
    # These will be used as labels on the y-axis
    ticks = list(miScores.index)

    # Create a horizontal bar chart with 'width' as the y positions and 'miScores' as the bar lengths
    plt.barh(width, miScores)

    # Set the y-axis ticks to the positions in 'width' and label them with 'ticks'
    plt.yticks(width, ticks)

    # Set the title of the plot to "Mutual Information Scores"
    plt.title("Mutual Information Scores")

    # Show the plot
    plt.show()