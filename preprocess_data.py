"""File which contains preprocessing functions
"""

# Import modules
import pandas as pd

# Functions
def replaceMissingValues(df:pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the df
    
    First case     (columns composed of numbers) : missing values are replaced by 0
    Second case (columns composed of categories) : missing values are replaced by None

    Args : 
        df (pd.DataFrame) : treated DataFrame

    Returns :
        pd.DataFrame : df with missing values replaced by 0 or None
    """
    raise NotImplementedError