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
    
    # First case
    # Replace every number type missing values by 0
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    
    # Second case
    # Replace every category type missing values by None
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    
    # Return the df with no missing values
    return df

def encodeCategoricalValues(df:pd.DataFrame) -> pd.DataFrame:
    """Encode categorical values in the DataFrame df

    Args : 
        df (pd.DataFrame) : treated DataFrame

    Returns :
        pd.DataFrame : df with categorical values encoded
    """

    # Encode for categorical values 
    for colName in df.select_dtypes('object'):
        df[colName],_ = df[colName].factorize()
    
    # Return df with categorical values encoded
    return df

def preprocessData(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataFrame df
    
    First step  : replace missing values
    Second step : encode categorical values

    Args : 
        df (pd.DataFrame) : treated DataFrame
    
    Returns :
        pd.DataFrame : preprocessed df
    """
    
    # Replace missing values
    df = replaceMissingValues(df)

    # Encode categorical values
    df = encodeCategoricalValues(df)

    # Return preprocessed df
    return df