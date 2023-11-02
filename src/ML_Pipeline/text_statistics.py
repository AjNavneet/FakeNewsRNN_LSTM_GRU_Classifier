import pandas as pd

def text_statistics(df, col):
    """
    Generate text statistics for a specific column in a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame containing text data.
    col (str): The name of the column in the DataFrame to analyze.

    Returns:
    pandas.Series: A Series containing descriptive statistics of the text data in the specified column.
    """
    result = df[col].str.split().str.len()
    return result.describe()
