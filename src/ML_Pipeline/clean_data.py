# Function to remove unused columns from a DataFrame
def remove_unused_columns(df, column_names):
    """
    Remove unused columns from a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to be cleaned.
    column_names (list): List of column names to be removed.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with specified columns removed.
    """
    for col in column_names:
        if col in df.columns:
            df = df.drop(column_names, axis=1)
    return df

# Function to impute null values in a DataFrame with "None"
def null_processing(feature_df):
    """
    Impute null values in a DataFrame with "None".

    Args:
    feature_df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    pandas.DataFrame: The DataFrame with null values replaced by "None".
    """
    print("Number of records with null values:", feature_df.isnull().sum())
    columns = (feature_df.columns[feature_df.isnull().sum() > 0])
    print("Columns having null values: ", columns)
    feature_df.dropna(axis=0, inplace=True)
    return feature_df

# Function to clean a DataFrame by removing unused columns and imputing null values with "None"
def clean_data(df, remove_column_names):
    """
    Clean a DataFrame by removing unused columns and imputing null values with "None".

    Args:
    df (pandas.DataFrame): The DataFrame to be cleaned.
    remove_column_names (list): List of column names to be removed.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # Remove unused columns
    df = remove_unused_columns(df, remove_column_names)
    
    # Impute null values with "None"
    feature_df = null_processing(df)
    
    return feature_df
