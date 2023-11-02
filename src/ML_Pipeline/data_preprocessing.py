##########################
###   Text DATA PREP   ###
##########################

# Import necessary libraries
from collections import Counter
from tensorflow import keras
from keras import preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize a Porter Stemmer and define stop words
ps = PorterStemmer()
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# Function to process labels and encode them
def process_labels(labels):
    """
    Process labels and encode them using LabelEncoder if they are of type 'object'.

    Args:
    labels (pandas.Series): The labels to be encoded.

    Returns:
    pandas.Series: Encoded labels.
    """
    if labels.dtype == 'object':
        lbl_enc = preprocessing.LabelEncoder()
        labels = lbl_enc.fit_transform(labels)
    return labels

# Function to convert categorical features to one-hot encoding
def convert_categorical_features(df, cat_columns):
    """
    Convert categorical features to one-hot encoding.

    Args:
    df (pandas.DataFrame): The DataFrame containing categorical features.
    cat_columns (list): List of column names to be converted.

    Returns:
    None
    """
    for cat in cat_columns:
        df[cat] = keras.utils.to_categorical(df[cat], num_classes=2)

# Function to clean text by removing unused characters and converting to lowercase
def clean_text(text):
    """
    Clean text by removing unwanted characters and converting to lowercase.

    Args:
    text (str): The text to be cleaned.

    Returns:
    str: Cleaned text.
    """
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # Removing URLs
    text = str(text).replace(r'[^\.\w\s]', ' ')  # Remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    return text

# Function to perform NLTK preprocessing, including stop word removal
def nltk_preprocessing(text):
    """
    Perform NLTK preprocessing, including stop word removal.

    Args:
    text (str): The text to be preprocessed.

    Returns:
    str: Processed text.
    """
    text = ' '.join([word for word in text.split() if word not in stopwords_dict])
    return text

# Function to merge text features together and create a new column
def merge_text_features(df, text_features, col_name='news'):
    """
    Merge text features together and create a new column in the DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame containing text features.
    text_features (list): List of text feature column names.
    col_name (str): Name of the new column to store merged text.

    Returns:
    pandas.DataFrame: The DataFrame with the new merged text column.
    """
    print("Features in dataset:", df.columns)
    df[col_name] = df[text_features].agg(' '.join, axis=1)
    print("Merge news text statistics:\n", df[col_name].str.split().str.len().describe())
    return df

# Function to prepare datasets for modeling
def preparing_datasets(df, text_features):
    """
    Prepare datasets for modeling by merging text features and performing text preprocessing.

    Args:
    df (pandas.DataFrame): The DataFrame containing the dataset.
    text_features (list): List of text feature column names.

    Returns:
    X (pandas.Series): The preprocessed text data.
    y (pandas.Series): The labels.
    """
    XY = df.copy()
    XY = merge_text_features(XY, text_features)
    XY["news"] = XY.news.apply(clean_text)
    print("Cleaning as remove special characters is done..")
    XY["news"] = XY.news.apply(nltk_preprocessing)
    X = XY['news']
    y = XY.label
    print("Text length statistics after merging news and preprocessing:\n", X.str.split().str.len().describe())
    if y.dtype == 'object':
        y = process_labels(y)
    return X, y
