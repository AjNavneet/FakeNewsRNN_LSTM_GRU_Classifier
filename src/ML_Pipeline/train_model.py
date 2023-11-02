import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime from date
from os.path from exists
from pandas as pd
from numpy as np
from ML_Pipeline.constants import *

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train a machine learning model.

    Parameters:
    - model: The machine learning model to be trained.
    - X_train: The training data.
    - y_train: The labels for the training data.
    - X_test: The test data.
    - y_test: The labels for the test data.

    Returns:
    - model: The trained machine learning model.
    - history: Training history.

    This function compiles the model with loss, optimizer, and metrics, then trains it with the specified data.
    """
    # Compile the model with a loss function, optimizer, and metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train the model with the training data
    # You can specify the number of epochs and batch size here
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model, history

def store_model(model, file_path='../output/models/', file_name='trained_model'):
    """
    Store the trained model to disk.

    Parameters:
    - model: The trained machine learning model to be stored.
    - file_path: The directory where the model and weights will be saved.
    - file_name: The base name for the model and weight files.

    Returns:
    - None

    This function serializes the model to JSON and saves the weights as an HDF5 file. It also prints confirmation messages.
    """
    # Serialize the model to JSON
    model_json = model.to_json()
    with open(file_path + file_name + '.json', "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights(file_path + file_name + '.h5')
    print(f"Saved model to disk in path {file_path} as {file_name + '.json'}")
    print(f"Saved weights to disk in path {file_path} as {file_name + '.h5'}")
