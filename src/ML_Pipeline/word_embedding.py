import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN
from tensorflow.python.keras.models import Sequential
from sklearn metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime import date
from os.path import exists
import pandas as pd
import numpy as np
import json
from ML_Pipeline.constants import *

# Define a function to train a neural network model
def train_model(model, X_train, y_train, X_test, y_test):
    # Compile the model with loss function, optimizer, and metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train the model with the training and test data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model, history

# Define a function to store the trained model
def store_model(model, file_path='../output/models/', file_name='trained_model'):
    # Serialize the model to JSON
    model_json = model.to_json()
    with open(file_path + file_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Serialize the model weights to HDF5
    model.save_weights(file_path + file_name + '.h5')
    
    print(f'Saved model to disk in path {file_path} as {file_name + ".json"}')
    print(f'Saved weights to disk in path {file_path} as {file_name + ".h5"}')

# Define a function to extract GloVe word embeddings
def extractGlovefile(glove_dir='../../data/glove/'):
    # Check if the directory exists, and if not, create it
    os.makedirs(glove_dir, exist_ok=True)
    
    # Check if the GloVe zip file exists
    file_zip = pathlib.Path(glove_dir + 'glove.6B.zip')
    
    if file_zip.exists():
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            zip.printdir()
            zip.extractall(glove_dir)
            print('Extracted GloVe word embeddings.')
    else:
        print('GloVe pretrained model not found. Downloading now...')
        
        # Download the GloVe zip file
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', out=glove_dir)
        
        # Extract the downloaded zip file
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            zip.printdir()
            zip.extractall(glove_dir)
            print('Extracted GloVe word embeddings.')

# Define a function to read GloVe word embeddings from a file
def read_glove_embedings(glove_file_path=glove_file_path):
    word_vec = pd.read_table(glove_file_path, sep=r'\s', header=None, engine='python', encoding='iso-8859-1', error_bad_lines=False)
    word_vec.set_index(0, inplace=True)
    print(f'Found {len(word_vec)} word vectors.')
    return word_vec

# Define a function to create an embedding matrix using GloVe embeddings
def glove_embedings(tokenizer):
    embeddings_index = read_glove_embedings()
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    
    # Find the indices of words that are in both the tokenizer and embeddings_index
    index_n_word = [(i, tokenizer.index_word[i]) for i in range(1, len(embedding_matrix)) if tokenizer.index_word[i] in embeddings_index.index]
    idx, word = zip(*index_n_word)
    
    # Fill the embedding matrix with GloVe word vectors
    embedding_matrix[idx, :] = embeddings_index.loc[word, :].values
    return embedding_matrix

# Define a function to create one-hot encoded embeddings
def onehot_embedding(tokenizer):
    # Convert words to one-hot encoded vectors
    onehot_vec = [one_hot(words, (len(tokenizer.word_counts) + 1)) for words in tokenizer.word_index.keys()]
    
    # Pad the one-hot encoded sequences
    embedded_docs = pad_sequences(onehot_vec, padding='pre', maxlen=max_text_length)
    return embedded_docs

# Define a function to build the embedding layer
def build_embeddings(tokenizer):
    vocab_len = vocab_size
    print("Vocabulary length:", vocab_size)
    
    if embedding_type == 'glove':
        embedding_matrix = glove_embedings(tokenizer)
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   weights=[embedding_matrix], trainable=False)
    else:
        embedding_matrix = onehot_embedding(tokenizer)
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   trainable=False)
    
    return embeddingLayer

# Define a function to plot and display training history
def performance_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Define a function to evaluate the model and return evaluation metrics
def model_evaluation(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    return score

# Define a function to calculate and report performance metrics
def performance_report(model, testX, testy, model_name, report_dir='../output/reports/'):
    time = date.today()
    
    yhat_probs = model.predict(testX, verbose=0)
    yhat_classes = model.predict_classes(testX, verbose=0)
    
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    
    accuracy = accuracy_score(testy, yhat_classes)
    print(f'Accuracy: {accuracy}')
    
    precision = precision_score(testy, yhat_classes)
    print(f'Precision: {precision}')
    
    recall = recall_score(testy, yhat_classes)
    print(f'Recall: {recall}')
    
    f1 = f1_score(testy, yhat_classes)
    print(f'F1 Score: {f1}')
    
    if exists(report_dir + 'report.csv'):
        total_cost_df = pd.read_csv(report_dir + 'report.csv', index_col=0)
    else:
        total_cost_df = pd.DataFrame(columns=['time', 'name', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
    
    total_cost_df = total_cost_df.append(
        {'time': time, 'name': model_name, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Accuracy': accuracy},
        ignore_index=True)
    
    total_cost_df.to_csv(report_dir + 'report.csv')
    return total_cost_df
