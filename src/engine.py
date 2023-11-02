# Import necessary modules and functions
from ML_Pipeline.utils import read_data
from ML_Pipeline.text_statistics import text_statistics
from ML_Pipeline.clean_data import clean_data
from ML_Pipeline.data_preprocessing import preparing_datasets
from ML_Pipeline.text_tokenizer import save_tokenizer, build_tokenizer, prepare_seqence_data, pad_sequence_data
from ML_Pipeline.constants import *
from ML_Pipeline.word_embedding import build_embeddings
from ML_Pipeline.build_model import build_network_GRU, build_network_RNN, build_network_lstm
from ML_Pipeline.train_model import train_model, store_model
from ML_Pipeline.evaluate_model import performance_history, model_evaluation, performance_report

# Read data from CSV files
news_df = read_data("../../data/input/train.csv")
test = read_data('../../data/input/test.csv')
submit_test = read_data('../../data/input/submit.csv')
test['label'] = submit_test.label

# Compute text statistics for 'text' and 'title' columns
text_stats = text_statistics(news_df, 'text')
title_stats = text_statistics(news_df, 'title')

# Create a copy of the dataset for cleaning
df = news_df.copy()

# Data cleaning
df = clean_data(df, remove_column_names=remove_columns)
df_test = clean_data(test, remove_column_names=remove_columns)

# Preprocess datasets for training and testing
X, y = preparing_datasets(df, text_features=text_features)
X_test, y_test = preparing_datasets(df_test, text_features=text_features)
X_train, y_train = X, y

# Tokenization and text sequence preparation
tokenizer, word_index = build_tokenizer(X_train)
save_tokenizer(tokenizer)
train_text_seq = prepare_seqence_data(X_train, tokenizer)
test_text_seq = prepare_seqence_data(X_test, tokenizer)
train_text_padded = pad_sequence_data(train_text_seq, max_text_length)
test_text_padded = pad_sequence_data(test_text_seq, max_text_length)

# Print tokenization details and data shapes
print("Padded Sequence: ", test_text_padded[0:1])
print("Tokenizer detail: ", tokenizer.document_count)
print('Vocabulary size:', len(tokenizer.word_counts))
print('Shape of data padded:', train_text_padded.shape)

# Build embedding layer
embeding_layer = build_embeddings(tokenizer)

# Model building and training (e.g., RNN model)
model_rnn = build_network_RNN(embeding_layer)
model_rnn, history = train_model(model_rnn, train_text_padded, y_train, test_text_padded, y_test)

# Plot and save the performance history of the model
performance_history(history)

# Store the model and evaluate its performance
store_model(model_rnn, file_name='rnn_1')
score = model_evaluation(model_rnn, test_text_padded, y_test)

# Generate and store a performance report
total_cost_df = performance_report(model_rnn, test_text_padded, y_test, 'rnn_1')

# (Additional code for LSTM and GRU models is commented out)
