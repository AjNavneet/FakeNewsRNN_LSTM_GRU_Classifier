# Feature Constants
remove_columns = ['id', 'author']
column_names = ['id', 'title', 'author', 'text', 'label']
categorical_features = []
target_col = ['label']
text_features = ['title', 'text']

# Tokenization Parameters
vocab_size = 100000
max_text_length = 100
oov_token = "<OOV>"
padding_type = "post"
truncation_type = "post"

# Embedding Parameters
emb_dim = 100
glove_dir = '../../data/glove/'
glove_file_path = glove_dir + "glove.6B." + str(emb_dim) + "d.txt"
vocab_size = 150000
max_text_length = 100
embedding_type = 'glove'
model_type = 'LSTM'

# Model Parameters
lstm_size = 50
gru_size = 50
rnn_size = 50
hidden_layer_1 = 32
epochs = 20
batch_size = 256
classifier = 'binary'
