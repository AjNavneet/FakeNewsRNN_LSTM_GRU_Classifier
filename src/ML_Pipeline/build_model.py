# Function to build a Sequential network with LSTM layers
def build_network_lstm(embedding_layer, lstm_size):
    """
    Build a Sequential network with LSTM layers.

    Args:
    embedding_layer (tensorflow.python.keras.layers.Embedding): The embedding layer.
    lstm_size (int): The size of the LSTM layer.

    Returns:
    tensorflow.python.keras.models.Sequential: The LSTM model.
    """
    # Create a Sequential model
    model = Sequential()
    
    # Add the embedding layer
    model.add(embedding_layer)
    
    # Add an LSTM layer with the specified size
    model.add(LSTM(lstm_size))
    
    # Add dropout to the model
    model.add(Dropout(0.2))
    
    # Add a dense layer with ReLU activation
    model.add(Dense(32, activation='relu'))
    
    # Add dropout to the model
    model.add(Dropout(0.2))
    
    # Add an output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Function to build a Sequential network with GRU layers
def build_network_GRU(embedding_layer):
    """
    Build a Sequential network with GRU layers.

    Args:
    embedding_layer (tensorflow.python.keras.layers.Embedding): The embedding layer.

    Returns:
    tensorflow.python.keras.models.Sequential: The GRU model.
    """
    # Create a Sequential model
    model = Sequential()
    
    # Add the embedding layer
    model.add(embedding_layer)
    
    # Add a GRU layer
    model.add(GRU(100))
    
    # Add dropout to the model
    model.add(Dropout(0.3))
    
    # Add a dense layer with ReLU activation
    model.add(Dense(hidden_layer_1, activation='relu'))
    
    # Add dropout to the model
    model.add(Dropout(0.3))
    
    # Add an output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Function to build a Sequential network with SimpleRNN layers
def build_network_RNN(embedding_layer):
    """
    Build a Sequential network with SimpleRNN layers.

    Args:
    embedding_layer (tensorflow.python.keras.layers.Embedding): The embedding layer.

    Returns:
    tensorflow.python.keras.models.Sequential: The SimpleRNN model.
    """
    # Create a Sequential model
    model = Sequential()
    
    # Add the embedding layer
    model.add(embedding_layer)
    
    # Add a SimpleRNN layer
    model.add(SimpleRNN(100))
    
    # Add dropout to the model
    model.add(Dropout(0.3))
    
    # Add a dense layer with ReLU activation
    model.add(Dense(hidden_layer_1, activation='relu'))
    
    # Add dropout to the model
    model.add(Dropout(0.3))
    
    # Add an output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    
    return model
