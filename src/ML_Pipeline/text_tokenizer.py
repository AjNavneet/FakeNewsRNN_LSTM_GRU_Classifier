from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import io
import json
from ML_Pipeline.constants import *

def save_tokenizer(tokenizer, num_words=vocab_size, model_dir='../output/models/', filename=None):
    """
    Save a Keras tokenizer to a JSON file.

    Parameters:
    - tokenizer: The Keras tokenizer object to be saved.
    - num_words: The maximum number of words to keep, based on word frequency.
    - model_dir: The directory where the tokenizer file will be saved.
    - filename: The filename to be used for saving the tokenizer (optional).

    Returns:
    - None

    This function saves the tokenizer to a JSON file and prints a confirmation message.
    """
    if filename is None:
        filepath = model_dir + 'tokenizer_' + str(num_words) + '.json'
    else:
        filepath = model_dir + filename
    with io.open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    f.close()
    print(f"Tokenizer saved in {filename}")

def build_tokenizer(df_train, num_words=vocab_size):
    """
    Build a Keras tokenizer from the training data.

    Parameters:
    - df_train: The training data (text data) from which the tokenizer will be built.
    - num_words: The maximum number of words to keep, based on word frequency (optional).

    Returns:
    - tokenizer: The Keras tokenizer object.
    - word_index: The word index generated from the tokenizer.

    This function creates a Keras tokenizer, fits it to the training data, and returns both the tokenizer and the word index.
    """
    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(oov_token=oov_token, num_words=num_words)
    tokenizer.fit_on_texts(df_train)
    word_index = tokenizer.word_index
    print("Word Index length", len(word_index))
    print("Number of Words:", tokenizer.num_words)
    return tokenizer, word_index

def prepare_seqence_data(df, tokenizer):
    """
    Transform text data into sequences of integers using a tokenizer.

    Parameters:
    - df: The text data that needs to be converted into sequences.
    - tokenizer: The Keras tokenizer used for conversion.

    Returns:
    - text_sequences: Sequences of integers for the text data.

    This function applies the tokenizer to the text data and returns sequences of integers for each text entry.
    """
    print("Create Sequence of tokens")
    text_sequences = tokenizer.texts_to_sequences(df)
    print("Text to sequence of Id:", text_sequences[0:1])
    return text_sequences

def pad_sequence_data(text_sequences, max_text_length):
    """
    Pad sequences of integers to ensure they have the same length.

    Parameters:
    - text_sequences: Sequences of integers to be padded.
    - max_text_length: The maximum sequence length (words) for padding.

    Returns:
    - text_padded: Padded sequences of integers.

    This function pads the sequences using the specified padding and truncation types and returns the padded sequences.
    """
    text_padded = pad_sequences(text_sequences, maxlen=max_text_length, padding=padding_type, truncating=trunction_type)
    return text_padded
