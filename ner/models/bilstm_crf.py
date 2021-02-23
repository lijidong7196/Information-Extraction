import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, LSTMCell, Bidirectional, Embedding

class BiLSTM(tf.keras.Model):
    def __init__(self, hidden, embed_size):
        super(BiLSTM, self).__init__()
        self.hidden = hidden
        self.embed_size = embed_size
        self.rnn = Bidirectional(LSTMCell)
