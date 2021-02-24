import tensorflow as tf
from tensorflow.keras.layers import  Layer

class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        super(PositionWiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.trainable = trainable

    def build(self, input_shape):
        self.weights_inner = self.add_weight(shape=(input_shape[-1], self.inner_dim),
                                             initializer="glorot_uniform",
                                             trainable=self.trainable,
                                             name="weights_inner")

        self.weights_out = self.add_weight(shape=(self.inner_dim,self.model_dim),
                                           initializer="glorot_uniform",
                                           trainable=self.trainable,
                                           name="weights_out")
        self.bias_inner = self.add_weight(shape=(self.inner_dim,),
                                          initializer="uniform",
                                          trainable=self.trainable,
                                          name="bias_inner")
        self.bias_out = self.add_weight(shape=(self.model_dim,),
                                        initializer="glorot_uniform",
                                        trainable=self.trainable,
                                        name="bias_out")

    def call(self, inputs, **kwargs):
        inner_out = tf.nn.relu(tf.matmul(inputs,self.weights_inner) + self.bias_inner)
        output = tf.matmul(inner_out, self.weights_out) + self.bias_out
        return output

class MutiHeadAttention(Layer):
    def __init__(self):
        super(MutiHeadAttention, self).__init__()

# Normalization layer
class LayerNormalization(Layer):
    def __init__(self, dim, eps = 1e-6):
        super(LayerNormalization, self).__init__()
        self.dim = self.dim
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(self.dim,self.dim),
                                     initializer=tf.keras.initializers.Ones(),
                                     name='gamma',
                                     trainable=True)
        self.beta = self.add_weight(shape=(self.dim,self.dim),
                                    initializer=tf.keras.initializers.Zeros(),
                                    name='beta',
                                    trainable=True)

    def call(self, inputs, **kwargs):
        #mean = tf.reduce_mean(inputs,keepdims=True,axis=-1)
        mean, var = tf.nn.moments(inputs,axes=-1,keepdims=True)
        std = tf.sqrt(var)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta
    

class EncoderLayer(tf.keras.Model):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        pass

class PositionalEncoding(Layer):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(shape=(input_shape[0],self.d_model),
                                       initializer=tf.keras.initializers.Zeros(),
                                       name='pos_encoding',
                                       trainable=False)


    def call(self, inputs, **kwargs):
        pass









