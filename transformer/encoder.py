import tensorflow as tf
from tensorflow.keras.layers import  Layer, Embedding
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
import math

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

class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(shape=(input_shape[0],self.d_model),
                                       initializer=tf.keras.initializers.Zeros(),
                                       name='pos_encoding',
                                       trainable=False)

        self.position = K.expand_dims(K.arange(0,self.max_len,dtype=tf.float32),1)
        self.div_term = K.exp(K.arange(0,self.d_model, 2,dtype='float32') * (np.log(10000.0) / self.d_model))
        self.pos_encoding[:,0::2] = K.sin(self.position * self.div_term)
        self.pos_encoding[:,1::2] = K.cos(self.position * self.div_term)
        self.pos_encoding = K.transpose(K.expand_dims(self.pos_encoding,0))

    def call(self, inputs, **kwargs):
        inputs = inputs + self.pos_encoding[:inputs.shape[0], :]
        return inputs


class MutiHeadAttention(Layer):
    def __init__(self, d_model,head):
        """
        :param d_model: the demension of model
        :param head: number of head
        """
        super(MutiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = head
        self.keys = []
        self.values = []
        self.query = []

    def build(self, input_shape):

        for i in range(self.h):
            weight_K = self.add_weight(shape=(self.d_model, self.d_model / self.h),
                                     initializer='uniform',
                                     trainable=True,
                                     name='key_weight_{}'.format(i))
            self.keys.append(weight_K)
            weight_V = self.add_weight(shape=(self.d_model, self.d_model / self.h),
                                            initializer='uniform',
                                            trainable=True,
                                            name='value_weight_{}'.format(i))
            self.values.append(weight_V)
            weight_Q = self.add_weight(shape=(self.d_model, self.d_model / self.h),
                                            initializer='uniform',
                                            trainable=True,
                                            name='query_weight_{}'.format(i))
            self.query.append(weight_Q)

        self.W_o = self.add_weight(shape=(self.d_model, self.d_model),
                                   initializer='uniform',
                                   trainable=True,
                                   name='weight_o')

    def call(self, inputs, **kwargs):
        width = self.d_model / self.h
        concats = [K.dot(K.softmax(K.dot(K.dot(inputs[i*width: (i + 1)*width],Q_w), K.transpose(K.dot(inputs[i*width: (i + 1)*width],Q_w)), K.dot(inputs[i*width: (i + 1)*width],K_w))
                                   / K.sqrt(width)), K.dot(inputs[i*width: (i + 1)*width],V_w))
                   for i,(Q_w,K_w,V_w) in enumerate(zip(self.query,self.keys,self.values))]
        att = K.dot(K.concatenate(concats), self.W_o)
        return att

# Normalization layer
class LayerNormalization(Layer):
    def __init__(self, dim, eps = 1e-6):
        super(LayerNormalization, self).__init__()
        self.dim = dim
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
    def __init__(self, d_model, head, inner_dim, trainable=True):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.h = head
        self.inner_dim = inner_dim
        self.multi_head_attention = MutiHeadAttention(d_model=d_model,head=self.h)
        self.fead_forward = PositionWiseFeedForward(model_dim=d_model, inner_dim=self.inner_dim, trainable=trainable)
        self.ln1 = LayerNormalization(dim=d_model)
        self.ln2 = LayerNormalization(dim=d_model)

    def call(self, inputs, training=True, mask=None):
        x = self.multi_head_attention(inputs)
        x = self.ln1(x + inputs)

        #Point wise Feed forward
        fead_out = self.fead_forward(x)
        output = self.ln2(x + fead_out)
        return output

class Encoder(tf.keras.Model):
    def __init__(self, d_model, head, inner_dim, max_len, n_encoder, trainable=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.h = head
        self.inner_dim = inner_dim
        self.max_len = max_len
        self.trainable = trainable
        self.n_encoder = n_encoder
        self.embedding = Embedding(input_dim=max_len,output_dim=d_model)
        self.pe = PositionalEncoding(d_model=self.d_model,max_len=self.max_len)
        self.encoder_layers = [EncoderLayer(d_model=self.d_model,
                                            head=self.h,
                                            inner_dim=self.inner_dim,
                                            trainable=self.trainable)]

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.pe(x)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        return x


class










