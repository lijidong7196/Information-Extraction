from tensorflow.keras.layers import Dense
from transformers import TFTransfoXLModel
import tensorflow as tf

a = tf.Variable(4)

class TENER(tf.keras.Model):
    def __init__(self, tag_vacab, embed, num_layer, d_mmodel, h_head, feedforward_dim, dropout, after_norm,
                 attn_type='adatrans', bi_embed=None, fc_dropout=0.3, pos_embed=None, scale=False, attn=None):
        super(TENER, self).__init__()
        self.embed = embed


