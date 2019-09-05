from keras.engine.topology import Layer
from keras import backend as K
from keras import losses
from keras import regularizers, initializers, activations, constraints
import tensorflow as tf
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def linspace_init(shape, dtype=None):
    minval = -2
    maxval = 2
    sep = maxval - minval
    return K.arange(minval,maxval,sep/shape,dtype=K.floatx())

def pairwise_dist_reg(weight_matrix):
    minval = -2.5
    maxval = 2.5
    a = tf.expand_dims(weight_matrix,-1)
    paddings = tf.constant([[1, 0,], [0, 0]])
    b = tf.pad(a,paddings,constant_values= minval)
    paddings = tf.constant([[0, 1,], [0, 0]])
    c = tf.pad(a,paddings,constant_values=maxval)
    d = (b-c)**2 + K.epsilon()
    e = 0.1*K.log(1/d)
    f = K.mean(K.sqrt(d+e))
    return 1e-4*f

class Quantize(Layer):

    def __init__(self, numCenters, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        self.numCenters = numCenters

    def build(self, input_shape):
        self.centers = self.add_weight(shape=(self.numCenters),
                                       initializer = linspace_init,
                                       name = 'centers',
                                       regularizer = regularizers.l2(0.1),
                                       trainable=True)
        super(Quantize, self).build(input_shape)

    def call(self, inputs):
        inputs = K.expand_dims(inputs, -1)
        dist = inputs - self.centers
        phi = K.square(K.abs(dist))
        qsoft = K.softmax(-1.0*phi, -1)
        symbols = K.argmax(K.abs(qsoft), -1)
        qsoft = K.sum(qsoft * self.centers, -1)
        one_hot_enc = K.one_hot(symbols, self.numCenters)
        symbols = K.cast(one_hot_enc,dtype=K.floatx())
        qhard = symbols * self.centers
        qhard = K.sum(qhard, axis=-1)
        qbar = qsoft + K.stop_gradient(qhard - qsoft)
        return [qbar, K.stop_gradient(qhard), K.stop_gradient(qsoft), K.stop_gradient(symbols)]

    def compute_output_shape(self, input_shape):
        return [input_shape]*4

    def get_config(self):
        config = {
            'numCenters': self.numCenters
        }
        base_config = super(Quantize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))