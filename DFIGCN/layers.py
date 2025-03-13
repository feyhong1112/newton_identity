#!/usr/local/bin/python
from DFIGCN.inits import *
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1

        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def BILinear_pooling(adj_, XW):

    #step1 sum_squared
    sum = dot(adj_, XW, True)
    sum_squared = tf.multiply(sum, sum)

    #step2 squared_sum
    squared = tf.multiply(XW, XW)
    squared_sum = dot(adj_, squared, True)

    #step3
    new_embedding = 0.5 * (sum_squared - squared_sum)

    return new_embedding

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class Attention(Layer):
    def __init__(self, hidden_size=32):
        super(Attention, self).__init__()

        self.dense1 = tf.layers.Dense(hidden_size, activation=tf.nn.tanh)
        self.dense2 = tf.layers.Dense(1, use_bias=False)

    def _call(self, inputs):
        w = self.dense1(inputs)
        w = self.dense2(w)
        a = tf.nn.softmax(w, axis=1)
        return tf.reduce_sum(a * inputs, axis=1), a

class BGraphConvolution(Layer):
    def __init__(self,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(BGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.attention = Attention()

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_'] = glorot([input_dim, output_dim],name='weights_')
            self.vars['weights_b'] = glorot([input_dim, output_dim], name='weights_b')
            self.vars['weights_c'] = glorot([input_dim, output_dim], name='weights_c')
            self.vars['alp'] = ones(1, name='alp')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        pre_sup_a = dot(x, self.vars['weights_'], sparse=self.sparse_inputs)
        pre_sup_b = dot(x, self.vars['weights_b'], sparse=self.sparse_inputs)
        pre_sup_c = dot(x, self.vars['weights_c'], sparse=self.sparse_inputs)

        pre_sup_2 = tf.multiply(pre_sup_a, pre_sup_a)
        pre_sup_b2 = tf.multiply(pre_sup_a, pre_sup_b)
        pre_sup_sec_ord = 0.5 * (pre_sup_2 - pre_sup_b2)

        pre_sup_al = pre_sup_a + pre_sup_sec_ord

        emb = tf.stack([pre_sup_a, pre_sup_al], axis=1)
        pre_sup, att = self.attention(emb)

        out_gcn = dot(self.support[0], pre_sup, sparse=True)

        out_bi1 = BILinear_pooling(self.support[1], pre_sup) - BILinear_pooling(self.support[3], pre_sup)
        out_bi1 = dot(self.support[5], out_bi1, True)
        out_bi2 = BILinear_pooling(self.support[2], pre_sup) - BILinear_pooling(self.support[4], pre_sup)
        out_bi2 = dot(self.support[6], out_bi2, True)
        out_bi = (1 - FLAGS.beta) * out_bi1 + FLAGS.beta * out_bi2
        output = (1 - FLAGS.alpha) * out_gcn + FLAGS.alpha * out_bi

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
