import tensorflow as tf
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

WEIGHTS_INITIALIZER = tf.contrib.layers.xavier_initializer()


def get_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise ValueError("You should specify a optimizer.")
    return optimizer

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x

def exp_mask(logits, mask):
    with tf.name_scope('exp_mask'):
        return tf.add(logits, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER)

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''
    June 2017 by kyubyong park. 
    kbpark.linguist@gmail.com.
    https://www.github.com/kyubyong/transformer
    '''
    # Embeds a given tensor.

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=WEIGHTS_INITIALIZER)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs

def GLU(inputs, dim, scope='GLU'):
    with tf.name_scope(scope):
        output = tf.slice(inputs, [0, 0, 0], [-1, -1, dim]) * tf.nn.sigmoid(tf.slice(inputs, [0, 0, dim], [-1, -1, -1]))
        return output

def conv2d(inputs,
          kernel_shape=None,
          num_outputs=None,
          stride=None,
          padding="SAME",
          activation=None,
          weights_initializer=WEIGHTS_INITIALIZER,
          weights_regularizer=None,
          bias_initializer=tf.zeros_initializer(),
          bias_regularizer=None,
          causal=False,
          scope='conv2d'):
  #import ipdb; ipdb.set_trace()
  with tf.variable_scope(scope):
    inputs_shape = inputs.get_shape().as_list()
    N, seqlen, channels = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    inputs = tf.reshape(inputs, [N, seqlen, channels, 1])
    kernel_h, kernel_w = kernel_shape
    stride = [1, 1, channels, 1]
    if num_outputs is None:
      num_outputs = channels
    w = tf.get_variable('w', [kernel_h, kernel_w, 1, 2 * num_outputs], dtype=tf.float32, \
                        initializer=weights_initializer, \
                        regularizer=weights_regularizer)
    if causal:
      center_h = kernel_h // 2
      mask = np.ones((kernel_h, kernel_w, 1, 2 * num_outputs), dtype=np.int32)
      mask[center_h + 1 : , :, :, :] = 0
      mask = tf.cast(tf.constant(mask), tf.float32)
      w = w * mask

    outputs = tf.nn.conv2d(inputs, w, stride, padding)

    if bias_initializer is not None:
      b = tf.get_variable('b', [2 * num_outputs], dtype=tf.float32, \
                          initializer=bias_initializer, \
                          regularizer=bias_regularizer)
      outputs = tf.nn.bias_add(outputs, b)
    outputs = tf.squeeze(outputs)
    # mask output
    outputs_mask = tf.sign(tf.reduce_sum(tf.abs(tf.squeeze(inputs)), -1)) # 
    outputs_mask = tf.tile(tf.expand_dims(outputs_mask, -1), [1, 1, 2 * num_outputs])
    outputs = outputs * outputs_mask
    outputs = GLU(outputs, num_outputs)
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)