import numpy as np
import tensorflow as tf


def weight_variable(shape, var_name):
    """Create a variable for a weight matrix.

    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable

    Returns:
        a TensorFlow variable for a weight matrix
    """
    initial = tf.random.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)

def bias_variable(shape, var_name):
    """Create a variable for a bias vector.

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable

    Returns:
        a TensorFlow variable for a bias vector
    """
    initial = tf.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)

def encoder(widths, num_shifts_max):
    """Create an encoder network: an input placeholder x, dictionary of weights, and dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        num_shifts_max -- number of shifts (time steps) that losses will use (max of num_shifts and num_shifts_middle)

    Returns:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases
    """
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float64, [num_shifts_max + 1, None, widths[0]])

    weights = dict()
    biases = dict()

    for i in np.arange(len(widths) - 1):
        weights['WE%d' % (i + 1)] = weight_variable([widths[i], widths[i + 1]], var_name='WE%d' % (i + 1))
        biases['bE%d' % (i + 1)] = bias_variable([widths[i + 1], ], var_name='bE%d' % (i + 1))
    
    return x, weights, biases

def encoder_apply_one_shift(prev_layer, weights, biases, act_type, name='E', num_encoder_weights=1):
    """Apply an encoder to data for only one time step (shift).

    Arguments:
        prev_layer -- input for a particular time step (shift)
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        name -- string for prefix on weight matrices (default 'E' for encoder)
        num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)

    Returns:
        final -- output of encoder network applied to input prev_layer (a particular time step / shift)
    """
    for i in np.arange(num_encoder_weights - 1):
        prev_layer = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]
        if act_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif act_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif act_type == 'elu':
            prev_layer = tf.nn.elu(prev_layer)

    # apply last layer without any nonlinearity
    final = tf.matmul(prev_layer, weights['W%s%d' % (name, num_encoder_weights)]) + biases[
        'b%s%d' % (name, num_encoder_weights)]

    return final

def encoder_apply(x, weights, biases, act_type, shifts_middle, name='E', num_encoder_weights=1):
    """Apply an encoder to data x.

    Arguments:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss
        name -- string for prefix on weight matrices (default 'E' for encoder)
        num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)
    Returns:
        y -- list, output of encoder network applied to each time shift in input x

    """
    y = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_middle + 1):
        if j == 0:
            shift = 0
        else:
            shift = shifts_middle[j - 1]
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        y.append(
            encoder_apply_one_shift(x_shift, weights, biases, act_type, name, num_encoder_weights))
    return y

def decoder(widths, name='D'):
    """Create a decoder network: a dictionary of weights and a dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        name -- string for prefix on weight matrices (default 'D' for decoder)

    Returns:
        weights -- dictionary of weights
        biases -- dictionary of biases
    """
    weights = dict()
    biases = dict()
    for i in np.arange(len(widths) - 1):
        ind = i + 1
        weights['W%s%d' % (name, ind)] = weight_variable([widths[i], widths[i + 1]], var_name='W%s%d' % (name, ind))
        biases['b%s%d' % (name, ind)] = bias_variable([widths[i + 1], ], var_name='b%s%d' % (name, ind))
    return weights, biases

def decoder_apply(prev_layer, weights, biases, act_type, num_decoder_weights):
    """Apply a decoder to data prev_layer

    Arguments:
        prev_layer -- input to decoder network
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        num_decoder_weights -- number of weight matrices (layers) in decoder network

    Returns:
        output of decoder network applied to input prev_layer
    """
    for i in np.arange(num_decoder_weights - 1):
        prev_layer = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
        if act_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif act_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif act_type == 'elu':
            prev_layer = tf.nn.elu(prev_layer)
        elif act_type == 'linear':
            prev_layer = prev_layer

    # apply last layer without any nonlinearity
    return tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]








