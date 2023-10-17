import numpy as np
import tensorflow as tf
import simple_helperfns

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

def encoder_apply_one_shift(prev_layer, weights, biases, act_type, name='E', num_encoder_weights=1, use_bias=True):
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
        
        if use_bias:
            prev_layer = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]
        else:
            prev_layer = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 1)])

        if act_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif act_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif act_type == 'elu':
            prev_layer = tf.nn.elu(prev_layer)

    # apply last layer without any nonlinearity
    if use_bias:
        final = tf.matmul(prev_layer, weights['WE%d' % (num_encoder_weights)]) + biases[
            'bE%d' % (num_encoder_weights)]
    else:
        final = tf.matmul(prev_layer, weights['WE%d' % (num_encoder_weights)])

    return final

def encoder_apply(x, weights, biases, act_type, shifts_middle, name='E', num_encoder_weights=1, use_bias=True):
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
            encoder_apply_one_shift(x_shift, weights, biases, act_type, name, num_encoder_weights, use_bias))
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

def decoder_apply(prev_layer, weights, biases, act_type, num_decoder_weights, use_bias=True):
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
        
        if use_bias:
            prev_layer = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
        else:
            prev_layer = tf.matmul(prev_layer, weights['WD%d' % (i + 1)])
        
        if act_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif act_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif act_type == 'elu':
            prev_layer = tf.nn.elu(prev_layer)
        elif act_type == 'linear':
            prev_layer = prev_layer

    # apply last layer without any nonlinearity
    if use_bias:
        output = tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]
    else:
        output = tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights])
    
    return output

def k_block(widths, name='K'):
    """Create a K block network: a dictionary of weights (linear, no biases).

    Arguments:
        widths -- array or list of widths for layers of network
        name -- string for prefix on weight matrices (default 'K' for K block)

    Returns:
        weights -- dictionary of weights
    """
    weights, biases = decoder(widths, name)
    return weights

def k_block_apply_one_shift(y, weights, name='K'):
    """Apply an K block to data for only one time step (shift).

    Arguments:
        y -- input for a particular time step (shift)
        weights -- dictionary of weights
        name -- string for prefix on weight matrices (default 'K' for K block)

    Returns:
        array same size as input y, but advanced to next time step
    """
    num_k_weights = 0
    for key, val in weights.items():
        if key[:2] == 'WK':
            num_k_weights += 1
    # num_k_weights = len(weights)
    for i in np.arange(num_k_weights):
        y = tf.matmul(y, weights['WK%d' % (i + 1)])
        
    return y

def k_block_apply_multiple_shifts(y, weights, shift):
    """Apply an K block to data for delta t time steps (shifts).

    Arguments:
        y -- input for a particular time step (shift)
        weights -- dictionary of weights
        name -- string for prefix on weight matrices (default 'K' for K block)

    Returns:
        array same size as input y, but advanced to next time step
    """
    for i in range(shift):
        y = k_block_apply_one_shift(y, weights)

    return y

def create_koopman_net(params):
    """Create a Koopman network that encodes, advances in time, and decodes.

    Arguments:
        params -- dictionary of parameters for experiment, includes
            encoder_widths -- widths of encoder layers, list
            max_shifts_to_stack -- number of shifts in data stack for encoder loss, int
            encoder_act_type -- activation function for encoder, str
            shifts_middle -- shifts to evaluate linearity loss, list
            decoder_widths -- widths of decoder layers, list
            decoder_act_type -- activation function for decoder, str
            k_widths -- widths of K block layers, list
            shifts -- shifts to evaluate prediction loss, list

    Returns:
        x -- placeholder for input
        y -- list, output of decoder applied to each shift: g_list[0], K*g_list[0], K^2*g_list[0], ..., length num_shifts + 1
        g_list -- list, output of encoder applied to each shift in input x, length num_shifts_middle + 1
        weights -- dictionary of weights
        biases -- dictionary of biases
    """
    weights = dict()
    biases = dict()

    max_shifts_to_stack = simple_helperfns.num_shifts_in_stack(params)

    x, weights_encoder, biases_encoder = encoder(params['encoder_widths'], num_shifts_max=max_shifts_to_stack)
    weights.update(weights_encoder)
    biases.update(biases_encoder)
    num_encoder_weights = len(weights_encoder)
    g_list = encoder_apply(x, weights, biases, params['encoder_act_type'], params['shifts_middle'], 
                           num_encoder_weights=num_encoder_weights, use_bias=params['use_bias'])
    
    weights_k = k_block(params['k_widths'])
    weights.update(weights_k)

    weights_decoder, biases_decoder = decoder(params['decoder_widths'])
    weights.update(weights_decoder)
    biases.update(biases_decoder)

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]
    num_decoder_weights = len(weights_decoder)
    y.append(decoder_apply(encoded_layer, weights, biases, params['decoder_act_type'], 
                           num_decoder_weights=num_decoder_weights, use_bias=params['use_bias']))

    # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
    advanced_layer = k_block_apply_one_shift(encoded_layer, weights)

    for j in np.arange(max(params['shifts'])):
        # considering penalty on subset of yk+1, yk+2, yk+3, ...
        if (j + 1) in params['shifts']:
            y.append(decoder_apply(advanced_layer, weights, biases, params['decoder_act_type'], 
                                   num_decoder_weights=num_decoder_weights, use_bias=params['use_bias']))
        advanced_layer = k_block_apply_one_shift(advanced_layer, weights)

    if len(y) != (len(params['shifts']) + 1):
        print("messed up looping over shifts! %r" % params['shifts'])
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, g_list, weights, biases
