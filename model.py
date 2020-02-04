import math
import tensorflow as tf


def load_variables_from_checkpoint(sess, start_checkpoint, var_list=None):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    if var_list is None:
        var_list = tf.global_variables()
    saver = tf.train.Saver(var_list)
    saver.restore(sess, start_checkpoint)


def get_fcn_config(nDFT, context_window_width):
    """Generate network configuration

    :param nDFT: number of DFT points
    :param context_window_width: number of input frames
    :return: a dict of network config
    """

    net_config = {
        'nDFT': nDFT,
        'context_window_width': context_window_width,
    }

    layers = dict()
    layers['conv2d_layers'] = (
        'conv1_1', 'leak1_1',
        'conv2_1', 'relu2_1',
        'conv3_1', 'relu3_1',
        'conv4_1', 'relu4_1',
        'conv5_1', 'relu5_1',
    )

    layers['conv1d_layers'] = (
        'conv1_1', 'leak1_1',
        'conv2_1', 'leak2_1',
        'conv3_1',
    )

    net_config.update({
        'use_pre_filter': False,
        'input_channel': 2,
        'pre_filter_height': 1,
        'pre_filter_width': 1,
        'pre_filter_channel': 32,
        'layers': layers,
        'filter_height': 3,
        'filter_width': 5,
        'filter_channel': 32,
        'skip_channel': 48,
        'input_height': context_window_width,
        'input_width': int(nDFT / 2) + 1,
        'dilation_time': [1, 1, 1, 1, 1],
        'dilation_freq': [1, 2, 4, 8, 16],
        'use_biases': True,
        'conv1d_channel': 64,
        'conv1d_width': 7,
    })
    net_config['dense_channel'] = net_config['filter_channel']
    net_config['conv1d_input_channel'] = net_config['skip_channel']
    net_config['conv1d_output_channel'] = 2

    if len(net_config['dilation_time']) != len(net_config['dilation_freq']):
        raise Exception('Length of time and freq dilation must be the same')

    if len(net_config['dilation_time']) != int(net_config['layers']['conv2d_layers'][-1][4]):
        raise Exception('Length of dilation is not the same as the number of layers')

    return net_config


def create_variable(name, shape):
    """ Utility function to create a convolution filter variable with the specified name and shape, and initialize it using Xavier initialition.
    :param name: name of the filter
    :param shape: shape of the filter
    :return: tensor variable with given name and shape
    """

    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_varaibles(net_config):
    """Generate network variables

    :param net_config: a dict containing network configuration
    :return: a dict containing network variables
    """

    dilation_time = net_config['dilation_time']
    dilation_freq = net_config['dilation_freq']
    layers = net_config['layers']

    if len(dilation_time) != len(dilation_freq):
        raise Exception('dilation_time must have same length as dilation_freq')

    var = dict()
    with tf.variable_scope('pre_conv'):
        cur = dict()
        cur['filter'] = create_variable('filter', [net_config['pre_filter_height'],
                                                   net_config['pre_filter_width'],
                                                   net_config['input_channel'],
                                                   net_config['pre_filter_channel']])
        if net_config['use_biases']:
            cur['bias'] = create_variable('bias', [net_config['pre_filter_channel']])
        var['pre_conv'] = cur

    var['conv2d'] = list()
    with tf.variable_scope('conv2d'):
        for i, dilation in enumerate(dilation_time):
            with tf.variable_scope('layer{}'.format(i)):
                cur = dict()
                # note that i starts with 0
                if i == 0:
                    if net_config['use_pre_filter']:
                        cur['filter'] = create_variable('filter', [net_config['filter_height'],
                                                                   net_config['filter_width'],
                                                                   # in_channel
                                                                   net_config['pre_filter_channel'],
                                                                   # out_channel, or num_filter
                                                                   net_config['filter_channel']])
                    else:
                        cur['filter'] = create_variable('filter', [net_config['filter_height'],
                                                                   net_config['filter_width'],
                                                                   # in_channel
                                                                   net_config['input_channel'],
                                                                   # out_channel, or num_filter
                                                                   net_config['filter_channel']])
                else:
                    cur['filter'] = create_variable('filter', [net_config['filter_height'],
                                                               net_config['filter_width'],
                                                               # in_channel
                                                               net_config['dense_channel'],
                                                               # out_channel, or num_filter
                                                               net_config['filter_channel']])
                cur['skip'] = create_variable('skip', [1,
                                                       1,
                                                       net_config['filter_channel'],
                                                       net_config['skip_channel']])
                cur['dense'] = create_variable('dense', [1,
                                                         1,
                                                         net_config['filter_channel'],
                                                         net_config['dense_channel']])

                if net_config['use_biases']:
                    cur['bias'] = create_variable('bias', [net_config['filter_channel']])
                    cur['skip_bias'] = create_variable('skip_bias', [net_config['skip_channel']])
                    cur['dense_bias'] = create_variable('dense_bias', [net_config['dense_channel']])

                var['conv2d'].append(cur)

    var['conv1d'] = list()
    with tf.variable_scope('conv1d'):
        layers = layers['conv1d_layers']
        how_many_conv1d_layers = int(layers[-1][4])
        for i in range(1, how_many_conv1d_layers+1):
            cur = dict()
            cur['filter'] = list()
            cur['bias'] = list()
            for j, name in enumerate(layers):
                layer_index = int(name[4])
                kind = name[:4]
                if layer_index != i:
                    continue
                with tf.name_scope('layer{}'.format(layer_index)):
                    if kind == 'conv':
                        filter_width = net_config['conv1d_width']
                        filter_height = 1
                        if layer_index == 1:
                            input_channel = net_config['conv1d_input_channel']
                            output_channel = net_config['conv1d_channel']
                        elif layer_index == how_many_conv1d_layers:
                            input_channel = net_config['conv1d_channel']
                            output_channel = net_config['conv1d_output_channel']
                        else:
                            input_channel = net_config['conv1d_channel']
                            output_channel = net_config['conv1d_channel']
                        cur_filter = create_variable('filter{}'.format(name[-1]),
                                                     [filter_height, filter_width, input_channel, output_channel])
                        cur['filter'].append(cur_filter)
                        if net_config['use_biases']:
                            cur_bias = create_variable('bias{}'.format(name[-1]),
                                                       [output_channel])
                            cur['bias'].append(cur_bias)
            var['conv1d'].append(cur)

    return var


def fcn_prefilter(input_specs, weights, net_config):
    """ Define pre-filtering operation

    :param input_specs: input spectrum
    :param weights: a dict containing network variables
    :param net_config: a dict containing net configurations
    :return: pre-filtered spectrum
    """

    if not net_config['use_pre_filter']:
        return input_specs
    weights = weights['pre_conv']
    filter = weights['filter']
    bias = weights['bias']
    with tf.name_scope('prefilter'):
        prefiltered = tf.nn.conv2d(input_specs, filter, strides=[1, 1, 1, 1], padding='SAME')
        if net_config['use_biases']:
            prefiltered = tf.nn.bias_add(prefiltered, bias)
    return prefiltered


def fcn_conv2d(input_specs, weights, net_config):
    """Define 2d convolution operations by layers

    :param input_specs: input spectrum
    :param weights: a dict containing network variables
    :param net_config: a dict containing network configurations
    :return: 4-dim output of conv2d layers
    """
    layers = net_config['layers']['conv2d_layers']
    weights = weights['conv2d']
    dilation_time = net_config['dilation_time']
    dilation_freq = net_config['dilation_freq']
    current = input_specs
    conv2d_output = 0.0
    filter_index = 0

    for i, name in enumerate(layers):
        layer_index = int(name[4])
        kind = name[:4]
        if kind == 'conv':
            with tf.name_scope('conv2d_layer_{}'.format(layer_index)):
                residual = current
                cur_weight = weights[filter_index]['filter']
                cur_bias = weights[filter_index]['bias']
                current = tf.nn.conv2d(current,
                                       cur_weight,
                                       strides=[1, 1, 1, 1],
                                       dilations=[1, dilation_time[filter_index], dilation_freq[filter_index], 1],
                                       padding="SAME")
                if net_config['use_biases']:
                    current = tf.nn.bias_add(current, cur_bias)

        elif kind == 'relu' or kind == 'leak':

            if kind == 'relu':
                with tf.name_scope('conv2d_layer_{}'.format(layer_index) + '/'):
                    current = tf.nn.relu(current)
            elif kind == 'leak':
                with tf.name_scope('conv2d_layer_{}'.format(layer_index) + '/'):
                    current = tf.nn.leaky_relu(current)

            with tf.name_scope('conv2d_layer_{}'.format(layer_index) + '/'):
                with tf.name_scope('skip'):
                    cur_skip_weight = weights[filter_index]['skip']
                    cur_skip_bias = weights[filter_index]['skip_bias']
                    skip = tf.nn.conv2d(current, cur_skip_weight, strides=[1, 1, 1, 1], padding='SAME', name='skip')
                    if net_config['use_biases']:
                        skip = tf.nn.bias_add(skip, cur_skip_bias)

                with tf.name_scope('dense'):
                    cur_dense_weight = weights[filter_index]['dense']
                    cur_dense_bias = weights[filter_index]['dense_bias']
                    current = tf.nn.conv2d(current, cur_dense_weight, strides=[1, 1, 1, 1], padding='SAME', name='dense')
                    if net_config['use_biases']:
                        current = tf.nn.bias_add(current, cur_dense_bias)

                with tf.name_scope('residual'):
                    if layer_index != 1:
                        current = current + residual
                    # current = tf.add(current, residual, name='layer{}'.format(layer_index) + 'residual_add')

            with tf.name_scope('output'):
                conv2d_output = conv2d_output + skip
            # conv2d_output = tf.add(conv2d_output, skip, name='conv2d_output_add')
            filter_index = filter_index + 1

        elif kind == 'norm':
            current = tf.layers.batch_normalization(current)

    return conv2d_output


def slice_by_frame(input_spec, net_config):
    """ Define slicing operation. Specifically, the operation extract the central frame.

    :param input_spec: input spectrum
    :param net_config: a dict containing network configurations.
    :return: sliced spectrum with size [num_batch, 1, num_frequency_bins, num_channel]
    """
    frame_index = int(math.floor(net_config['context_window_width'] / 2) + 1)
    sliced = input_spec[:, frame_index, :, :]
    # sliced = tf.reshape(sliced, [conv2d_out[0],  1, conv2d_out_shape[2], conv2d_out_shape[3]])
    sliced = tf.expand_dims(sliced, 1)
    sliced = tf.nn.relu(sliced)
    return sliced


def fcn_conv1d(conv2d_out, weights, net_config):
    """ Define 1d convolution operation by layers

    :param conv2d_out: the output from conv2d layers
    :param weights: a dict containing network variables
    :param net_config: a dict containing network configurations
    :return: estimated spectrum of clean speech
    """

    layers = net_config['layers']['conv1d_layers']
    weights = weights['conv1d']
    current = conv2d_out
    how_many_conv1d_layers = int(layers[-1][4])
    conv1d_out = []
    for i, name in enumerate(layers):
        layer_index = int(name[4])
        kind = name[:4]
        with tf.name_scope('conv1d_layer_{}'.format(layer_index) + '/'):
            if kind == 'conv':
                filter_index = int(name[-1])
                cur_weight = weights[layer_index-1]['filter'][filter_index-1]
                cur_bias = weights[layer_index-1]['bias'][filter_index-1]
                current = tf.nn.conv2d(current, cur_weight, strides=[1, 1, 1, 1], padding='SAME')
                if net_config['use_biases']:
                    current = tf.nn.bias_add(current, cur_bias)
                if layer_index == how_many_conv1d_layers:
                    conv1d_out = tf.reshape(current, [tf.shape(current)[0], net_config['input_width'], net_config['conv1d_output_channel']])
            if kind == 'relu':
                current = tf.nn.relu(current)
            if kind == 'norm':
                current = tf.layers.batch_normalization(current)
            if kind == 'leak':
                current = tf.nn.leaky_relu(current)
    return conv1d_out


def se_fcn(input_specs, nDFT, context_window_width):
    """Build the sefcn model

    :param input_specs: input spectrum of speech
    :param nDFT: number of DFT points
    :param context_window_width: number of input frames
    :return: estimated spectrum of clean speech
    """

    fcn_confg = get_fcn_config(nDFT, context_window_width)
    fcn_var = create_varaibles(fcn_confg)

    with tf.name_scope('prefilter_layers'):
        prefiltered = fcn_prefilter(input_specs, fcn_var, fcn_confg)
    with tf.name_scope('conv2d_layers'):
        conv2d_out = fcn_conv2d(prefiltered, fcn_var, fcn_confg)
    with tf.name_scope('slice'):
        conv2d_out = slice_by_frame(conv2d_out, fcn_confg)
    with tf.name_scope('conv1d_layers'):
        conv1d_out = fcn_conv1d(conv2d_out, fcn_var, fcn_confg)

    return conv1d_out

