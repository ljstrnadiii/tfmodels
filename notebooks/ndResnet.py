import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers


def block1(x, d, bn_axis, filters, kernel_size=3,
           stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Args:
        x: tensor
            input tensor.
        d: int
            dimension of the Conv layers to use
        bn_axis: int
            axis for batch norm
        filters: int
            filters of the bottleneck layer.
        kernel_size: int
            default 3, kernel size of the bottleneck layer.
        stride: int
            default 1, stride of the first layer.
        conv_shortcut: bool
            default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string
            block label.
    Returns:
        x: tensor
            output tensor for the residual block.
    """
    if d==1:
        conv_op = layers.Conv1D
    elif d==2:
        conv_op = layers.Conv2D
    elif d==3:
        conv_op = layers.Conv3D
    else:
        raise ValueError("d must be 1, 2, or 3")

    if conv_shortcut:
        shortcut = conv_op(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = conv_op(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
          axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = conv_op(
          filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
          axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = conv_op(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
          axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    # add the residual
    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x

def resblock(x, d, bn_axis, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    Returns:
        Output tensor for the stacked blocks.
    """
    x = block1(x, d, bn_axis, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        # no shortcut skips residual add
        x = block1(x, d, bn_axis, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def stack_blocks(x, d, bn_axis, nblocks):
    """Stacks up the residual network

    Args:
        x: tensor
            output form previous layer
        d: int
            inferred dim of conv operators for ndresnet
        bn_axis: int
            axis for the batch norm
        nblocks: list
            the list of blocks for each stack
    Returns:
        x: tensor
            the complete stack of residual blocks
    """
    x = resblock(x, d, bn_axis, 64, nblocks[0], stride1=1, name='conv2')
    x = resblock(x, d, bn_axis, 128, nblocks[1], name='conv3')
    x = resblock(x, d, bn_axis, 256, nblocks[2], name='conv4')
    x = resblock(x, d, bn_axis, 512, nblocks[3], name='conv5')

    return x

def ndResnet(input_tensor,
             bn_axis,
             n_output,
             size,
             model_name='model_name',
             use_bias=False,
             preact = True,
             include_top=True):
    """Constructs a ndResnet where the ResNet is built with a 1D or
    2D convolution operator. The 1D is useful for signals that have
    shape (batch_size, length, n_channels) and the 2d is useful for
    the regular case of 2d signals like images with shape
    (batch_size, w, h, c). The 'nd' dim is inferred given input_tensor.

    Most code comes from tensorflow's implementation here:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py

    Args:
        input_tensor: layers.Input
            The input for the model. This is used to infer the
            operators used to contstruct the relevant 'nd' model.
        bn_axis: int
            The axis to use for batch_norm. For signals (1D), specify
            the channel axis (2). For images, also specify the channel
            axis (this might depend on tensor format; its usually 3)
        n_output: int
            Number of outputs
        size: int
            one of {18, 32, 50, 101, 152} to determine with model to build
    Returns:
        model: tf.keras.Model
            The instantiated model
    """
    # infer the dimension of data; drop batch, channel
    d = len(input_tensor.shape) - 2

    if d==1:
        zeropadding = layers.ZeroPadding1D
        maxpooling = layers.MaxPool1D
        globalavgpooling = layers.GlobalAveragePooling1D
        globalmaxpooling = layers.GlobalMaxPooling1D
        convop = layers.Conv1D
        model_name += "_1DResNet"
    elif d==2:
        zeropadding = layers.ZeroPadding2D
        maxpooling = layers.MaxPool2D
        globalavgpooling = layers.GlobalAveragePooling2D
        globalmaxpooling = layers.GlobalMaxPooling2D
        convop = layers.Conv2D
        model_name += "_2DResNet"
    elif d==3:
        zeropadding = layers.ZeroPadding3D
        maxpooling = layers.MaxPool3D
        globalavgpooling = layers.GlobalAveragePooling3D
        globalmaxpooling = layers.GlobalMaxPooling3D
        convop = layers.Conv3D
        model_name += "_3DResNet"

    else:
        raise NotImplementedError("ResNet not built for "\
                                 "dims > 3")

    padding = ((3, 3), (3, 3), (3, 3))
    padding = padding[:d] if d > 1 else padding[1]

    x = zeropadding(
          padding=padding, name='conv1_pad')(input_tensor)
    x = convop(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    padding = ((1, 1), (1, 1), (1, 1))
    padding = padding[:d] if d > 1 else padding[1]

    x = zeropadding(
          padding=padding, name='conv1_pad')(input_tensor)
    x = maxpooling(3, strides=2, name='pool1_pool')(x)

    if size==18:
        blocks = [2, 1, 1, 1] #todo: verify
    elif size==32:
        blocks = [2, 2, 5, 2] #todo: verify
    elif size==50:
        blocks = [3, 4, 6, 3]
    elif size==101:
        blocks = [3, 4, 23, 3]
    elif size==152:
        blocks = [3, 8, 36, 3]
    else:
        raise NotImplementedError(
            "No Resnet for this size: {}".format(size))

    # the meat
    x = stack_blocks(x, d, bn_axis, blocks)

    if preact:
        x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = globalavgpooling(name='avg_pool')(x)
        x = layers.Dense(n_output, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = globalavgpooling(name='avg_pool')(x)
        elif pooling == 'max':
            x = globalmaxpooling(name='max_pool')(x)

    # Create the functional model.
    model = tf.keras.Model(input_tensor, x, name=model_name)

    return model
