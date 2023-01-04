import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform

def conv2d_bn(X_input, filters, kernel_size, strides, padding='same', activation=None,
              name=None):
    """
    Implementation of a conv block as defined above
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- integer, defining the number of filters in the CONV layer
    kernel_size -- (f1, f2) tuple of integers, specifying the shape of the CONV kernel
    s -- integer, specifying the stride to be used
    padding -- padding approach to be used
    name -- name for the layers
    
    Returns:
    X -- output of the conv2d_bn block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'conv_'
    bn_name_base = 'bn_'

    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
               padding = padding, name = conv_name_base + name, 
               kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = bn_name_base + name)(X)
    if activation is not None:
        X = Activation(activation)(X)
    return X

def Simple_Conv(img_shape):
    """
    Implementation of a simple multi-layer convolutional architecture

    Arguments:
    img_shape -- shape of the images of the dataset

    Returns:
    Model() instance in Keras
    """

    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))

    # Convolutional block (image data)
    X = Conv2D(32, (3,3), activation='relu')(X_input)
    # X = conv2d_bn(X_input, 32, (3,3), 1, activation='relu', name='first')
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(64, (3,3), activation='relu')(X)
    # X = conv2d_bn(X_input, 64, (3,3), 1, activation='relu', name='second')
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(128, (3,3), activation='relu')(X)
    # X = conv2d_bn(X_input, 128, (3,3), 1, activation='relu', name='third')
    X = MaxPooling2D((2, 2))(X)
    X = Flatten()(X)
    # X = Dense(128, activation='relu')(X)

    # Dense block (gender data)
    gen = Dense(32, activation='relu')(gen_input)

    # Concatenation of image and gender data
    X = tf.concat(values=[X, gen], axis=1)

    # First Dense block
    X = Dense(1000, activation='relu')(X)

    # Second Dense block
    X = Dense(1000, activation='relu')(X)

    # Fully connected layer
    X = Dense(1)(X)

    return tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name='Simple_Conv')