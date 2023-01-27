import tensorflow as tf
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, ReLU
from keras.initializers import glorot_uniform


def ResNet50(input_tensor=None, img_shape=None, classification=False, include_top=True):
    if input_tensor==None:
        input_tensor = Input(shape=img_shape)
# def ResNet50(X, classification=False, include_top=True):
    X = Conv2D(64, (7, 7), (2, 2))(input_tensor)
    X = BatchNormalization()(X)
    X = MaxPooling2D((3, 3), (2, 2))(X)
    X = conv_block(X, [64, 64, 256], 3, 1)
    X = identity_block(X, [64, 64, 256], 3)
    X = identity_block(X, [62, 62, 256], 3)
    X = conv_block(X, [128, 128, 512], 3, 2)
    X = identity_block(X, [128, 128, 512], 3)
    X = identity_block(X, [128, 128, 512], 3)
    X = identity_block(X, [128, 128, 512], 3)
    X = conv_block(X, [256, 256, 1024], 3, 2)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = conv_block(X, [512, 512, 2048], 3, 2)
    X = identity_block(X, [512, 512, 2048], 3)
    X = identity_block(X, [512, 512, 2048], 3)

    if include_top:
        X = AveragePooling2D((2, 2))(X)
        X = Flatten()(X)
        X = Dense(1)(X)

    if classification:
        X = Activation('sigmoid')(X)

    return tf.keras.Model(inputs=[input_tensor], outputs=[X])
    # return X
    

def main_path(X_input, filters, kernel_size, stride=1, init_weights=False):
    if init_weights:
        X = Conv2D(filters[0], strides=stride, kernel_size=1, padding='valid',
                   kernel_initializer = glorot_uniform(seed=0))(X_input)
    else:
        X = Conv2D(filters[0], strides=stride, kernel_size=1, padding='valid')(X_input)

    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    X = Conv2D(filters[1], strides=1, kernel_size=kernel_size, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    X = Conv2D(filters[2], strides=1, kernel_size=1, padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    return X

def identity_block(X_input, filters, kernel_size):
    X = main_path(X_input, filters, kernel_size)
    return Activation('relu')(X)

def conv_block(X_input, filters, kernel_size, stride=2, init_weights=False):
    if init_weights:
        X_main_path = main_path(X_input, filters, kernel_size, stride=stride, 
                           kernel_initializer = glorot_uniform(seed=0))
        X_shortcut_path = Conv2D(filters[2], strides=stride, kernel_size=1, padding='valid', 
                                 kernel_initializer = glorot_uniform(seed=0))(X_input)
    else:
        X_main_path = main_path(X_input, filters, kernel_size, stride=stride)
        X_shortcut_path = Conv2D(filters[2], strides=stride, kernel_size=1, padding='valid')(X_input)

    X_shortcut_path = BatchNormalization()(X_shortcut_path)
    X = tf.keras.layers.Add()([X_shortcut_path, X_main_path])
    X = Activation('relu')(X)
    return X
