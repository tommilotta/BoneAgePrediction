import tensorflow as tf
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, ReLU
from keras.initializers import glorot_uniform
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda

### CONVOLUTIONAL AND BATCHNORM HELPER FUNCTION ###
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
    if activation:
        X = Activation(activation)(X)
    return X


### STEM BLOCK ###
def stem_block(X_input):
    """
    Implementation f the stem block as defined above
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the stem block, tensor of shape (n_H, n_W, n_C)
    """
    # First conv 
    X = conv2d_bn(X_input, filters = 32, kernel_size = (3, 3), strides = (2, 2), 
                  padding = 'valid', activation='relu', name = 'stem_1th')
    
    # Second conv
    X = conv2d_bn(X, filters = 32, kernel_size = (3, 3), strides = (1, 1), 
                  padding = 'valid', activation='relu', name = 'stem_2nd')

    # Third conv
    X = conv2d_bn(X, filters = 64, kernel_size = (3, 3), strides = (1, 1), 
                  padding = 'same', activation='relu', name =  'stem_3rd')

    # First branch: max pooling
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                           padding = 'valid', name = 'stem_1stbranch_1')(X)

    # Second branch: conv
    branch2 = conv2d_bn(X, filters = 96, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'stem_1stbranch_2')

    # Concatenate (1) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)

    # First branch: 2 convs
    branch1 = conv2d_bn(X, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_1_1') 
    branch1 = conv2d_bn(branch1, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'valid', activation='relu', 
                        name = 'stem_2ndbranch_1_2') 
    
    # Second branch: 4 convs
    branch2 = conv2d_bn(X, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_1') 
    branch2 = conv2d_bn(branch2, filters = 64, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_2') 
    branch2 = conv2d_bn(branch2, filters = 64, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'stem_2ndbranch_2_3') 
    branch2 = conv2d_bn(branch2, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'valid', activation='relu', 
                        name = 'stem_2ndbranch_2_4') 

    # Concatenate (2) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)

    # First branch: conv
    branch1 = conv2d_bn(X, filters = 192, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'stem_3rdbranch_1')

    # Second branch: max pooling
    branch2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                           padding = 'valid', name = 'stem_3rdbranch_2')(X)

    # Concatenate (3) branch1 and branch2 along the channel axis
    X = tf.concat(values=[branch1, branch2], axis=3)
    
    return X

### INCEPTION-A BLOCK ###
def incres_a_block(X_input, base_name):
    """
    Implementation of the Inception-A block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 32, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 32, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 32, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_2')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 32, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 32, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 32, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_3')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch2, branch3, branch4], axis=3)

    # X = conv2d_bn(X, filters = 256, kernel_size = (1, 1), strides = (1, 1), 
    #               padding='same', activation='relu', name = base_name + 'ia_branch_4_4')
    X = conv2d_bn(X, filters = 384, kernel_size = (1, 1), strides = (1, 1), 
                  padding='same', activation='relu', name = base_name + 'ia_branch_4_4')
    
    X = tf.keras.layers.add([X, X_input])
    
    return X


### INCEPTION-B ###
def incres_b_block(X_input, base_name):
    """
    Implementation of the Inception-B block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 128, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_2_1')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 128, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 128, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 128, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_3')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch2, branch4], axis=3)

    # X = conv2d_bn(X, filters = 896, kernel_size = (1, 1), strides = (1, 1), 
    #               padding='same', activation='relu', name = base_name + 'ib_branch_4_6')
    X = conv2d_bn(X, filters = 1152, kernel_size = (1, 1), strides = (1, 1), 
                  padding='same', activation='relu', name = base_name + 'ib_branch_4_6')
    
    X = tf.keras.layers.add([X, X_input])
    
    return X


### INCEPTION-C ###
def incres_c_block(X_input, base_name):
    """
    Implementation of the Inception-C block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_2_1')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 192, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 192, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_3')

    # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
    X = tf.concat(values=[branch2, branch4], axis=3, name='concat_' + base_name)
    
    # X = conv2d_bn(X, filters = 1792, kernel_size = (1, 1), strides = (1, 1), 
    #               padding='same', activation='relu', name = base_name + 'ic_branch_4_6')
    X = conv2d_bn(X, filters = 2048, kernel_size = (1, 1), strides = (1, 1), 
                  padding='same', activation='relu', name = base_name + 'ic_branch_4_6')
    
    X = tf.keras.layers.add([X, X_input])
    
    return X


### REDUCTION-A ###
def reduction_a_block(X_input):
    """
    Implementation of the Reduction-A block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = 'valid', name = 'ra_branch_1_1')(X_input)
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 384, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'ra_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 384, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'ra_branch_3_3')

    # Concatenate branch1, branch2 and branch3 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3], axis=3)
    
    return X


### REDUCTION-B ###
def reduction_b_block(X_input):
    """
    Implementation of the Reduction-B block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), 
                           padding = 'valid', name = 'rb_branch_1_1')(X_input)
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_2_1')
    branch2 = conv2d_bn(branch2, filters = 384, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_2_2')
    
    # Branch 2
    branch4 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 256, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_4_2')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_3_4')

    # Concatenate branch1, branch2 and branch3 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    
    return X


### InceptionResNetV2 COMPOSITION ###
def InceptionResNetV2(X_input, include_top=True):
    """
    Implementation of the InceptionResNetV2 architecture

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    X -- output of the InceptionResNetV2 block
    """

    # X_input = Input(input_shape)

    # Stem block
    X = stem_block(X_input)

    # Four Inception A blocks
    X = incres_a_block(X, 'a1')
    X = incres_a_block(X, 'a2')
    X = incres_a_block(X, 'a3')
    X = incres_a_block(X, 'a4')

    # Reduction A block
    X = reduction_a_block(X)

    # Seven Inception B blocks
    X = incres_b_block(X, 'b1')
    X = incres_b_block(X, 'b2')
    X = incres_b_block(X, 'b3')
    X = incres_b_block(X, 'b4')
    X = incres_b_block(X, 'b5')
    X = incres_b_block(X, 'b6')
    X = incres_b_block(X, 'b7')

    # Reduction B block
    X = reduction_b_block(X)

    # Three Inception C blocks
    X = incres_c_block(X, 'c1')
    X = incres_c_block(X, 'c2')
    X = incres_c_block(X, 'c3')

    # Final pooling and prediction 
    if include_top:
        # AvgPool
        kernel_pooling = X.get_shape()[1:3]
        X = AveragePooling2D(kernel_pooling, name='avg_pool')(X)
        X = Flatten()(X)

        # Dropout
        X = Dropout(rate = 0.2)(X)

        # Output layer
        X = Dense(1, activation='softmax', name='fc')(X)
    
    return X