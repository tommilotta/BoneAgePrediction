import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import glorot_uniform


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
    if activation is not None:
        X = Activation(activation)(X)
    return X


### STEM BLOCK ###
def stem_block(X_input):
    """
    Implementation of the stem block as defined above
    
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
def inception_a_block(X_input, base_name):
    """
    Implementation of the Inception-A block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ia_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 96, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 96, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_3_2')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ia_branch_4_3')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    
    return X


### INCEPTION-B ###
def inception_b_block(X_input, base_name):
    """
    Implementation of the Inception-B block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ib_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 128, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 224, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 192, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_3')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_4')
    branch4 = conv2d_bn(branch4, filters = 256, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ib_branch_4_5')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    
    return X


### INCEPTION-C ###
def inception_c_block(X_input, base_name):
    """
    Implementation of the Inception-C block
    
    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    
    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                           padding = 'same', name = base_name + 'ic_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_1_2')
    
    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_2_1')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_1')
    branch3_1 = conv2d_bn(branch3, filters = 256, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_2')
    branch3_2 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 448, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 512, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_3')
    branch4_1 = conv2d_bn(branch4, filters = 256, kernel_size = (3, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_4')
    branch4_2 = conv2d_bn(branch4, filters = 256, kernel_size = (1, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = base_name + 'ic_branch_4_5')

    # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3_1, branch3_2, branch4_1, 
                          branch4_2], axis=3, name='concat_' + base_name)
    
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
    branch3 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 224, kernel_size = (3, 3), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'ra_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), 
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
    branch2 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_2_1')
    branch2 = conv2d_bn(branch2, filters = 192, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_2_2')
    
    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (1, 7), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 320, kernel_size = (7, 1), 
                        strides = (1, 1), padding = 'same', activation='relu', 
                        name = 'rb_branch_3_3')
    branch3 = conv2d_bn(branch3, filters = 320, kernel_size = (3, 3), 
                        strides = (2, 2), padding = 'valid', activation='relu', 
                        name = 'rb_branch_3_4')

    # Concatenate branch1, branch2 and branch3 along the channel axis
    X = tf.concat(values=[branch1, branch2, branch3], axis=3)
    
    return X


### INCEPTION V4 COMPOSITION ###
def Inceptionv4(X_input, include_top=True):
# def Inceptionv4(input_shape, include_top=True):
    """
    Implementation of the Inception-v4 architecture

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    X -- a Model() instance in Keras
    """

    # X_input = Input(input_shape)

    # Stem block
    X = stem_block(X_input)

    # Four Inception A blocks
    X = inception_a_block(X, 'a1')
    X = inception_a_block(X, 'a2')
    X = inception_a_block(X, 'a3')
    X = inception_a_block(X, 'a4')

    # Reduction A block
    X = reduction_a_block(X)

    # Seven Inception B blocks
    X = inception_b_block(X, 'b1')
    X = inception_b_block(X, 'b2')
    X = inception_b_block(X, 'b3')
    X = inception_b_block(X, 'b4')
    X = inception_b_block(X, 'b5')
    X = inception_b_block(X, 'b6')
    X = inception_b_block(X, 'b7')

    # Reduction B block
    X = reduction_b_block(X)

    # Three Inception C blocks
    X = inception_c_block(X, 'c1')
    X = inception_c_block(X, 'c2')
    X = inception_c_block(X, 'c3')

    # Final pooling and prediction 
    if include_top:
        # AvgPool
        kernel_pooling = X.get_shape()[1:3]
        X = AveragePooling2D(kernel_pooling, name='avg_pool')(X)
        X = Flatten()(X)

        # Dropout
        X = Dropout(rate = 0.2)(X)

        # Output layer
        X = Dense(1, activation='sigmoid', name='fc')(X)
    
    return X

    # # Create model
    # model = Model(inputs = X_input, outputs = X, name='Inceptionv4')

    # return model


def Bone_Age_incV4(img_shape): 
    """
    Implementation of the Inception-v4 architecture

    Arguments:
    img_shape -- shape of the images of the dataset

    Returns:
    Model() instance in Keras
    """

    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))
    
    # Inception block (image data)
    X = Inceptionv4(X_input, include_top=False) # 14 x 14 x 1536
    # to use it as a model check Github

    X = AveragePooling2D((2, 2))(X) # 7 x 7 x 1536 (or 2048 for v3)
    X = Flatten()(X)

    # Dense block (gender)
    gen = Dense(32, activation='relu')(gen_input)

    # Concatenation of image and gender data
    X = tf.concat(values=[X, gen], axis=1)

    # First Dense block
    X = Dense(1000, activation='relu')(X)

    # Second Dense block
    X = Dense(1000, activation='relu')(X)

    # Fully connected layer
    X = Dense(1)(X)

    return tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name='Bone_Age_incV4')
