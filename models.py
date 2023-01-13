import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import glorot_uniform
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

from inc_v4 import *
from utils import *


def create_model(model_name, img_shape, loss, optim, metric, with_gender=True):
    if model_name == 'incV3':
        model = Bone_Age_incV3(img_shape, with_gender)
        model.compile(loss=loss, optimizer=optim, metrics=[metric])

    if model_name == 'incV4':
        model = Bone_Age_incV4(img_shape, with_gender)
        model.compile(loss=loss, optimizer=optim, metrics=[metric])

    if model_name == 'simple_conv':
        model = Simple_Conv(img_shape, with_gender)
        model.compile(loss=loss, optimizer=optim, metrics=[metric])

    if model_name == 'vgg16':
        model = VGG16_boneage(img_shape, with_gender)
        model.compile(loss=loss, optimizer=optim, metrics=[metric])

    return model


# def fit_boneage(model, train_gen, train_steps, val_gen, val_steps, epochs=1, callbacks=None, path='./', gender=True):
def fit_boneage(model, train_gen, train_steps, val_gen, val_steps, epochs, callbacks, gender):
    start_time = time.time()
    model_history = model.fit(train_gen, steps_per_epoch=train_steps, validation_data=val_gen, 
                              validation_steps=val_steps, epochs=epochs, callbacks=callbacks)
    model_time = time.time() - start_time

    # save model
    # model.save(path + 'models/{}-gender={}-epochs={}.h5'.format(model.name, gender, epochs))
    model.save('./{}-gender={}-epochs={}.h5'.format(str(model.name), str(gender), str(epochs)))

    # plot and save learning curve
    plot_loss(model_history)
    # plt.savefig(path + 'plots/{}-gender={}-epochs={}_train_loss.png'.format(model.name, gender, epochs))
    plt.savefig('./{}-gender={}-epochs={}_train_loss.png'.format(str(model.name), str(gender), str(epochs)))

    print('TRAINING FINISHED')
    print('Training time: {}'.format(model_time))
    print('Loss: {}'.format(model_history.history['loss']))
    print('MAE in months: {}'.format(model_history.history['mae_in_months']))
    print('Parameters: {}'.format(model.count_params()))

    return model_history, model_time


##### FIRST PLACE SOLUTION (INCEPTIONV3) #####
def Bone_Age_incV3(img_shape, with_gender=True):
    """
    Implementation of the first place solution with Inception-v3-based architecture

    Arguments:
    img_shape -- shape of the images of the dataset
    with_gender -- if False, gender data is not used in the model

    Returns:
    Model() instance in Keras
    """

    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))
    
    # Inception block (image data)    
    inc_model = InceptionV3(input_tensor=X_input, input_shape=img_shape, include_top=False, weights=None)
    inc_model.trainable=True
    X = inc_model.get_layer('mixed10').output # 14 x 14 x 2048

    X = AveragePooling2D((2, 2))(X) # 7 x 7 x 2048
    X = Flatten()(X)

    if with_gender:
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

    return tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name='incV3_boneage')


##### INCEPTIONV4 #####
def Bone_Age_incV4(img_shape, with_gender=True): 
    """
    Implementation of an Inception-v4-based architecture for bone age prediction 

    Arguments:
    img_shape -- shape of the images of the dataset
    with_gender -- if False, gender data is not used in the model

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

    if with_gender:
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

    return tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name='incV4_boneage')


##### SIMPLE-CNN #####
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

def Simple_Conv(img_shape, with_gender=True):
    """
    Implementation of a simple multi-layer convolutional architecture

    Arguments:
    img_shape -- shape of the images of the dataset
    with_gender -- if False, gender data is not used in the model

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

    if with_gender:
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


##### VGG16 #####
def VGG16_boneage(img_shape, with_gender=True):
    """
    Implementation of a simple multi-layer convolutional architecture

    Arguments:
    img_shape -- shape of the images of the dataset
    with_gender -- if False, gender data is not used in the model

    Returns:
    Model() instance in Keras
    """

    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))

    # VGG16 model
    vgg16_model = VGG16(input_tensor=X_input, include_top=False, weights=None)
    vgg16_model.trainable = True
    vgg16_out = vgg16_model.get_layer('block5_conv3').output

    X = Flatten()(vgg16_out)

    if with_gender:
        # Dense block (gender)
        gen = Dense(32, activation='relu')(gen_input)

        # Concatenation
        # X = tf.concat(values=[X, gen], axis=3)
        X = tf.concat(values=[X, gen], axis=1)

    # First Dense block
    X = Dense(1000, activation='relu')(X)

    # Second Dense block
    X = Dense(1000, activation='relu')(X)

    # Fully connected layer
    X = Dense(1)(X)

    return tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name='vgg16_boneage')
