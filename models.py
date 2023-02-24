import time
import tensorflow as tf
import matplotlib as mpl
from sklearn.metrics import r2_score
from keras.layers import Input, Dense, Flatten, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications.vgg16 import VGG16

from inc_v4 import *
from utils import *

# define constants
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
BATCH_SIZE = 32
MODELS = ['incV3', 'incV4', 'vgg16', 'incRes', 'incV3 nogender', 'incV3 Male', 'incV3 Female']
LOSS = 'mean_absolute_error'

cmap = mpl.cm.get_cmap('viridis')
COLORS = [cmap(x) for x in np.linspace(0, 1, num=len(MODELS), endpoint=False)]


def create_model(model_name, img_shape, loss, optim, metric, with_gender=True):
    """
    Implementation of an architecture for bone age prediction with different backbone models
    
    Arguments:
    model_name -- name of the chosen model
    img_shape -- shape of the images of the dataset
    loss -- loss with which to compile the model
    optim -- optimizer with which to compile the model
    metric -- metric with which to compile the model
    with_gender -- if False, gender data is not used in the model
    
    Returns:
    compiled model
    """

    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))

    # choose backbone
    if model_name == 'incV3':
        # Inception block (image data)    
        inc_model = InceptionV3(input_tensor=X_input, input_shape=img_shape, include_top=False, weights=None)
        inc_model.trainable=True
        X = inc_model.get_layer('mixed10').output
        X = AveragePooling2D((2, 2))(X)
        
    if model_name == 'incV4':
        # Inception block (image data)
        X = Inceptionv4(X_input, include_top=False)
        X = AveragePooling2D((2, 2))(X)

    if model_name == 'vgg16':
        # VGG16 block (image data)
        vgg16_model = VGG16(input_tensor=X_input, include_top=False, weights=None)
        vgg16_model.trainable = True
        X = vgg16_model.get_layer('block5_conv3').output

    if model_name == 'incRes':
        # Inception block (image data)
        incres = InceptionResNetV2(input_tensor=X_input, include_top=False, weights=None)
        X = incres.layers[-1].get_output_at(0)
        X = AveragePooling2D((2, 2))(X)
        
    # common structure
    X = Flatten()(X)

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

    model = tf.keras.Model(inputs=[X_input, gen_input], outputs=X, name=model_name)
    model.compile(loss=loss, optimizer=optim, metrics=[metric])

    return model


def fit_boneage(model, train_gen, train_steps, val_gen, val_steps, epochs, callbacks):
    """
    Train the model with given training and validation data.
    
    Arguments:
    model -- chosen model
    train_gen -- training generator to fit
    train_steps -- training steps between epochs
    train_gen -- validation generator to fit
    train_steps -- validation steps between epochs
    epochs -- number of training epochs
    callbacks -- list of callback functions called during training
    """

    start_time = time.time()
    model_history = model.fit(train_gen, steps_per_epoch=train_steps, validation_data=val_gen, 
                              validation_steps=val_steps, epochs=epochs, callbacks=callbacks)
    model_time = time.time() - start_time

    print('TRAINING FINISHED')
    print('Training time: {}'.format(model_time))
    print('Loss: {}'.format(model_history.history['loss']))
    print('MAE in months: {}'.format(model_history.history['mae_in_months']))
    print('Parameters: {}'.format(model.count_params()))


def evaluate_and_predict(model_name, t_gen, weight_path, logs_path, std_boneage, mean_boneage, gender=True,
                          plot=False, metric='mean_absolute_error', optim='adam'):
    """
    Evaluate the model with test data and get the predictions.
    
    Arguments:
    model_name -- chosen model
    t_gen -- test generator to evaluate
    weight_path -- full path where the weights of the model are saved
    logs_path -- full path where data about the training of the model is saved
    std_boneage -- standard deviation of the bone age values used to get the predictions in months
    mean_boneage -- mean of the bone age values used to get the predictions in months
    gender -- if True uses the full model structure with the gender input
    plot -- if True plot the regression
    metric -- metric with which to compile the model
    optim -- optimizer with which to compile the model

    Returns:
    model -- created and compiled model
    logs -- data relative to the training of the model
    params -- number of parameters of the model
    test_eval -- (test_loss, test_mae) data relative to the evaluation
    r2 -- R**2 score on the test set
    prediction -- predictions of the model on test data
    test_gtruth_months -- bone age ground truth of test data
    """
    # create model and load weights
    model = create_model(model_name, IMG_SHAPE, LOSS, optim, metric, with_gender=gender)
    model.load_weights(weight_path)

    test_data, test_gtruth = next(t_gen)
    # predict model on test data
    prediction = std_boneage * model.predict(test_data, batch_size=BATCH_SIZE, verbose=True) + mean_boneage
    test_gtruth_months = std_boneage * test_gtruth + mean_boneage
    # evaluate model
    test_eval = model.evaluate(t_gen, verbose=False, steps=1)

    if plot:
        miss = np.abs(prediction.T - test_gtruth_months)
        # print regression
        print('EVALUATION FINISHED\n')
        print('Loss: {}'.format(test_eval[0]))
        print('Mean Absolute Error (months): {}'.format(test_eval[1]))
        print('Max Error (months): {}'.format(np.max(miss)))
        print('Median Error (months): {}'.format(np.median(miss)))

        # plot regression
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(test_gtruth_months, prediction, '.', c=COLORS[0], label='predictions')
        ax.plot(test_gtruth_months, test_gtruth_months, '-', c=COLORS[-1], label='ground truth')
        ax.legend()
        ax.set_xlabel('Actual Age (months)')
        ax.set_ylabel('Predicted Age (months)')
        plt.show()

    # load training data
    logs = pd.read_csv(logs_path)
    # model params
    params = model.count_params()
    # R**2 score
    r2 = r2_score(test_gtruth_months, prediction)

    return model, logs, params, test_eval, r2, prediction, test_gtruth_months

def create_weight_ensamble(models, img_shape, loss='mean_absolute_error', optim=tf.keras.optimizers.Adam(10**-4),
                            metric='mean_absolute_error'):
    X_input = Input(shape=img_shape)
    gen_input = Input(shape=(1,))

    num_models = len(models)

    models_results = [model([X_input, gen_input], training=False) for model in models]
    models_results = tf.concat(values=[m for m in models_results], axis=1)

    # Weighting CNN
    X = Conv2D(2, (3,3), activation='relu')(X_input)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(4, (3,3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(8, (3,3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(16, (3,3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Flatten()(X)

    # Gender input
    gen = Dense(32, activation='relu')(gen_input)

    # Models result dense layer
    Y = Dense(300, activation='relu')(models_results)

    # Concatenation of image and gender data
    X = tf.concat(values=[X, Y, gen], axis=1)

    # Dense net
    X = Dense(100, activation='relu')(X)
    Weights = Dense(num_models, activation='softmax')(X)

    out = tf.reduce_sum(Weights*models_results, 1, keepdims=True)   

    model = tf.keras.Model(inputs=[X_input, gen_input], outputs=out, name='ensamble')
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=[metric]) 

    return model

def visualize_preds(df, predictions, gtruths, indexes=None):
    """
    Visualize images with the repective ground truth and prediction 
    
    Arguments:
    df -- dataframe containing the path for the images
    predictions -- bone age predictions relative to each image
    gtruths -- bone age ground truths relative to each image
    indexes -- indexes of the images to plot
    """
    _, axs = plt.subplots(2, 2, figsize=(8, 9))
    [ax.axis('off') for ax in axs.flatten()]

    if not indexes:
      indexes = np.random.randint(0, df.shape[0], size=4)
    for i, ax in enumerate(axs.flatten()):
      ind = indexes[i]
      name = df.loc[ind, 'image'].split('/')[-1].split('.')[0]
      ax.imshow(cv2.imread(df.loc[ind, 'image']))
      gtruth = int(gtruths[ind]*100)/100
      prediction = int(predictions[ind]*100)/100
      ax.set_title('Image: {}\nground truth: {}\nprediction: {}'.format(name, gtruth, prediction), size=12)


def plot_bars(values, colors, title, xlabel, ylabel, values_2=None, values_3=None, annotate=True):
    """
    Bar plot to visualize training and testing results 
    
    Arguments:
    values -- values to plot on y-axes
    colors -- set of colors to use for the bars
    title -- title of the plot
    xlabel -- label of the x-axis
    ylabel -- label of the y-axis
    values_2 -- additional value to plot in adiacent bar
    values_3 -- second addition value to plot in adiacent bar
    annotate -- if True write values in the plot
    """
    x_pos = np.arange(len(MODELS))
    
    if values_3:
        fig, ax = plt.subplots(figsize=(10, 5))
        # plot data in grouped manner of bar type
        plt.bar(x_pos-0.3, values, 0.2, color=colors[0])
        plt.bar(x_pos, values_2, 0.2, color=colors[int(len(colors)/2)])
        plt.bar(x_pos+0.3, values_3, 0.2, color=colors[-1])
        plt.legend(['train', 'validation', 'test'])
    elif values_2:
        fig, ax = plt.subplots(figsize=(8, 5))
        # plot data in grouped manner of bar type
        plt.bar(x_pos-0.2, values, 0.4, color=colors[1])
        plt.bar(x_pos+0.2, values_2, 0.4, color=colors[3])
        plt.legend(['train', 'validation'])
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.bar(x_pos, values, color=colors)
        if annotate:
            for i in range(len(MODELS)):
                plt.annotate(int(values[i]*1000)/1000, (x_pos[i], values[i]), ha='center', va='center',
                             size=10, xytext=(0,5), textcoords='offset points')

    plt.xticks(x_pos, MODELS)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()