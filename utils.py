import pandas as pd
import numpy as np
from keras.metrics import mean_absolute_error

# IMG_SIZE = 224 # to be modified


def load_filenames(file, path, path_p=None):
    """
    Create pandas dataframe with image path, gender and bone age as columns 
    
    Arguments:
    file -- file to read
    path -- full path where to find the images
    path_p -- additional path for images (needed for validation set)

    Returns:
    df -- output pandas dataframe
    """

    if 'Test' in file:
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    # Uniforming the dfs
    # reorder columns
    if 'train' in file:
        df = df[['id', 'male', 'boneage']]
    
    # rename columns
    df.set_axis(['image', 'gender(M/F)', 'boneage'], axis=1, inplace=True)

    # changing values for gender
    df['gender(M/F)'].replace({'M':np.array([1]), 'F':np.array([0])}, inplace=True) # for test set only
    df['gender(M/F)'].replace({True:np.array([1]), False:np.array([0])}, inplace=True) 

    # setting up the paths to get the images
    if path_p: # validation set images are divided in two folders
        mask_1 = df['image'] < 9727
        mask_2 = df['image'] >= 9727
        df.loc[mask_1, 'image'] = path + '/' + df.loc[mask_1, 'image'].astype(str) + '.png'
        df.loc[mask_2, 'image'] = path_p + '/' + df.loc[mask_2, 'image'].astype(str) + '.png'
    else: 
        df['image'] = path + '/' + df['image'].astype(str) + '.png'

    return df


def mae_in_months(x, y):
    """
    Return mean absolute error in months

    Arguments:
    x -- predicted value
    y -- ground-truth
    """
    return mean_absolute_error((std_bone_age*x + mean_bone_age), (std_bone_age*y + mean_bone_age)) 


def gen_2inputs(datagen, df, batch_size, shuffle=False, seed=None):
    """
    Merge images and gender in a single generator 
    
    Arguments:
    datagen -- ImageDataGenerator used for generating batches of augmented data
    df -- dataframe used to flow data into the generator
    batch_size -- size of the training batches
    suffle -- if True generator shuffles the data received
    seed -- required in order to have the same shuffling for image and gender data
    """

    gen_img = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='boneage_n', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale', drop_duplicates=False)
    
    gen_gender = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='gender(M/F)', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale', drop_duplicates=False)
    
    while True:
        X1i = gen_img.next()
        X2i = gen_gender.next()
        yield [X1i[0], X2i[1]], X1i[1]