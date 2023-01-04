import pandas as pd
import numpy as np
import cv2
from keras.metrics import mean_absolute_error


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
    

def apply_masks(df, masks_path, img_size, num_images):
    """
    Apply masks to original images

    Arguments:
    df -- dataframe containing images paths
    masks_path -- full path where images are placed
    img_size -- size of the images
    num_images -- number of images to which the masks have to be applied
    """

    for i in range(num_images):
        im_path = df.loc[i, 'image']
        im_name = im_path.split('/')[-1]
        image = cv2.imread(im_path)
        # mask_im = cv2.imread(path + '/' + masks_path + im_name, 0)
        mask_im = cv2.imread(masks_path + im_name, 0)
        # apply mask to original image
        masked_im = cv2.bitwise_and(image, image, mask=mask_im)
        # resize image
        masked_im = cv2.resize(masked_im, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # overwrite the original image with its masked version
        cv2.imwrite(im_path, masked_im)


def mae_in_months(x, y, std, mean):
    """
    Return mean absolute error in months

    Arguments:
    x -- predicted value
    y -- ground-truth
    std -- standard deviation of data
    mean -- mean of data
    """
    return mean_absolute_error((std*x + mean), (std*y + mean)) 


def gen_2inputs(datagen, df, img_size, batch_size, shuffle=False, seed=None):
    """
    Merge images and gender in a single generator 
    
    Arguments:
    datagen -- ImageDataGenerator used for generating batches of augmented data
    df -- dataframe used to flow data into the generator
    img_size -- size of the images
    batch_size -- size of the training batches
    suffle -- if True generator shuffles the data received
    seed -- required in order to have the same shuffling for image and gender data
    """

    gen_img = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='boneage_n', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(img_size, img_size), color_mode='grayscale', drop_duplicates=False)
    
    gen_gender = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='gender(M/F)', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(img_size, img_size), color_mode='grayscale', drop_duplicates=False)
    
    while True:
        X1i = gen_img.next()
        X2i = gen_gender.next()
        yield [X1i[0], X2i[1]], X1i[1]
