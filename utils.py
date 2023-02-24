import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# define constants
IMG_SIZE = 224
BATCH_SIZE = 32


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
    

def apply_masks(df, masks_path, img_size, masked_path):
    """
    Apply masks to original images

    Arguments:
    df -- dataframe containing images paths
    masks_path -- full path where masks of the images are saved
    img_size -- size of the images
    masked_path -- full path where masked images are saved
    """
    for i in range(df.shape[0]):
        im_path = df.loc[i, 'image']
        im_name = im_path.split('/')[-1]
        image = cv2.imread(im_path)
        mask_im = cv2.imread(masks_path + im_name, 0)
        # resize image
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # apply mask to original image
        masked_im = cv2.bitwise_and(image, image, mask=mask_im)
        # save masked image
        cv2.imwrite(masked_path + im_name, masked_im)


def visualize_preprocessing(df, or_path, clahe_path, mask_path, masked_path, img_size=IMG_SIZE, ind=None):
    """
    Visualize preprocessing pipeline

    Arguments:
    df -- dataframe containing images paths
    or_path -- full path where original images are saved
    clahe_path -- full path where contrast enhanced images are saved
    mask_path -- full path where the masks of the images are saved
    masked_path -- full path where masked images are saved
    img_size -- size of the images
    ind -- if not None, index corresponding to the image to visualize
    """
    _, axs = plt.subplots(1, 5, figsize=(13, 17))
    [ax.axis('off') for ax in axs.flatten()]

    if not ind:
       ind = np.random.randint(0, df.shape[0])
    
    name = df.loc[ind, 'image'].split('/')[-1]
    original = cv2.imread(or_path + name)
    clahe = cv2.imread(clahe_path + name)
    resize = cv2.resize(clahe, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.imread(mask_path + name)
    masked = cv2.imread(masked_path + name)

    imgs = [original, clahe, resize, mask, masked]
    titles = ['original image', 'CLAHE', 'resized image', 'mask', 'masked image']

    for i, ax in enumerate(axs.flatten()):
      ax.imshow(imgs[i])
      ax.set_title(titles[i])
      
    plt.tight_layout()


def gen_2inputs(datagen, df, img_size, batch_size=BATCH_SIZE, shuffle=False, seed=None, color_mode='grayscale'):
    """
    Merge images and gender in a single generator 
    
    Arguments:
    datagen -- ImageDataGenerator used for generating batches of augmented data
    df -- dataframe used to flow data into the generator
    img_size -- size of the images
    batch_size -- size of the training batches
    suffle -- if True generator shuffles the data received
    seed -- required in order to have the same shuffling for image and gender data
    color_mode -- if grayscale images are in the format (img_size, img_size, 1)
    """

    gen_img = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='boneage_n', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(img_size, img_size), color_mode=color_mode, drop_duplicates=False)
    
    gen_gender = datagen.flow_from_dataframe(dataframe=df,
        x_col='image', y_col='gender(M/F)', batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode='other',
        target_size=(img_size, img_size), color_mode=color_mode, drop_duplicates=False)
    
    while True:
        X1i = gen_img.next()
        X2i = gen_gender.next()
        yield [X1i[0], X2i[1]], X1i[1]


def load_hand_image(img_name, img_size=IMG_SIZE, grayscale=False):

    if isinstance(img_name, bytes):
        img_name = img_name.decode()

    if grayscale:
      img = cv2.imread(img_name,0)
    else:
      img = cv2.imread(img_name, cv2.IMREAD_COLOR)

    img = np.array(cv2.resize(img, (img_size,img_size)), dtype='float32')

    return img

def normalize_values(image, min=0, max=255):
  return(  (np.asarray(image)-min) / (max-min) )

def image_contrast_enhancement(image, factor=2.5, min=0, max=1):
  img = tf.image.adjust_contrast(image, factor)
  return tf.clip_by_value(img, min, max)

def contrast_enhancement(image, label, factor=3.0, min=0, max=1):
  img = tf.image.adjust_contrast(image, factor)
  return tf.clip_by_value(img, min, max), tf.convert_to_tensor(label, dtype=tf.float32)

def display_image(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
def plot_loss(model_history):
  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  plt.figure(figsize=(10, 5))
  plt.yscale("log")
  plt.plot(model_history.epoch, loss, label='Training loss')
  plt.plot(model_history.epoch, val_loss, label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.legend()
  plt.show()

def plot_accuracy(model_history):
  acc = model_history.history['accuracy']
  val_acc = model_history.history['val_accuracy']

  plt.figure(figsize=(10, 5))
  plt.plot(model_history.epoch, acc, label='Training accuracy')
  plt.plot(model_history.epoch, val_acc, label='Validation accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

def show_predictions(model, dataset, batch_size):
  for a in dataset.take(1):
    length = len(a)

  if length==2: #train dataset
    for image, label in dataset.unbatch().shuffle(buffer_size=batch_size).take(1):
      numpy_img = np.repeat(np.expand_dims(image.numpy(), axis=0), batch_size, axis=0)      
      pred = model.predict(numpy_img)
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(pred[0])
    ax[1].imshow(label)

  else:      #test dataset
    for image in dataset.unbatch().shuffle(buffer_size=batch_size).take(1):
      numpy_img = np.repeat(np.expand_dims(image.numpy(), axis=0), batch_size, axis=0)
      pred = model.predict(numpy_img)
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    pred = tf.where(pred < 0.5, 0.0, 1.0)               # binarization provides the actual mask
    ax[0].imshow(pred[0])
    ax[1].imshow(image)
    
def plot_dataset(ds, batched):
  if batched:
    ds_list = list(ds.unbatch().as_numpy_iterator())
    fig, ax = plt.subplots(2,len(ds_list), figsize=(20, 5))
  else:
    ds_list = list(ds.as_numpy_iterator())
    fig, ax = plt.subplots(2,len(ds_list), figsize=(20, 5))

  for i in range(len(ds_list)):
      img, label = ds_list[i]
      ax[0,i].imshow(img)
      ax[0,i].axis('off')
      ax[1,i].imshow(label)
      ax[1,i].axis('off')
  plt.show()