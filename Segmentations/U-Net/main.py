from UNet_model import UNet 
import os
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt



# main function



if __name__ == "__main__":


    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    TRAIN_PATH = './dataset/train/'
    TEST_PATH = './dataset/test/'

    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img  #Fill empty X_train with values from img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)  

        Y_train[n] = mask   

    # test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Resizing test images') 
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    print('Done!')

    image_x = random.randint(0, len(train_ids))
    imshow(X_train[image_x])
    plt.show()
    imshow(np.squeeze(Y_train[image_x]))
    plt.show()




      
    # create the model
    unet = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)  # create the model

    model = unet.build()  # build the model
    print(model.summary())  # print the model summary

    # # save the model as a png file
    # tf.keras.utils.plot_model(model, to_file='./U-Net.png', show_shapes=True, show_layer_names=True)  # save the model as a png file

    #  compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #  fit the model
    model.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=16)
    
    
    
        ####################################

    idx = random.randint(0, len(X_train))


    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)


    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)


    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    imshow(X_train[int(X_train.shape[0]*0.9):][ix])
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()