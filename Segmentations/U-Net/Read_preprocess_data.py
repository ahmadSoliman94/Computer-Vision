import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)

def read_train_images(train_ids, path, channels):
    X_train = []
    Y_train = []
    for id_ in tqdm(train_ids, total=len(train_ids)):
        img = imread(path + id_ + '/images/' + id_ + '.png')[:,:,:channels]
        X_train.append(img)
        mask = []
        for mask_file in next(os.walk(path + id_ + '/masks/'))[2]:
            mask_ = imread(path + id_ + '/masks/' + mask_file)
            mask.append(mask_)
        Y_train.append(mask)

    return X_train, Y_train

def read_test_images(test_ids, path, channels):
    X_test = []
    for id_ in tqdm(test_ids, total=len(test_ids)):
        img = imread(path + id_ + '/images/' + id_ + '.png')[:,:,:channels]
        X_test.append(img)

    return X_test

def visualize_data(X_train, Y_train):
    image_x = random.randint(0, len(X_train))
    imshow(X_train[image_x])
    plt.show()
    for mask in Y_train[image_x]:
        imshow(mask)
        plt.show()
