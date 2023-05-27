import tensorflow as tf
from UNet_model import UNet 
import Read_preprocess_data as rp
import os


# main function



if __name__ == "__main__":

    # read and preprocess the data
    # Set the parameters
    seed = 42
    # IMG_WIDTH = 256
    # IMG_HEIGHT = 256
    IMG_CHANNELS = 3
    TRAIN_PATH = './dataset/train/'
    TEST_PATH = './dataset/test/'

    # Set the seed
    rp.set_seed(seed)

    # Read and preprocess the training data
    train_ids = next(iter(os.walk(TRAIN_PATH)))[1]
    X_train, Y_train = rp.read_train_images(train_ids, TRAIN_PATH, channels=IMG_CHANNELS)

    print("X_train shape: {}".format(len(X_train)))
    print("Y_train shape: {}".format(len(Y_train)))

    # Read and preprocess the test data
    test_ids = next(iter(os.walk(TEST_PATH)))[1]
    X_test = rp.read_test_images(test_ids, TEST_PATH, channels= IMG_CHANNELS)

    print("X_test shape: {}".format(len(X_test)))

    print('Done!')

    # Visualize the data
    # visualize_data(X_train, Y_train)



      
    # # create the model
    # IMG_WIDTH = 256
    # IMG_HEIGHT = 256
    # IMG_CHANNELS = 3

    # unet = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)  # create the model

    # model = unet.build()  # build the model
    # print(model.summary())  # print the model summary

    # # # save the model as a png file
    # # tf.keras.utils.plot_model(model, to_file='./U-Net.png', show_shapes=True, show_layer_names=True)  # save the model as a png file

    # #  compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # #  fit the model
    # model.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=16)

    



