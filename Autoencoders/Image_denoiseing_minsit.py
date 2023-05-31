from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ====================== MNIST image denoising ======================

def mnist_image_denoising():

    """ This function denoises the MNIST image using autoencoders """

    np.random.seed(42)  # Set the seed for reproducibility

    # Load the dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Add noise to the data
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    # Clip the data to be between 0 and 1
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # Reshape the data
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 28, 28, 1))
    x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 28, 28, 1))

    # Display the noisy images
    plt.figure(figsize=(20, 2))
    for i in range(1, 10):
        ax = plt.subplot(1, 10, i)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.show()

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Up sampling
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    # Train the model
    history = model.fit(x_train_noisy, x_train, epochs=50, batch_size=256, shuffle=True,
                        validation_data=(x_test_noisy, x_test))

    # Evaluate the model
    print(model.evaluate(x_test_noisy, x_test))

    # Plot the accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.show()

    # Predict the test data
    x_test_denoised = model.predict(x_test_noisy)

    # Display the denoised images and the original images
    plt.figure(figsize=(20, 6))
    for i in range(10):
        ax = plt.subplot(3, 10, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, 10, i + 1 + 10)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, 10, i + 1 + 20)
        plt.imshow(x_test_denoised[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

    # Save the denoised images
    for i in range(10):
        cv2.imwrite(f'./images/denoised_images/denoised_image_{i}.png', x_test_denoised[i].reshape(28, 28) * 255)


if __name__ == '__main__':
    mnist_image_denoising()
