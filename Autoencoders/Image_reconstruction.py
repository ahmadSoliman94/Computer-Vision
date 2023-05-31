from matplotlib.pyplot import imshow, show
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential



# ====================== Image reconstruction ======================



def image_reconstruction():

    """ This function reconstructs the image using autoencoders """

    np.random.seed(42) # Set the seed for reproducibility

    SIZE = 256 # Size of the image
    img_data = [] # Empty list to store the image data

    # Read and preprocess the image
    img = cv2.imread('images/sea.jpg', 1)   # Change 1 to 0 for grey images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB to show images in true colors
    img = cv2.resize(img, (SIZE, SIZE)) # Resize the image to 256x256
    img_data.append(img_to_array(img)) # Convert the image to an array

    # Prepare the input image array
    img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3)) # Convert the array to numpy array and reshape it
    img_array = img_array.astype('float32') / 255. # Normalize the array

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

    # Upsampling layers for reconstruction
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(img_array, img_array, epochs=1000, shuffle=True)

    # Generate reconstructed image using the trained model
    print("Neural network output")
    pred = model.predict(img_array)

    # Scale the reconstructed image
    pred = np.clip(pred, 0., 1.)  # Limit the values in the array between 0 and 1



    # Save the reconstructed image
    cv2.imwrite('images/reconstructed_image.jpg', pred[0].reshape(SIZE, SIZE, 3) * 255.)


if __name__ == '__main__':
    image_reconstruction()  # Call the image_reconstruction function