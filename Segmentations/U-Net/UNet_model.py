import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Lambda


# =============================== U-Net Model =============================== # 

class UNet: 

    # __ init __ function
    def __init__(self, input_shape, num_classes):

        """
        input_shape: (height, width, channels)
        num_classes: number of classes
        """
        self.input_shape = input_shape # (height, width, channels)
        self.num_classes = num_classes # number of classes
        self.build() # build the model


    # conv_block function 
    def conv_block(self, inputs, num_filters, kernel_size=3, padding="same", strides=1, kernel_initializer="he_normal"):

        """
        inputs: input tensor
        num_filters: number of filters
        kernel_size: kernel size
        padding: padding type
        strides: strides
        kernel_initializer: kernel initializer


        he normal: He normal initializer draws samples from a truncated normal distribution centered on 0 with
        stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
        it preserves the magnitude of the variance of each layer's output.
        """

        x = Conv2D(num_filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation="relu")(inputs) # convolution layer
        x = Dropout(0.2)(x) # dropout layer
        x = Conv2D(num_filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation="relu")(x) # convolution layer

        return x
    
    # encoder_block function
    def encoder_block(self, inputs, skip, num_filters):
            
        """
        inputs: input tensor
        skip: skip tensor
        num_filters: number of filters
        """

        x = self.conv_block(inputs, num_filters) # convolution block
        p = MaxPooling2D((2, 2))(x) # max pooling layer
        return x, p
    
    # decoder_block function
    def decoder_block(self, inputs, skip, num_filters):
             
        """
        inputs: input tensor
        skip: skip tensor
        num_filters: number of filters
        """
    
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs) # convolution transpose layer
        x = concatenate([skip, x]) # concatenate skip and x
        x = self.conv_block(x, num_filters) # convolution block
        return x
    
    # build function
    def build(self):
              
        """
        build the U-Net model
        """
    
        inputs = Input(self.input_shape)
        input_normalized = Lambda(lambda x: x / 255.0)(inputs)  # Normalize input images

        # encoder
        s1, p1 = self.encoder_block(input_normalized, None, 16)  # encoder block 1
        s2, p2 = self.encoder_block(p1, s1, 32)  # encoder block 2
        s3, p3 = self.encoder_block(p2, s2, 64)  # encoder block 3
        s4, p4 = self.encoder_block(p3, s3, 128)  # encoder block 4

        # bottleneck
        b1 = self.conv_block(p4, 256)  # convolution block

        # decoder
        d1 = self.decoder_block(b1, s4, 128)  # decoder block 1
        d2 = self.decoder_block(d1, s3, 64)  # decoder block 2
        d3 = self.decoder_block(d2, s2, 32)  # decoder block 3
        d4 = self.decoder_block(d3, s1, 16)  # decoder block 4

        # output
        outputs = Conv2D(self.num_classes, (1, 1), padding="same", activation="softmax")(d4)  # convolution layer

        model = Model(inputs, outputs)  # create the model
        return model  # return the model
    



