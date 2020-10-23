import tensorflow as tf
import cv2 
import numpy as np
from tensorflow.keras.layers import Conv2D,Activation,Lambda,Flatten,Dense

# Model architecture taken from Nvidias "End to End Learning of Self-Driving Cars" paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def build_model(input_shape):
    # Input Normalization
    x = Lambda(lambda x: x/255.0)(input_shape)

    # Layer 1: CONV-RELU 24 5x5
    x = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(x)
    x = Activation("relu")(x)
    # Layer 2: CONV-RELU 36 5x5
    x = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(x)
    x = Activation("relu")(x)
    # Layer 3: CONV-RELU 48 5x5
    x = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(x)
    x = Activation("relu")(x)
    # Layer 4: CONV-RELU 64 3x3
    x = Conv2D(64, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    # Layer 5: CONV-RELU 64 3x3
    x = Conv2D(64, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)

    # Flatten Layer
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(1164, kernel_initializer='normal', activation='relu')(x)
    x = Dense(100, kernel_initializer='normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='normal', activation='relu')(x)
    x = Dense(10, kernel_initializer='normal', activation='relu')(x)
    x = Dense(1, kernel_initializer='normal', name="Linear")(x)

    return x

def build_model():
    # Shape of dataset images is...


    input_shape = tf.keras.Input(shape=(480, 640, 3))
    model = FrankNet.build_linear_branch(inputs)
    angularVelocity = FrankNet.build_angular_branch(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=[
        linearVelocity, angularVelocity], name="FrankNet")

    return model

def train_model():
    # Shape of dataset images is...


    input_shape = tf.keras.Input(shape=(X, Y, Z))
    model = build_model(input_shape)

observation = []
linear = []
angular = []
  
# Code below for extracting frames adapted from: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
# Function to extract frames 
def getFrames(path, size): 
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    frameArray = []
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        print(np.max(image))

        if len(frameArray) == 0:
            frameArray = np.zeros((size, image.shape[0], image.shape[1], image.shape[2]), dtype=np.int8)

        frameArray[count] = image
  
        count += 1
  

# ***** MAIN *****
if __name__ == '__main__': 
  
    # Get Input Shape
    frameArray = getFrames('/home/tarnoa2/Downloads/train.mp4', 20400)
    shapeImage = frameArray(0).shape

    print(shapeImage)