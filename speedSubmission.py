import tensorflow as tf
import cv2 
import numpy as np
from tensorflow.keras.layers import Conv2D,Activation,Lambda,Flatten,Dense
from tensorflow.keras.models import Sequential

# Model architecture taken from Nvidias "End to End Learning of Self-Driving Cars" paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def build_model(input_shape):
    #create model
    model = Sequential()

    #add model layers
    model.add(Lambda(lambda x: x/255.0, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))

    return model
  
# Code below for extracting frames adapted from: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
# Function to extract frames 
def get_frames(path, size): 
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    frameArray = []
  
    while success and count < size: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 

        if len(frameArray) == 0:
            frameArray = np.zeros((size, image.shape[0], image.shape[1], image.shape[2]))

        frameArray[count] = image
  
        count += 1

    return frameArray

def get_speeds(path, size):
    speeds = []
    count = 0
    with open(path) as my_file:
        for line in my_file:
            if count < size:
                speeds.append(float(line))
                count += 1
            else:
                return speeds

    return speeds
  

# ***** MAIN *****
if __name__ == '__main__': 
    sizeDataset = int(20400 / 100)
  
    # Get Input Shape
    frameArray = get_frames('/home/tarnoa2/Downloads/train.mp4', sizeDataset)
    speeds     = get_speeds('/home/tarnoa2/Downloads/train.txt', sizeDataset)
    shapeImage = frameArray[0].shape

    #Change speeds to np array with right shape
    speeds = np.array(speeds)
    speeds = np.reshape(speeds, (sizeDataset, 1))

    model = build_model(shapeImage)
    model.summary()

    print(np.array(speeds).shape)
    print(frameArray.shape)
    numEpochs = 100

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3 / numEpochs), loss="mse", metrics="mse")
    history = model.fit(frameArray, speeds, epochs=numEpochs)

    # Save trained model
    model.save('./')