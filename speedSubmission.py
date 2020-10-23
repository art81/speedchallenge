# Init W and B
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
import cv2 
import os
import sys
import numpy as np
from tensorflow.keras.layers import Conv2D,Activation,Lambda,Flatten,Dense
from tensorflow.keras.models import Sequential

# ***** USER INPUTS *****
modelPath = "/home/tarnoa2/Desktop/speedchallenge/wandb/run-20201022_221123-3o4qtdfc/files/model.h5"

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

def count_frames_manually(path):
    vid = cv2.VideoCapture(path)

    # Will count the total number of frames in the object
    count = 0
    success = 1
    while success: 
        # vidObj object calls read 
        # function extract frames and count while we can get a new frame
        success, image = vid.read() 
        count += 1

    vid.release()

    return count - 1 #since one extra is added in the loop
  
# Code below for extracting frames adapted from: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
# Function to extract frames 
def get_frames(path, size=np.inf): 
    # Path to video file 
    vidObj = cv2.VideoCapture(path)

    if size == np.inf:
        numFrames = count_frames_manually(path)
        size = numFrames
  
    # checks whether frames were extracted 
    success = 1
    frameArray = []
    count = 0
    while success and count < size: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 

        if len(frameArray) == 0:
            frameArray = np.zeros((size, image.shape[0], image.shape[1], image.shape[2]))

        frameArray[count] = image
        count += 1

    vidObj.release()
    return frameArray

def get_speeds(path):
    speeds = []
    with open(path) as my_file:
        for line in my_file:
            speeds.append(float(line))

    return speeds

def writeToFile(path, a):
    f = open(path, "w")
    for val in a:
        f.write(str(val) + "\n")

    f.close()
  

# ***** MAIN *****
if __name__ == '__main__': 
    # Read inputs and give help if no argument given
    numArguments = len(sys.argv)
    arguments    = sys.argv
    trainDataset = True

    if numArguments != 2:
        print("USAGE: \npython3", str(arguments[0]), "{TRAIN / TEST}")
    else:
        #wandb.init(project="comma_ai_challenge")
        if arguments[1] == "TEST":
            trainDataset = False

    if trainDataset:
        print("*****TRAINING MODEL*****")

        # Get Speeds
        speeds = get_speeds('/home/tarnoa2/Downloads/train.txt')
        sizeDataset = int(len(speeds) / 100)

        # Get Input Shape
        frameArray = get_frames('/home/tarnoa2/Downloads/train.mp4', sizeDataset)
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
        history = model.fit(frameArray, speeds, epochs=numEpochs, callbacks=[WandbCallback()])

        # Save trained model to wandb
        model.save(os.path.join(wandb.run.dir, "model.h5"))
    else:
        print("*****TESTING TRAINED MODEL*****")

        # Get Test dataset
        frameTestArray = get_frames('/home/tarnoa2/Downloads/test.mp4', 204)

        trainedModel = tf.keras.models.load_model(modelPath)

        predictedSpeeds = trainedModel.predict(frameTestArray)

        writeToFile("test.txt", predictedSpeeds)

