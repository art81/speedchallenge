#!/usr/bin/python3

import pandas as pd
import tensorflow as tf
import os
import sys
import numpy as np
from tensorflow.keras.layers import Input,Flatten,Dense,ConvLSTM2D,BatchNormalization,MaxPooling3D,TimeDistributed
from tensorflow.keras.models import Sequential

from datagenerator_LSTM_Class import DataGenerator_LSTM

homePath = os.path.expanduser("~")

# ***** GLOBAL USER INPUTS *****
modelPath = homePath + "/Desktop/speedchallenge/wandb/run-20201107_194431-3lvjr1fo/files/model.h5"

prePath = homePath + "/Desktop/speedchallenge/data/"
trainDatasetPath = prePath + 'train/'
testDatasetPath  = prePath + 'test/'
trainVideoPath   = prePath + 'train.mp4'
testVideoPath    = prePath + 'test.mp4'
trainResultPath  = prePath + 'train.txt'
testResultPath   = prePath + 'test.txt'

imageShape = (480, 640, 3) # resolution of video

# ***** GLOBAL CONSTANTS *****
BATCH_SIZE = 32
MAX_SPEED  = 100.0
NUM_EPOCHS = 2

# Implemented with Keras sequential with the architecture taken from Nvidias "End to End Learning of Self-Driving Cars" paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def build_model(input_shape):
    #create model
    model = Sequential()

    print(input_shape)
    model.add(Input(input_shape))

    #add model layers
    model.add(ConvLSTM2D(filters=24, kernel_size=(5, 5), strides=(2, 2), data_format='channels_last', padding="same", activation='tanh', return_sequences=True))

    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first'))

    model.add(ConvLSTM2D(filters=48, kernel_size=(5, 5), strides=(2, 2), data_format='channels_last', padding="same", activation='tanh', return_sequences=True))

    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first'))

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', padding='same', activation='tanh', return_sequences=True))

    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first'))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dense(216, activation='relu')))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    #model.add(Flatten())
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #model.add(Dense(1, activation='sigmoid'))

    return model

# Generates a dataset (folder) of images from a video path
def generate_dataset(videoPath, datasetPath):
    vidObj = cv2.VideoCapture(videoPath)
      
    # checks whether frames were extracted 
    success = 1
    count = 0
    while True:
        # vidObj object calls read 
        # Saves the image to the images folder and 
        success, image = vidObj.read() 
        if success:
            cv2.imwrite(datasetPath + str(count) + ".jpg", image)
            count += 1
        else:
            break

    vidObj.release()
  
# Function reads in the path to a video file and creates a Keras datagenerator with the frames in the video and the labels array that is passed as an input
# If labels is None then a datagen is created with just the images - useful for predicting speeds from a given video
def get_imageFileNames(datasetPath): 
    datasetSize = len(os.listdir(datasetPath))
    fileNames = [(str(i) + ".jpg") for i in range(datasetSize)]

    return fileNames, datasetSize

# Returns an array of values from a txt file that has a new value on each line - in this application these values are speeds
def get_speeds(path):
    speeds = []
    with open(path) as my_file:
        for line in my_file:
            speeds.append(float(line))

    #Change speeds to np array with right shape
    size = len(speeds)
    speeds = np.array(speeds)
    speeds = np.reshape(speeds, (size,))
    speeds = speeds / MAX_SPEED #normalize 0 to 1

    return speeds

# Writes an array to a text file
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

    useWandB = None #set by user later
    if numArguments != 2 or (not arguments[1] is None and not (arguments[1] in ["BUILD", "TRAIN", "TEST"])):
        print("USAGE: \npython3", str(arguments[0]), "{BUILD / TRAIN / TEST}")
        sys.exit(0)
    else:
        useWandB = input("Enter 1 to log to Weights and Biases: ")
        useWandB = (useWandB == "1")

        if useWandB:
            # Init W and B
            import wandb
            from wandb.keras import WandbCallback

            wandb.init(project="comma_ai_challenge")

    # Main menu logic
    if arguments[1] == "BUILD":
        print("*****Building Dataset from input Videos*****")

        generate_dataset(trainVideoPath, trainDatasetPath)
        generate_dataset(testVideoPath, testDatasetPath)

    elif arguments[1] == "TRAIN":
        print("*****Training Model from Built Dataset*****")

        # Get Speeds and image file names
        speeds = get_speeds(trainResultPath)
        imageFileNames, datasetSize = get_imageFileNames(trainDatasetPath)

        # Get Datagenerator from imported custom py file
        DG = DataGenerator_LSTM(speeds, imageFileNames, trainDatasetPath, to_fit=True, batch_size=BATCH_SIZE, image_shape=(480, 640), n_channels=3, shuffle=False)

        batch, y = DG.__getitem__(0)
        print("Batch Shape: " + str(batch.shape))
        print("Output Value: " + str(y))

        model = build_model((BATCH_SIZE, imageShape[0], imageShape[1], imageShape[2]))
        model.summary()

        print("***** Training Model *****")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3), loss="mse", metrics=['mse', 'mae', 'mape'])

        if useWandB:
            history = model.fit(DG, steps_per_epoch=datasetSize / BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[WandbCallback()])

            # Save trained model to wandb
            model.save(os.path.join(wandb.run.dir, "model.h5"))
        else:
            history = model.fit(DG.generator_wrapper(), steps_per_epoch=datasetSize, epochs=NUM_EPOCHS)

    elif arguments[1] == "TEST":
        print("*****Testing Trained Model*****")

        #Load Trained Model
        trainedModel = tf.keras.models.load_model(modelPath)

        # Get Datagenerator
        datagen, datasetSize = get_datagen_from_dataset_and_labels(testDatasetPath)

        predictedSpeeds = trainedModel.predict(datagen)
        predictedSpeeds = predictedSpeeds * MAX_SPEED # scale back up to be from [0, MAX_SPEED] instead of [0, 1]

        print(predictedSpeeds)

        writeToFile(testResultPath, predictedSpeeds)

