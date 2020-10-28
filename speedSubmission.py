# Init W and B
import wandb
from wandb.keras import WandbCallback

import pandas as pd
import tensorflow as tf
import cv2 
import os
import sys
import numpy as np
from tensorflow.keras.layers import Conv2D,Activation,Lambda,Flatten,Dense
from tensorflow.keras.models import Sequential

homePath = os.path.expanduser("~")

# ***** GLOBAL USER INPUTS *****
modelPath = homePath + "/Desktop/speedchallenge/wandb/run-20201027_174505-15b8a5sm/files/model.h5"

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
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

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
def get_datagen_from_dataset_and_labels(datasetPath, labels=None): 
    global imageShape, modelPath

    datasetSize = len(os.listdir(datasetPath))
    fileNames = [(str(i) + ".jpg") for i in range(datasetSize)]

    # All images are saved in datasetPath - convert to keras dataset with image_dataset_from_directory
    if labels is None:
        df =  pd.DataFrame(list(fileNames), columns=["imageFileNames"])

        # Convert dataset to a data generator with pandas
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        datagen = datagen.flow_from_dataframe(df, directory=datasetPath, x_col="imageFileNames", class_mode=None, target_size=(imageShape[1], imageShape[0]), batch_size=BATCH_SIZE, shuffle=True)
    else:
        labels = labels.flatten()

        # convert to keras dataset using a pandas dataframe
        df =  pd.DataFrame(list(zip(list(fileNames), list(labels))), columns=["imageFileNames", "speeds"])

        # Convert dataset to a data generator with pandas
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        datagen = datagen.flow_from_dataframe(df, directory=datasetPath, x_col="imageFileNames", y_col="speeds", class_mode="raw", target_size=(imageShape[1], imageShape[0]), batch_size=BATCH_SIZE, shuffle=False)

    return datagen, datasetSize

# Returns an array of values from a txt file that has a new value on each line - in this application these values are speeds
def get_speeds(path):
    speeds = []
    with open(path) as my_file:
        for line in my_file:
            speeds.append(float(line))

    #Change speeds to np array with right shape
    size = len(speeds)
    speeds = np.array(speeds)
    speeds = np.reshape(speeds, (size, 1))
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
    if numArguments != 2:
        print("USAGE: \npython3", str(arguments[0]), "{BUILD / TRAIN / TEST}")
    else:
        useWandB = input("Enter 1 to log to Weights and Biases: ")

        if useWandB:
            wandb.init(project="comma_ai_challenge")

    # Main menu logic
    if arguments[1] == "BUILD":
        print("*****Building Dataset from input Videos*****")

        generate_dataset(trainVideoPath, trainDatasetPath)
        generate_dataset(testVideoPath, testDatasetPath)

    elif arguments[1] == "TRAIN":
        print("*****Training Model from Built Dataset*****")

        # Get Speeds
        speeds = get_speeds(trainResultPath)

        # Get Datagenerator
        datagen, datasetSize = get_datagen_from_dataset_and_labels(trainDatasetPath, speeds)

        model = build_model(imageShape)
        model.summary()

        print("***** Training Model *****")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3), loss="mse", metrics=['mse', 'mae', 'mape'])

        if useWandB:
            history = model.fit(datagen, steps_per_epoch=datasetSize / BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[WandbCallback()])

            # Save trained model to wandb
            model.save(os.path.join(wandb.run.dir, "model.h5"))
        else:
            history = model.fit(datagen, steps_per_epoch=datasetSize / BATCH_SIZE, epochs=NUM_EPOCHS)

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

