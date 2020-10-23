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

# ***** USER INPUTS *****
modelPath = "/home/tarnoa2/Desktop/speedchallenge/wandb/run-20201022_221123-3o4qtdfc/files/model.h5"
datasetDir = '/home/tarnoa2/Desktop/speedchallenge/images/images2/'

prePath = '/home/tarnoa2/Desktop/speedchallenge/data/'
trainDatasetPath = prePath + 'train.mp4'
testDatasetPath  = prePath + 'test.mp4'
trainResultPath  = prePath + 'train.txt'
testResultPath   = prePath + 'test.txt'

imageShape = (480, 640, 3)

BATCH_SIZE = 32
MAX_SPEED = 100

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
  
# Function to read in the path to a video file and create a Keras ImageGenerator dataset with the frames in the video
def build_dataset_from_video(path, labels): 
    doDatasetGen = False

    if doDatasetGen:
        vidObj = cv2.VideoCapture(path)
      
        # checks whether frames were extracted 
        success = 1
        count = 0
        while True:
            # vidObj object calls read 
            # Saves the image to the images folder and 
            success, image = vidObj.read() 
            if success:
                cv2.imwrite(datasetDir + "a" + str(count) + ".jpg", image)
                count += 1
            else:
                break

        vidObj.release()

    print("LABELS***********")
    labels = labels.flatten()

    print(labels)

    # All images are now saved to the datasetDir - convert to keras dataset with image_dataset_from_directory
    # convert to keras dataset using a pandas dataframe
    fileNames = [("a" + str(i) + ".jpg") for i in range(len(labels))]
    df =  pd.DataFrame(list(zip(list(fileNames), list(labels))), columns=["imageFileNames", "speeds"])
    #df = df.astype({'imageFileNames': "string", 'speeds': np.float32}).dtypes

    return df

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

    return speeds, size

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
        wandb.init(project="comma_ai_challenge")
        if arguments[1] == "TEST":
            trainDataset = False

    if trainDataset:
        print("*****Building Dataset to Train Model*****")

        # Get Speeds
        speeds, datasetSize = get_speeds(trainResultPath)

        # Get Datagenerator
        df = build_dataset_from_video(trainDatasetPath, speeds)

        print("Dataframe************************")
        print(df)

        # Convert dataset to a data generator with pandas
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        print("Datagen************************")
        print(datagen)

        datagen = datagen.flow_from_dataframe(df, directory=datasetDir, x_col="imageFileNames", y_col="speeds", class_mode="raw", target_size=(imageShape[1], imageShape[0]), batch_size=BATCH_SIZE, shuffle=True)

        print(datagen.next())

        model = build_model(imageShape)
        model.summary()
        numEpochs = 100


        print("***** Training Model *****")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3), loss="sparse_categorical_crossentropy", metrics="mse")
        history = model.fit(datagen, steps_per_epoch=datasetSize / BATCH_SIZE, epochs=numEpochs, callbacks=[WandbCallback()])

        # Save trained model to wandb
        model.save(os.path.join(wandb.run.dir, "model.h5"))
    else:
        print("*****Testing Trained Model*****")

        # Get Test dataset
        frameTestArray = get_frames(testDatasetPath)

        trainedModel = tf.keras.models.load_model(modelPath)

        predictedSpeeds = trainedModel.predict(frameTestArray)

        writeToFile(testResultPath, predictedSpeeds)

