import glob
import numpy as np
import keras as K
from keras import backend
from PIL import Image

directory = '' # Set working directory
categories = [''] # Set Categories

imgRows = 0 # Set image height
imgCols = 0 # Set image width
imgChannels = 3 # Red, Green, Blue

class Dataset():
    def __init__(self):
        numOfClass = 0 # Write Number of classes

        X, y = [], []

        for i, category in enumerate(categories):
            label = i

            imageDir = directory + '/' + category
            files = glob.glob(imageDir + '/*.jpg')

            for j, file in enumerate(files):
                img = Image.open(file)
                img = img.convert('RGB')
                img = img.resize((imgWidth, imgHeight))
                data = np.asarray(img)

                X.append(data)
                y.append(label)

        X_train = np.array(X[0:(3 / 4) * len(X), :]) # Set range of X_train data
        y_train = np.array(y[0:(3 / 4) * len(y), :]) # Set range of y_train data
        X_test = np.array(X[0:(1 / 4) * len(X), :]) # Set range of X_test data
        y_test = np.array(y[0:(1 / 4) * len(y), :]) # Set range of y_test data

        backend.set_image_data_format('channels_last')
        X_train = X_train.reshape(X_train.shape[0], imgRows, imgCols, imgChannels)
        X_test = X_test.reshape(X_test.shape[0], imgRows, imgCols, imgChannels)
        inputShape = (imgRows, imgCols, imgChannels)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        y_train = K.utils.to_categorical(y_train, numOfClass)
        y_test = K.utils.to_categorical(y_test, numOfClass)

        self.inputShape = inputShape
        self.numOfClass = numOfClass
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test