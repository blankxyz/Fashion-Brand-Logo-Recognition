import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Model
from keras.utils import np_utils
from keras import layers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
from . import skeras
from . import sfile

path = ''

class CNN(Model):
    def __init__(model, numClass, in_shape = None):
        model.nb_classes = numClass
        model.in_shape = in_shape
        model.BuildModel()

        super().__init__(model.x, model.y)
        model.compile()

    def BuildModel(model):
        numClass = model.nb_classes
        inputShape = model.in_shape
        
        x = layers.Input(inputShape)

        h = layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu',
                              inputShape = inputShape)(x)
        h = layers.Conv2D(64, (3, 3), activation = 'relu')(h)
        h = layers.MaxPooling2D(pool_size = (2, 2))(h)
        h = layers.Dropout(0.25)(h)
        h = layers.Flatten()(h)
        
        zCl, zFL = h, h

        y = layers.Dense(numClass, activation = 'softmax', name = 'preds')(h)

        model.cl_part = Model(x, zCl)
        model.fl_part = Model(x, zFl)

        model.x, model.y = x, y

class DataSet():
    def __init__(self, X, y, numClass, scaling = True,
                 test_size = 0.2, random_state = 0):
        self.X = X
        self.AddChannels()
        X = self.X

        X_train, X_test, y_train, yTest = model_selection.train_test_split(
            X, y, test_size = 0.2, random_state = random_state)
        X_train = X_train.astype('float32')
        X_test = Xtest.astype('float32')

        if scaling:
            scaler = MinMaxScaler()

            n = X_train.shape[0]
            X_train = scaler.fit_transform(
                X_train.reshape(n, -1)).reshape(X_train.shape)
            
            n = X_test.shape[0]
            X_test = scaler.fit_transform(
                X_test.reshape(n, -1)).reshape(X_test.shape)

            self.scaler = scaler

            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')

            Y_train = np_utils.to_categorical(Y_train, numClass)
            Y_test = np_utils.to_categorical(Y_test, numClass)

            self.X_train, self.X_test = X_train, X_test
            self.Y_train, self.Y_test = Y_train, Y_test
            self.y_train, self.yTest = y_train, yTest

    def AddChannels(self):
        X = self.X

        if len(X.shape) == 3:
            N, imgRows, imgCols = X.shape

            if K.image_dim_ordering() == 'th':
                X = X.reshape(X.shape[0], 1, imgRows, imgCols)
                inputShape = (1, imgRows, imgCols)
            else:
                X = X.reshape(X.shape[0], imgRows, imgCols, 1)
                inputShape = (imgRows, imgCols, 1)
        else:
            inputShape = X.shape[1:]

        self.X = X
        self.inputShape = inputShape

class Machine():
    def __init__(self, X, y, numClass = 2, figure = True):
        self.numClass = numClass
        self.SetData(X, y)
        self.SetModel()
        self.figure = figure

    def SetData(self, X, y):
        numClass = self.numClass
        self.data = DataSet(X, y, numClass)

    def SetModel(self):
        numClass = self.numClass
        data = self.data
        self.model = CNN(numClass = nb,
                         in_shape = data.inputShape)

    def fit(self, numEpoch = 10, batchSize = 128, verbose = 1):
        data = self.data
        Model = self.model
        
        history = model.fit(data.X_train, data.Y_train,
                            batchSize = batchSize, epochs = numEpoch,
                            verbose = verbose,
                            validation_data = (data.X_test, data.Y_test))

        return history

    def run(self, numEpoch = 10, batchSize = 128, verbose = 1):
        data = self.data
        model = self.model
        figure = self.figure

        history = self.fit(numEpoch = numEpoch, batchSize = batchSize,
                           verbose = 1)
        
        score = model.evaluate(data.X_test, data.Y_test, verbose = 0)

        print('Confusion matrix')
        Y_test_pred = model.predict(data.X_test, verbose = 0)
        y_test_pred = np.argmax(Y_test_pred, axis = 1)
        print(metrics.confusion_matrix(data.y_test, y_test_pred))

        print('Test score:', score[0])
        print('Test accurary:', score[1])

        suffix = sfile.unique_filename('datatime')
        foldname = 'output_' + suffix
        os.makedirs(foldname)
        skeras.save_history('history_history.npy',
                             history.history, fold = foldname)
        model.save_weights(os.path.join(foldname, 'dl_model.h5'))
        print('Output results are saved in', foldname)

        if figure:
            plt.figure(figsize = (12, 4))
            plt.subplot(1, 2, 1)
            skeras.plot_acc(history)
            plt.subplot(1, 2, 2)
            plt.show()

        self.history = history