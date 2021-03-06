import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

def format_array(array):
    array = array.reshape(1, 784)
    array = array.astype('float32')
    array /= 255
    return array


def prepare_data() -> tuple:

    (x_training, y_training), (x_testing, y_testing) = mnist.load_data()  #load mnist dataset

    print(type(x_training))
    x_training = x_training.reshape(60000, 784)                   #reshape data from 28x28 matricies to 784 length arrays
    x_testing = x_testing.reshape(10000, 784)

    x_training = x_training.astype('float32')           #change types to floats
    x_testing = x_testing.astype('float32')

    x_training /= 255                   #normalize data from 0-255 to 0-1
    x_testing /= 255

    num_classes = 10

    y_training = np_utils.to_categorical(y_training, num_classes)
    y_testing = np_utils.to_categorical(y_testing, num_classes)

    return (x_training, x_testing, y_training, y_testing)

def create_model(summary=False):

    model = Sequential()

    #input and hidden layer 1
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))   #relu is a faster function than sigmoid and provides slightly more acuate results
    model.add(Dropout(0.2))

    #second hidden layer
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #output layer
    model.add(Dense(10))
    model.add(Activation('softmax'))        #softmax function normalizes output

    if summary is True: 
        model.summary

    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, data):
    x_training, y_training = data[0], data[2]
    model.fit(x_training, y_training, batch_size=128, epochs=10, verbose=1)

    return model

def evaluate_model(model, data):
    x_testing, y_testing = data[1], data[3]
    score = model.evaluate(x_testing, y_testing)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])

def visualize_output(model, data):
    x_testing, y_testing = data[1], data[3]
    for i in range(1):
        num = random.randint(0, 10000)
        correct_array = y_testing[num]
        guess_array = model.predict(x_testing[[num]])
        for j in range(10):
            if correct_array[j] == 1:
                correct = j 
                break

        max_digit = np.amax(guess_array)
        guess = np.where(guess_array == max_digit)
        guess = guess[1][0]


        fig = plt.imshow(x_testing[[num]].reshape(28,28), cmap='gray', interpolation='none')
        print(type(fig))
        print(fig)
        plt.show()

        print('Correct:', correct)
        print('Guess:', guess, '\n')



def predict(model, input):
    return model.predict(input)
