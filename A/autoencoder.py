#!/usr/bin/env python3.8

import sys
from user_input import user_input
import keras
from keras import layers, optimizers, losses, metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
from encoder_decoder import decoder, encoder
from graphs import printGraphs
from Reading import readMNIST


argc = len(sys.argv)

if argc != 3:
    print("wrong arguments")
    exit(1)

filename = sys.argv[2]   # pairnei to orisma tou arxeiou pou dothike kai to dinei sto filename

print("Filename: {0} ".format(filename))

images_collection = readMNIST(filename)  # kaleitai i readmnist tou reading.py gia to sygkrkrimeno filename

#  akolouthei to splitting tou arxeiou eikonwn se training set kai validation set
trainingSetArr, evaluationSetArr, train_ground, valid_ground = train_test_split(images_collection, images_collection, test_size=0.3, random_state=15)

print("evaluation set")
print(len(evaluationSetArr))
print("training set")
print(len(trainingSetArr))

# ftiaxnoume to input_img gia tis diastaseis twn eikonwn mas
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

# dimiourgoume pinakes gia kathe hyperparameter kai gia to total loss kai to total val loss
# wste na kratisoume tis times gia na ftiaksoume tis grafikes meta apo polles ekteleseis
numberOfFilters = []
numberOfLayers = []
filterSize_dim = []
numberOfEpochs = []
batchSize = []
learningRate = []
total_loss = []
total_val_loss = []
totloss = []
total_val_loss = []


while True:

    # dexomaste timi gia kathe yperparametro apo to xristi xrisimopoiwntas to user_input() ap to user_input.py

    print("Enter the number of filters: ")
    numberOfFilters.append(user_input())   # eisagoume kathe timi ston antistoixo pinaka yperparametrou
    print(numberOfFilters)
    print("Enter the number of hidden layers: ")
    numberOfLayers.append(user_input())
    print(numberOfLayers)
    print("Enter the filter size dimension: ")
    filterSize_dim.append(user_input())
    print(filterSize_dim)   # dexomaste san input to megethos tis mias pleuras enos tetragwnou
    print("Enter the number of epochs: ")
    numberOfEpochs.append(user_input())
    print(numberOfEpochs)
    print("Enter the batch size: ")
    batchSize.append(user_input())
    print(batchSize)
    print("Enter the learning rate: ")
    learningRate.append(user_input())
    print(learningRate)

    # xrisimopoioume tin teleutaia timi pou dextikame, diladi to teleutaio stoixeio tou pinaka kathe parametrou
    # gi auto xrisimopoioume to [-1] stoixeio kathe pinaka, diladi to teleutaio
    filterSize = (filterSize_dim[-1], filterSize_dim[-1])  # thewroume tetragwni ti diastasi, opote einai pleura x pleura

    useDropout = True  # energopoihmeni xrisi dropout

    # dimiourgia montelou
    autoencoder = Model(input_img, decoder(encoder(input_img, filterSize, numberOfLayers[-1], numberOfFilters[-1], useDropout), filterSize, numberOfLayers[-1], numberOfFilters[-1]))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate = learningRate[-1]))

    # training montelou
    autoencoder_train = autoencoder.fit(trainingSetArr, train_ground, batch_size=batchSize[-1], epochs=numberOfEpochs[-1], verbose=1, validation_data=(evaluationSetArr, valid_ground))

    # kratame tin timi tou loss kai val_loss tis teleutaias epoxis k tis vazoume stous antistoixous pinakes
    totloss = autoencoder_train.history['loss']
    total_loss.append(totloss[-1])
    totval_loss = (autoencoder_train.history['val_loss'])
    total_val_loss.append(totval_loss[-1])

    # vgazoume tis treis epiloges gia to xristi
    print("You can choose to: \n1) Repeat the experiment with different hyperparameters\n")
    print("2) Display error graphs based on the current hyperparameteres  of the executed experiments\n")
    print("3) Save the current model, trained with the current hyperparameter values ")
    option = input("Enter 1, 2, or 3: ")

    # dexomaste ton arithmo epilogis pali xrisimopoiwntas to user_input
    while (option != "1") and (option != "2") and (option != "3"):
        print("Please enter 1,2, or 3")
        option = str(user_input())
    if option == "1":
        continue
    if option == "2":  # emfanisi grafikwn
        printGraphs(total_loss, total_val_loss, numberOfFilters, numberOfLayers, numberOfEpochs, filterSize_dim, filterSize, batchSize, learningRate)
        break
    if option == "3":
        # saving to disk
        autoencoder.save("my_h5_model.h5")  # h "my_h5_model" an thelw to fakelo me ta arxeia
        print("Model saved successfully !")
        break









