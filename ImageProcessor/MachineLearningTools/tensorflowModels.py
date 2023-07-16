"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Classification
    Namespace:      N/A
    File:           convolutionalNeuralNetworks.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import tensorflow as tf


        #### FUNCTION DEFINTIONS ####

def getAffineModel(inputShape:tuple,numClasses: int):
    """ Return a basic Single-Layer Perceptron """
    model = tf.keras.Sequential()
    return model

def getMultiTieredConvolutional2DModel(inputShape:tuple,numClasses:int):
    """ Return a Multi-Tiered Convolutional Nueral Network """
    model = tf.keras.Sequential()
    model.add( tf.keras.layers.InputLayer(input_shape=inputShape,dtype=tf.float32) )
    # 1st Layer Group
    model.add( tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)) )
    # 2nd Layer Group
    model.add( tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(1,1)) )
    # 3rd Layer Group
    model.add( tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu') )
    model.add( tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(1,1)) )
    # Flatten + Perceptron
    model.add( tf.keras.layers.Flatten() )
    model.add( tf.keras.layers.Dense(units=128,activation='relu') )
    model.add( tf.keras.layers.Dense(units=128,activation='relu') )
    model.add( tf.keras.layers.Dense(units=64,activation='relu') )
    model.add( tf.keras.layers.Dense(units=numClasses,activation='softmax') )
    return model

"""
    Author:         Landon Buell
    Date:           June 2023
"""