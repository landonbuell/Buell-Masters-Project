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

def getAffineModel(inputShape:tuple,numClasses: int,modelName:int):
    """ Return a basic Single-Layer Perceptron """
    model = tf.keras.Sequential(name=modelName)
    return model

def getSingleTierImageClassifier(inputShape:tuple,numClasses:int,modelName: str):
    """ Return a Single-Tiered Convolutional Nueral Network """
    model = tf.keras.Sequential(name=modelName)
    layers = [  tf.keras.Input(shape=inputShape,dtype=tf.float32),
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),activation='relu'),  
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),activation='relu'),  
                tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256,activation='relu'),
                tf.keras.layers.Dense(units=128,activation='relu'),
                tf.keras.layers.Dense(units=64,activation='relu'),
                tf.keras.layers.Dense(units=numClasses,activation='softmax'),
            ]
    for ii,layer in enumerate(layers):
        layerName = "{0}_layer{1}".format(modelName,ii)
        layer._name = layerName
        model.add(layer)
    return model

def getMultiTierImageClassifier(inputShape:tuple,numClasses:int,modelName:int):
    """ Return a Multi-Tiered Convolutional Nueral Network """
    model = tf.keras.Sequential(name=modelName)
    layers = [  tf.keras.Input(shape=inputShape,dtype=tf.float32),
                # 1st layer group
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)),
                # 2nd layer group
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)),
                # 3rd layer group
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)),
                # 4th layer group
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu'), 
                tf.keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2)),
                # Flatten + Dense
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256,activation='relu'),
                tf.keras.layers.Dense(units=128,activation='relu'),
                tf.keras.layers.Dense(units=64,activation='relu'),
                tf.keras.layers.Dense(units=numClasses,activation='softmax'),
            ]
    for ii,layer in enumerate(layers):
        layerName = "{0}_layer{1}".format(modelName,ii)
        layer._name = layerName
        model.add(layer)
    return model

"""
    Author:         Landon Buell
    Date:           June 2023
"""