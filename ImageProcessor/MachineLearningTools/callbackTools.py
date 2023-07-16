"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        MachineLearningTools
    Namespace:      N/A
    File:           callbacks.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf

import modelHistoryInfo


        #### FUNCTION DEFINITIONS ####

class TensorflowModelTrain(tf.keras.callbacks.Callback):
    """ Callback to run when fitting a Tensorflow Model """
    
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._trainHistory = modelHistoryInfo.ModelTrainHistoryInfo()

    def getTrainHistory(self):
        """ Return the trainign history instance """
        return self._trainHistory

    def on_train_batch_end(self,batch: int,logs: dict):
        """ When a batch is finished training """
        self._trainHistory.updateFromBatchLog(logs)
        return None


class TensorflowModelTest(tf.keras.callbacks.Callback):
    """ Callback to run when predicting on a Tensorflow Model """
    pass




"""
    Author:         Landon Buell
    Date:           May 2023
"""