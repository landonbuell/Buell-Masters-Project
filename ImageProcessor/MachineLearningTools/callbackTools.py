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
    
    def __init__(self,tensorflowManager):
        """ Constructor """
        super().__init__()
        self._tfMgr = tensorflowManager

    def on_train_batch_end(self,batch: int,logs: dict):
        """ When a batch is finished training """
        self._tfMgr.getTrainingHistory().updateFromBatchLog(logs)
        return None

class TensorflowModelTest(tf.keras.callbacks.Callback):
    """ Callback to run when fitting a Tensorflow Model """
    
    def __init__(self,tensorflowManager):
        """ Constructor """
        super().__init__()
        self._tfMgr = tensorflowManager

    def on_predict_batch_end(self,batch:int,logs: dict):
        """ When a batch is finished predictions """
        self._tfMgr.getEvaluationHistory().updateFromBatchLog(logs)
        return None

    


"""
    Author:         Landon Buell
    Date:           May 2023
"""