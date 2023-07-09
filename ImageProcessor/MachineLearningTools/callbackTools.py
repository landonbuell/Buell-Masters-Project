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

DEVICE_CPU  = "cpu"

        #### FUNCTION DEFINITIONS ####

def multiclassPrecisionScore(preds: np.ndarray,
                             truth: np.ndarray,
                             numClasses: int):
    """ Compute & Return the precision score for each class """
    result  = np.empty(shape=(numClasses,),dtype=np.float32)
    preds   = np.argmax(preds,axis=-1)
    truth   = np.argmax(truth,axis=-1)

    # Iterate through the classes
    for ii in range(numClasses):
        classPreds = (preds == ii)
        classTruth = (truth == ii)
        result[ii] = metrics.precision_score(classTruth,classPreds,zero_division=0)
    return result

def multiclassRecallScore(  preds: np.ndarray,
                            truth: np.ndarray,
                            numClasses: int):
    """ Compute & Return the recall score for each class """
    result  = np.empty(shape=(numClasses,),dtype=np.float32)
    preds   = np.argmax(preds,axis=-1)
    truth   = np.argmax(truth,axis=-1)

    # Iterate through the classes
    for ii in range(numClasses):
        classPreds = (preds == ii)
        classTruth = (truth == ii)
        result[ii] = metrics.recall_score(classTruth,classPreds,zero_division=0)
    return result







"""
    Author:         Landon Buell
    Date:           May 2023
"""