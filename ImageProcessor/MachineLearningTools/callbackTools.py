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

import torch
import sklearn.metrics as metrics

DEVICE_CPU  = "cpu"

        #### FUNCTION DEFINITIONS ####

def multiclassPrecisionScore(preds: torch.Tensor,
                             truth: torch.Tensor,
                             numClasses: int):
    """ Compute & Return the precision score for each class """
    result = torch.empty(size=(numClasses,),dtype=torch.float32,device=DEVICE_CPU)
    preds = torch.argmax(preds,dim=0).to(device=DEVICE_CPU)
    truth = torch.argmax(truth,dim=0).to(device=DEVICE_CPU)
    # Iterate through the classes
    for ii in range(numClasses):
        classPreds = (preds == ii)
        classTruth = (truth == ii)
        result[ii] = metrics.precision_score(classTruth,classPreds)
    return result

def multiclassRecallScore(preds: torch.Tensor,
                             truth: torch.Tensor,
                             numClasses: int):
    """ Compute & Return the recall score for each class """
    result = torch.empty(size=(numClasses,),dtype=torch.float32,device=DEVICE_CPU)
    preds = torch.argmax(preds,dim=0).to(device=DEVICE_CPU)
    truth = torch.argmax(truth,dim=0).to(device=DEVICE_CPU)
    # Iterate through the classes
    for ii in range(numClasses):
        classPreds = (preds == ii)
        classTruth = (truth == ii)
        result[ii] = metrics.recall_score(classTruth,classPreds)
    return result







"""
    Author:         Landon Buell
    Date:           May 2023
"""