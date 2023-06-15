"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           processors.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import batch

import torch
import torchvision

        #### CLASS DEFINITIONS ####

def castToSinglePrecision(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Cast the features of an input tensor to Single-Precision floats """
    sampleBatch.setDataTypeX(torch.float32)
    return sampleBatch

def torchVisionScaler(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Scale input features to have unit variance and zero mean """
    means = torch.mean(sampleBatch.getX(),dim=0)
    stdds = torch.std(sampleBatch.getX(),dim=0)


    return sampleBatch

    

"""
    Author:         Landon Buell
    Date:           June 2023
"""