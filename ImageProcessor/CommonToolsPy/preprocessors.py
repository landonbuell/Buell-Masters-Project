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

import numpy as np
import matplotlib.pyplot as plt

        #### CLASS DEFINITIONS ####

def showSampleAtIndex(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Show the sample at the chosen index """
    sampleIndex = 0
    image,label = sampleBatch[sampleIndex]
    X = image.permute(1,2,0)
    if (X.dtype != torch.uint8):
        X = X.type(torch.uint8)
    plt.imshow(X)
    plt.show()
    return sampleBatch

def replaceBordeBlueWithBlack(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Remove blue Border around every image & Replace it w/ a white one """
    X0 = sampleBatch.getX()[0]
    X1 = X0.permute(1,2,0)
    plt.imshow(X1)
    plt.show()

    X0[:,0:4,:]     = 0     # Mask the top side
    X0[:,198:200,:] = 0     # Mask the bottom side
    X0[:,:,0:3]     = 0     # Mask the left side
    X0[:,:,197:200]  = 0     # Mask the right side

    X2 = X0.permute(1,2,0)
    plt.imshow(X2)
    plt.show()
    return sampleBatch

def crop8PixelsFromEdges(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Crop 4 pixels from the edge of each image """
    numVerticalPixelsToRemove = 8
    numHorizontalPixelsToRemove = 8   
    initHeight  = sampleBatch.getX().shape[3]
    initWidth   = sampleBatch.getX().shape[2]
    finalHeight = initHeight - (2*numVerticalPixelsToRemove)
    finalWidth  = initWidth - (2*numHorizontalPixelsToRemove)
    # Apply Torch cropper
    sampleBatch._X    = torchvision.transforms.functional.crop(
                        sampleBatch.getX(),
                        top=numVerticalPixelsToRemove,
                        left=numHorizontalPixelsToRemove,
                        height=finalHeight,width=finalWidth)
    # New Image size if (3 x 184 x 184)
    return sampleBatch
    
def castToSinglePrecision(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Cast the features of an input tensor to Single-Precision floats """
    sampleBatch.setDataTypeX(torch.float32)
    return sampleBatch

def torchVisionNormalize(sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
    """ Scale input features to have unit variance and zero mean """
    means = torch.mean(sampleBatch.getX(),dim=0)
    stdds = torch.std(sampleBatch.getX(),dim=0)

    return sampleBatch

        #### PRIVATE HELPER FUNCTIONS ####

    

"""
    Author:         Landon Buell
    Date:           June 2023
"""