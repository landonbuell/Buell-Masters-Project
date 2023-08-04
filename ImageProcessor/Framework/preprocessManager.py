"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           preprocessManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import commonEnumerations

import manager
import batch

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class PreprocessManager(manager.Manager):
    """
        PreprocessManager preprocesses a given sample batch of samples
    """

    __NAME = "PreprocessManager"

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,PreprocessManager.__NAME)
        self._steps     = list()
        self._displayInitialImage   = False
        self._displayAfterEachStep  = False

        self.__registerPreprocessStep( Preprocessors.crop8PixelsFromEdges )
        #self.__registerPreprocessStep( Preprocessors.rescaleTo32by32 )
        self.__registerPreprocessStep( Preprocessors.rescaleTo64by64 )
        self.__registerPreprocessStep( Preprocessors.divideBy255 )
        self.__registerPreprocessStep( Preprocessors.tensorflowNormalize )

    def __del__(self):
        """ Destructor """
        self._steps.clear()

    # Accessors

    def invokeBatchNormalizer(self, sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Return the batch normalization layer """
        normX = self._batchNormalizer(sampleBatch.getX())
        sampleBatch.setX( normX )
        return sampleBatch

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def processBatch(self,sampleBatch: batch.SampleBatch):
        """ Process a batch of Samples """
        if (self._displayInitialImage == True):
            self.__showSampleAtIndex(sampleBatch)
        for ii,step in enumerate(self._steps):
             sampleBatch = step.__call__(self,sampleBatch)
             if (self._displayAfterEachStep == True):
                self.__showSampleAtIndex(sampleBatch)
        return sampleBatch

    # Private Interface 

    def __registerPreprocessStep(self,step) -> None:
        """ Register a Preprocessing Step with this manager """
        self._steps.append(step)
        return None

    def __showSampleAtIndex(self,sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Show the sample at the chosen index """
        sampleIndex = 0
        image,label = sampleBatch[sampleIndex]
        #X = image.permute(1,2,0)
        if (image.dtype != np.uint8):
            image = image.astype(np.uint8)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Class Num: {0}".format(label))
        plt.show()
        return sampleBatch

class Preprocessors:
    """ Static class of preprocessors for batches of images """

    @staticmethod
    def crop8PixelsFromEdges(   preprocessMgr: PreprocessManager,
                                sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Crop 4 pixels from the edge of each image """
        numVerticalPixelsToRemove = 8
        numHorizontalPixelsToRemove = 8 
        initWidth   = sampleBatch.getX().shape[1]
        initHeight  = sampleBatch.getX().shape[2]
        finalpixelWidth  = initWidth - (2 * numHorizontalPixelsToRemove) + numHorizontalPixelsToRemove
        finalpixelHeight = initWidth - (2 * numVerticalPixelsToRemove) + numVerticalPixelsToRemove
        # Crop + Save
        newX = sampleBatch.getX()[:,numHorizontalPixelsToRemove:finalpixelWidth,numVerticalPixelsToRemove:finalpixelHeight,:]
        sampleBatch.setX(newX)
        # New Image size if (184 x 184 x 3)
        return sampleBatch

    @staticmethod
    def rescaleTo32by32(preprocessMgr: PreprocessManager,
                        sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Resize each input image to 32 x 32 """
        Xresized = tf.image.resize(
            sampleBatch.getX(),
            size=(32,32),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False)
        Xresized = Xresized.numpy()
        sampleBatch.setX( Xresized ) 
        return sampleBatch

    @staticmethod
    def rescaleTo64by64(preprocessMgr: PreprocessManager,
                        sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Resize each input image to 64 x 64 """
        Xresized = tf.image.resize(
            sampleBatch.getX(),
            size=(64,64),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False)
        Xresized = Xresized.numpy()
        sampleBatch.setX( Xresized ) 
        return sampleBatch

    @staticmethod
    def rescaleTo128by128(preprocessMgr: PreprocessManager,
                        sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Resize each input image to 64 x 64 """
        Xresized = tf.image.resize(
            sampleBatch.getX(),
            size=(128,128),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False)
        Xresized = Xresized.numpy()
        sampleBatch.setX( Xresized ) 
        return sampleBatch

    @staticmethod
    def divideBy255(preprocessMgr: PreprocessManager,
                    sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Divide each element in the Batch by 255 """
        sampleBatch._X = sampleBatch._X / 255.0
        return sampleBatch
    
    @staticmethod
    def tensorflowNormalize(preprocessMgr: PreprocessManager,
                            sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Apply Standard Scaling to each image in a batch """
        Xscaled = tf.image.per_image_standardization(sampleBatch.getX())
        Xscaled = Xscaled.numpy()
        sampleBatch.setX(Xscaled)
        return sampleBatch

"""
    Author:         Landon Buell
    Date:           May 2023
"""
