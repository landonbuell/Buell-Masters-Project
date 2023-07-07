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
import matplotlib.pyplot as plt

import torch
import torchvision

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
        self._batchNormalizer = torch.nn.BatchNorm2d(
                                    num_features=3,
                                    eps=1e-6,
                                    momentum=0.1,
                                    affine=False,
                                    track_running_stats=True,
                                    device=self.getApp().getConfig().getTorchConfig().getActiveDevice())

        #self.__registerPreprocessStep( Preprocessors.showSampleAtIndex )
        self.__registerPreprocessStep( Preprocessors.crop8PixelsFromEdges )
        self.__registerPreprocessStep( Preprocessors.castToSinglePrecision )
        #self.__registerPreprocessStep( Preprocessors.rescaleTo32by32 )
        self.__registerPreprocessStep( Preprocessors.rescaleTo64by64 )

        self.__registerPreprocessStep( Preprocessors.divideBy255 )
        #self.__registerPreprocessStep( Preprocessors.torchVisionNormalize )
        #elf.__registerPreprocessStep( Preprocessors.showSampleAtIndex )


    def __del__(self):
        """ Destructor """
        pass

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

        # Populate Sample Databse 
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def processBatch(self,batch: batch.SampleBatch):
        """ Process a batch of Samples """
        # TODO: self._scaler.applyTransformation(batch)
        for ii,step in enumerate(self._steps):
             batch = step.__call__(self,batch)
        return batch

    # Private Interface 

    def __registerPreprocessStep(self,step) -> None:
        """ Register a Preprocessing Step with this manager """
        self._steps.append(step)
        return None

class Preprocessors:
    """ Static Class of Prepreocessors for batches of images """

    @staticmethod
    def showSampleAtIndex(  preprocessMgr: PreprocessManager,
                            sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Show the sample at the chosen index """
        sampleIndex = 0
        image,label = sampleBatch[sampleIndex]
        X = image.permute(1,2,0)
        X = X.to(device="cpu")
        if (X.dtype != torch.uint8):
            X = X.type(torch.uint8)
        plt.imshow(X)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Class Num: {0}".format(label))
        plt.show()
        return sampleBatch

    @staticmethod
    def crop8PixelsFromEdges(   preprocessMgr: PreprocessManager,
                                sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Crop 4 pixels from the edge of each image """
        numVerticalPixelsToRemove = 8
        numHorizontalPixelsToRemove = 8   
        initHeight  = sampleBatch.getX().shape[3]
        initWidth   = sampleBatch.getX().shape[2]
        finalHeight = initHeight - (2*numVerticalPixelsToRemove)
        finalWidth  = initWidth - (2*numHorizontalPixelsToRemove)
        # Apply Torch cropper
        sampleBatch.setX(   torchvision.transforms.functional.crop(
                                sampleBatch.getX(),
                                top=numVerticalPixelsToRemove,
                                left=numHorizontalPixelsToRemove,
                                height=finalHeight,width=finalWidth) )
        # New Image size if (3 x 184 x 184)
        return sampleBatch

    @staticmethod
    def castToSinglePrecision(  preprocessMgr: PreprocessManager,
                                sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Cast the features of an input tensor to Single-Precision floats """
        sampleBatch.setDataTypeX(torch.float32)
        return sampleBatch

    @staticmethod
    def rescaleTo32by32(preprocessMgr: PreprocessManager,
                        sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Resize each input image to 32 x 32 """
        resizer = torchvision.transforms.Resize(
            size=(32,32),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR ) 
        Xresized = resizer.forward(sampleBatch.getX())
        sampleBatch.setX( Xresized ) 
        return sampleBatch

    @staticmethod
    def rescaleTo64by64(preprocessMgr: PreprocessManager,
                        sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Resize each input image to 64 x 64 """
        resizer = torchvision.transforms.Resize(
            size=(64,64),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
           antialias=True) 
        Xresized = resizer.forward(sampleBatch.getX())
        sampleBatch.setX( Xresized ) 
        return sampleBatch

    @staticmethod
    def divideBy255(   preprocessMgr: PreprocessManager,
                            sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Divide each element in the Batch by 255 """
        sampleBatch._X = sampleBatch._X / 255.0
        return sampleBatch

    @staticmethod
    def torchVisionNormalize(   preprocessMgr: PreprocessManager,
                                sampleBatch: batch.SampleBatch) -> batch.SampleBatch:
        """ Scale input features to have unit variance and zero mean """
        return preprocessMgr.invokeBatchNormalizer(sampleBatch)

"""
    Author:         Landon Buell
    Date:           May 2023
"""
