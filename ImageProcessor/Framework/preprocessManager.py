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

import commonEnumerations
import preprocessors

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
        
        self.__registerPreprocessStep( preprocessors.castToSinglePrecision )
        self.__registerPreprocessStep( preprocessors.torchVisionScaler )

    def __del__(self):
        """ Destructor """
        pass

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
             batch = step.__call__(batch)
        return batch

    # Private Interface 

    def __registerPreprocessStep(self,step) -> None:
        """ Register a Preprocessing Step with this manager """
        self._steps.append(step)
        return None

    def __fitStandardScalerParams(self) -> None:
        """ Fit a standard scaler to the current dataset in groups """
        #self._scaler.fitToDatabase(self.getApp().getSampleManager())
        return None

    def __loadStandardScalerParams(self) -> None:
        """ Load params for a standard scaler from disk """
        return None

"""
    Author:         Landon Buell
    Date:           May 2023
"""
