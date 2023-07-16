"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           ImageProcessingApplication.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os
import datetime
import numpy as np

import commonEnumerations

import appConfig
import textLogger

import strategyManager
import sampleManager
import dataManager
import augmentationManager
import preprocessManager
import tensorflowManager

        #### CLASS DEFINITIONS ####

class ImageProcessingApp:
    """
        Image Processing Application Instance
    """

    # Static Memebers
    __instance = None

    # Constructors

    def __init__(self, 
                 config: appConfig.AppConfig):
        """ Constructor """
        if (ImageProcessingApp.__instance is not None):
            msg = "Instance of {0} already exists!".format(self.__class__)
            raise RuntimeError(msg)
        ImageProcessingApp.__instance = self

        self._config        = config
        self._logger        = textLogger.TextLogger.fromConfig(appConfig)
        self._exitStatus    = commonEnumerations.Status.SUCCESS
        np.random.seed(config.getShuffleSeed()) # Set the numpy Random Seed

        self._strategyManager = strategyManager.StrategyManager(self)

        self._sampleManager = sampleManager.SampleManager(self)
        self._dataManager   = dataManager.DataManager(self)

        self._augmentationManager   = augmentationManager.AugmentationManager(self)
        self._preprocessManager     = preprocessManager.PreprocessManager(self)

        self._classificationManager = tensorflowManager.ClassificationManager(self)
        self._segmentationManager   = tensorflowManager.SegmentationManager(self)

    def __del__(self):
        """ Destructor """
        ImageProcessingApp.__instance = None

    @staticmethod
    def getInstance():
        """ Return or Generate the Singleton Instance """
        if (ImageProcessingApp.__instance is None):
            msg = "Instance of {0} does not exist - call ImageProcessingApp(<args>) to create it"
            raise RuntimeError(msg)
        return ImageProcessingApp.__instance

    # Accessors

    def getConfig(self) -> appConfig.AppConfig:
        """ Return the App's Configuration Structure """
        return self._config

    def getStatus(self) -> commonEnumerations.Status:
        """ Return the App's Status """
        return self._exitStatus

    def getCurrentFold(self) -> int:
        """ Get the Index of the current fold that we are working on """
        return self._currentFold

    def getStrategyManager(self) -> strategyManager.StrategyManager:
        """ Return the strategy Manager """
        return self._strategyManager
     
    def getSampleManager(self) -> sampleManager.SampleManager:
        """ Return the sample manager """
        return self._sampleManager

    def getDataManager(self) -> dataManager.DataManager:
        """ Return the Data Manager """
        return self._dataManager

    def getAugmentationManager(self):
        """ Return the Augmentation Manager """
        return self._augmentationManager

    def getPreprocessManager(self):
        """ Return the Prepreocessing Manager """
        return self._preprocessManager

    def getClassificationManager(self):
        """ Return the Classification Manager """
        return self._classificationManager

    def getSegmentationManager(self):
        """ Return the Segmentation Manager """
        return self._segmentationManager

    # Public Interface

    def logMessage(self,message: str) -> None:
        """ Log Message to Logger / Console """
        self._logger.logMessage(message)
        return None

    def startup(self) -> int:
        """ Run App Startup """
        self._strategyManager.init()

        self._sampleManager.init()
        self._dataManager.init()

        self._augmentationManager.init()
        self._preprocessManager.init()

        self._classificationManager.init()
        self._segmentationManager.init()

        return self._exitStatus

    def execute(self) -> int:
        """ Run App Execution """
        self._strategyManager.call()
        return self._exitStatus

    def shutdown(self) -> int:
        """ Run App shutdown """

        self._sampleManager.cleanup()
        self._dataManager.cleanup()

        self._augmentationManager.cleanup()
        self._preprocessManager.cleanup()

        self._classificationManager.cleanup()
        self._segmentationManager.cleanup()

        return self._exitStatus

    # Private Interface

    # Static Interface

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

"""
    Author:         Landon Buell
    Date:           May 2023
"""