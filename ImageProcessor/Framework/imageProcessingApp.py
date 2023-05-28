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

import appConfig
import textLogger

import sampleManager
import dataManager
import augmentationManager
import preprocessManager
import classificationManager
import segmentationManager

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
        self._exitStatus    = appConfig.Status.SUCCESS

        self._sampleManager = sampleManager.SampleManager(self)
        self._dataManager   = dataManager.DataManager(self)

        self._augmentationManager   = augmentationManager.AugmentationManager(self)
        self._preprocessManager     = preprocessManager.PreprocessManager(self)

        self._classificationManager = classificationManager.ClassificationManager(self)
        self._segmentationManager   = segmentationManager.SegmentationManager(self)

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

    def getStatus(self) -> Status:
        """ Return the App's Status """
        return self._exitStatus

    def getSampleManager(self):
        """ Return the sample manager """
        return self._sampleManager

    def getDataManager(self):
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
        self._sampleManager.init()
        self._dataManager.init()

        self._augmentationManager.init()
        self._preprocessManager.init()

        self._classificationManager.init()
        self._preprocessManager.init()

        return self._exitStatus

    def execute(self) -> int:
        """ Run App Execution """

        return self._exitStatus

    def shutdown(self) -> int:
        """ Run App shutdown """

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

class Status(enum.IntEnum):
    """ Stores Exist Status for Application """
    SUCCESS     = 0
    WARNING     = 1
    ERROR       = 2


"""
    Author:         Landon Buell
    Date:           May 2023
"""