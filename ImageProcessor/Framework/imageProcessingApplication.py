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

        self._sampleManager = None
        self._dataManager   = None

        self._augmentationManager   = None
        self._preprocessManager     = None

        self._classificationManager = None
        self._segmentationManager   = None

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

    def getStatus(self) -> appConfig.Status:
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

        return None

    def startup(self) -> int:
        """ Run App Startup """

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


"""
    Author:         Landon Buell
    Date:           May 2023
"""