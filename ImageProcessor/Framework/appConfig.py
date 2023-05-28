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
import enum
        #### CLASS DEFINITIONS ####

class AppConfig:
    """ 
        Stores configuration Information for ImageProcessingApplication Instance
    """

    def __init__(self,
                 inputPaths: list,
                 outputPath: str):
        """ Constructor """

        self._pathStartup   = os.getcwd()
        self._pathInputs    = set(inputPaths)
        self._pathOutput    = outputPath
        self._isSerialized  = False

        self._logToConsole  = True
        self._logToFile     = True

        self._batchSize     = 128
        self._shuffleSeed   = 123456789
        
        self._enableCrossValidation     = False

        self._maxSampleDatabaseSize = int(2**16)

    # Accessors

    def getStartupPath(self) -> str:
        """ Return the app's startup path """
        return self._pathStartup

    def getInputPaths(self) -> list:
        """ Return a list of the app's input files """
        return list(self._pathInputs)

    def getOutputPath(self) -> str:
        """ Return the output Path """
        return self._pathOutput

    def getIsSerializer(self) -> bool:
        """ Return T/F is this instance has been serialized """
        return self._isSerialized

    def getLogToConsole(self) -> bool:
        """ Return T/F if messages should be logged to console """
        return self._logToConsole

    def getLogToFile(self) -> bool:
        """ Return T/F is messages should be logged to a File """
        return self._logToFile

    def getBatchSize(self) -> int:
        """ Return size of each batch """
        return self._batchSize

    def getShuffleSeed(self) -> int:
        """ Return the shuffle Seed """
        return self._shuffleSeed

    def getMaxSampleDatabseSize(self) -> int:
        """ Return the maximum allowed size for the SampleManager's Database """
        return self._maxSampleDatabaseSize

    # Public Interface

    # Private Interface

    # Static Interface

    @staticmethod
    def getDevelopmentConfig():
        """ Return Instace Designed for App Development """
        inputPaths = [os.path.abspath("..","..","dataset","asl_alphabet_train"),]
        outputPath = os.path.abspath("..","..","outputs","devRun0")
        config = AppConfig(inputPaths,outputPath)
        return config

"""
    Author:         Landon Buell
    Date:           May 2023
"""

