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

class Status(enum.IntEnum):
        """ Stores Exist Status for Application """
        SUCCESS     = 0
        WARNING     = 1
        ERROR       = 2

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

    # Public Interface

    # Private Interface

    # Static Interface





"""
    Author:         Landon Buell
    Date:           May 2023
"""

