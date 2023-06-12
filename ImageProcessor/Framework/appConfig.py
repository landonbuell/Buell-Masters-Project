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

        self._sampleDatabaseCapacity = 4096 # temp limit for development
        
        self._crossValidationFolds  = 1
        self._testSplitRatio        = 0.2

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

    def getIsSerialized(self) -> bool:
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

    def getSampleDatabaseCapacity(self) -> int:
        """ Return the intended capacity for the sample database """
        return self._sampleDatabaseCapacity

    def getNumCrossValFolds(self) -> int:
        """ Return the Number of folds for cross validation """
        return max(self._crossValidationFolds,1)

    def getTestSplitRatio(self) -> float:
        """ Ratio of the Test size to the full dataset """
        return self._testSplitRatio

    def getTrainSplitRatio(self) -> float:
        """ Ratio of the Train size to the full dataset """
        return (1.0 - self._testSplitRatio)

    # Public Interface

    # Private Interface

    # Static Interface

    @staticmethod
    def getDevelopmentConfig():
        """ Return Instace Designed for App Development """
        inputPaths = [os.path.abspath(os.path.join("..","..","inputFiles","labeledSamples.csv")),]
        outputPath = os.path.abspath(os.path.join("..","..","outputs","devRun0"))
        config = AppConfig(inputPaths,outputPath)
        return config

class Constants:
    """ Static Class of Constants """

    MAX_SAMPLE_DATA_BASE_SIZE   = int(1e5)
    MAX_RECURSION_DEPTH         = 4


"""
    Author:         Landon Buell
    Date:           May 2023
"""

