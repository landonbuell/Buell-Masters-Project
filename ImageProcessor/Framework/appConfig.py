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

        self._logToConsole  = True
        self._logToFile     = True

        self._batchSize     = 128
        self._shuffleSeed   = 123456789

        self._sampleDatabaseCapacity = int(2**18) # temp limit for development
        self._numClasses            = 29 # temp hard-code for development
        
        self._crossValidationFolds  = 1
        self._testSplitRatio        = 0.2

        self._epochsPerBatch        = 1     # Number of consecutive times we see a bach
        self._epochsPerFold         = 2     # Number of time we train on a fold

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

    def getNumClasses(self) -> int:
        """ Get the Number of Classes in the dataset """
        return self._numClasses

    def getNumCrossValFolds(self) -> int:
        """ Return the Number of folds for cross validation """
        return max(self._crossValidationFolds,1)

    def getTestSplitRatio(self) -> float:
        """ Ratio of the Test size to the full dataset """
        return self._testSplitRatio

    def getTrainSplitRatio(self) -> float:
        """ Ratio of the Train size to the full dataset """
        return (1.0 - self._testSplitRatio)

    def getNumEpochsPerBatch(self) -> int:
        """ Return the number of times a model is trained on a batch in a row """
        return self._epochsPerBatch

    def getNumEpochsPerFold(self) -> int:
        """ Return the number of times a fold is seen by a model for training """
        return self._epochsPerFold

    # Public Interface

    def export(self):
        """ Export this instance to a text file """
        outputPath = os.path.join(self._pathOutput,"appConfig.txt")
        printFormat = lambda x,y : "{0:<32}{1}\n".format(x,y)
        outputStream = open(outputPath,"w")
        for key,val in self.__dict__.items():
            outputStream.write(printFormat(key,val))
        outputStream.close()
        return None

    # Private Interface

    # Static Interface

    @staticmethod
    def getDevelopmentConfig():
        """ Return Instace Designed for App Development """
        inputPaths = [os.path.abspath(os.path.join("..","..","inputFiles","every10Samples.csv")),]
        outputPath = os.path.abspath(os.path.join("..","..","outputs","multiTierClassifierV2"))
        config = AppConfig(inputPaths,outputPath)
        return config

"""
    Author:         Landon Buell
    Date:           May 2023
"""

