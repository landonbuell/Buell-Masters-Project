"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           sampleManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import simpleQueue
import commonEnumerations

import manager

        #### FUNCTION DEFINITIONS ####

def defaultFileToSample(itemPath: str) -> None:
    """ Turns a directory path item to an image sample """


    return None
        #### CLASS DEFINTIONS ####

class SampleManager(manager.Manager):
    """
        SampleManager is a database of all input samples
    """

    __NAME = "SampleManager"
    __ACCEPTED_EXTENSIONS = ["png","jpg","jpeg"]

    def __init__(self,
                 app): #imageClassifierApp.ImageClassifierApp
        """ Constructor """
        super().__init__(app,SampleManager.__NAME)

        self._database = simpleQueue.SimpleQueue()
        self._callbackFileToSample = None   # Callback to turn a path into a Sample Instance

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def isFull(self) -> bool:
        """ Return T/F if the sample database is full """
        return self._database.isFull()

    def isEmpty(self) -> bool:
        """ Return T/F is the sample database is empty """
        return self._database.isEmpty()

    def getSize(self) -> int:
        """ Return the number of items in the sample database """
        return self._database.getSize()

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Populate Sample Databse 
        self.__populateSampleDatabase()


        self._setInitFinished(True)
        return self._status

    def call(self) -> commonEnumerations.Status:
        """ Run this manager """
        if (super().call() == commonEnumerations.Status.ERROR):
            return self._status

        self._setExecuteFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def getNextBatch(self):
        """ Return the Next Batch of Samples """
        # TODO: IMPLEMENT THIS
        return None

    # Private Interface 

    def __enqueueSample(self,pathToFile: str) -> None:
        """ Enqueue Sample to the database, Return T/F if it was successful """
        

        return None

    def __populateSampleDatabase(self) -> None:
        """ Populate the Sample Database """
        datasetPaths = self.getConfig().getInputPaths()     #list
        for path in datasetPaths:
            self.__enqueueSamplesInpath(path,currentDepth=0)

        return None

    def __enqueueSamplesInpath(self,
                               pathToSearch: str,
                               currentDepth: int) -> None:
        """ Find Samples (recursively) in path, and enqueue them to this database """
        contents = os.listdir(pathToSearch)
        for item in contents:
            fullPathToItem = os.path.join(item,pathToSearch)
            if (os.path.isdir(fullPathToItem) == True):
                # Is a directory
                if (currentDepth >= self.getConfig().getMaxSampleSearchDepth()):
                    # Maximum depth attained - skip this directory
                    continue
                self.__enqueueSamplesInpath(fullPathToItem,currentDepth + 1)
            elif (os.path.isfile(fullPathToItem) == True):
                ext = fullPathToItem.split(".")[-1]
                if ((ext in SampleManager.__ACCEPTED_EXTENSIONS) == False):
                    # Not an image that we can work with
                    continue
                self.__enqueueSample(fullPathToItem)

"""
    Author:         Landon Buell
    Date:           May 2023
"""