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
import numpy as np
import pandas as pd

import simpleQueue
import commonEnumerations

import manager

        #### FUNCTION DEFINITIONS ####


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

        self._sampleDatabase    = simpleQueue.SimpleQueue()
        self._classDatabase     = dict()

        self.__loadSamplesFromInputFile()

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def isFull(self) -> bool:
        """ Return T/F if the sample database is full """
        return self._sampleDatabase.isFull()

    def isEmpty(self) -> bool:
        """ Return T/F is the sample database is empty """
        return self._sampleDatabase.isEmpty()

    def getSize(self) -> int:
        """ Return the number of items in the sample database """
        return self._sampleDatabase.getSize()

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

    # Private Class

    class __LabeledSample:
        """ Stores a Labeled Sample Instance """

        def __init__(self,
                     filePath: str,
                     classIndex: int):
            """ Constructor """
            self.filePath   = filePath
            self.classIndex = classIndex

        def __del__(self):
            """ Destructor """
            pass

    # Private Interface 

    def __enqueueSample(self,pathToFile: str) -> None:
        """ Enqueue Sample to the database, Return T/F if it was successful """
        

        return None

    def __loadSamplesFromInputFile(self) -> None:
        """ Load samples from an specified input file """
        inputFiles = self.getApp().getConfig().getInputPaths()
        for item in inputFiles:
            if (os.path.isfile(item) == False):
                msg = "File '{0}' does not exist. Skipping..."
                self.logMessage(msg)
            # Otherwise, read the file
            self.__readInputFile(item)

    def __readInputFile(self,inputFilePath: str) -> None:
        """ Read the Contents of an input file, and use it to enqueue new samples """
        inputFrame = pd.read_csv(inputFilePath,index_col=None)
        for ii,data in inputFrame.iterrows():
            sample = SampleManager.__LabeledSample(
                data.filePath,data.classInt)


        return self


"""
    Author:         Landon Buell
    Date:           May 2023
"""