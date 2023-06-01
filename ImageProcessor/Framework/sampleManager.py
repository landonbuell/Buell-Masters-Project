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
        self.__loadSamplesFromInputFile()
        self.__shuffleAllSamplesInDatabase()

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

    def nextBatch(self,batchSizeOverride=None) -> np.ndarray:
        """ Return the Next Batch of Samples """
        batchSize = self.getApp().getConfig().getBatchSize()
        if (batchSizeOverride is not None):
            batchSize = int(batchSizeOverride)
        # Dequeue samples and return batch
        batch = np.array()
        while ((self._sampleDatabase.isEmpty() == False) and
               (len(batch) < batchSize)):
               # Get a sample from the database
               labeledSample = self._sampleDatabase.front()
               batch = np.append(batch,labeledSample)
               self._sampleDatabase.dequeue()
        return batch

    # Public Class

    class LabeledSample:
        """ Stores a Labeled Sample Instance """

        def __init__(self,
                     filePath: str,
                     classStr: str,
                     classInt: int):
            """ Constructor """
            self.filePath   = filePath
            self.classStr   = classStr
            self.classInt   = classInt

        def __del__(self):
            """ Destructor """
            pass

    # Private Interface 

    def __enqueueSample(self,labeledSample: LabeledSample) -> bool:
        """ Enqueue Sample to the database, Return T/F if it was successful """
        
        # TODO: Register class w/ Data Manager
        
        # Enqueue the Sample
        self._sampleDatabase.enqueue(labeledSample)
        return True

    def __loadSamplesFromInputFile(self) -> None:
        """ Load samples from an specified input file """
        inputFiles = self.getApp().getConfig().getInputPaths()
        for item in inputFiles:
            if (os.path.isfile(item) == False):
                msg = "File '{0}' does not exist. Skipping...".format(item)
                self.logMessage(msg)
                continue
            msg = "Loading samples from file: {0}".format(item)
            self.logMessage(msg)
            # Otherwise, read the file
            self.__readInputFile(item)
        return self

    def __readInputFile(self,inputFilePath: str) -> None:
        """ Read the Contents of an input file, and use it to enqueue new samples """
        inputFrame = pd.read_csv(inputFilePath,index_col=None)
        msg = "\tFound {0} samples in file".format(len(inputFrame))
        for ii,data in inputFrame.iterrows():
            if (self._sampleDatabase.isFull() == True):
                msg = "Sample Database is full. Not enqueing any new samples"
                self.logMessage(msg)
                return self
            # Make a new Sample Instance
            sample = SampleManager.LabeledSample(
                data.filePath,
                data.classStr,
                data.classInt)
            self.__enqueueSample(sample)
        return self


    def __shuffleAllSamplesInDatabase(self):
        """ Shuffle All samples in the database """
        seed = self.getApp().getConfig().getShuffleSeed()
        if (seed < 0):
            return self
        msg = "Shuffling samples w/ seed = {0}".format(seed)
        self._sampleDatabase.shuffle(seed)
        return self
"""
    Author:         Landon Buell
    Date:           May 2023
"""