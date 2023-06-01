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
    __MAX_NUM_FOLDS = 10

    def __init__(self,
                 app): #imageClassifierApp.ImageClassifierApp
        """ Constructor """
        super().__init__(app,SampleManager.__NAME)
        databaseCapacity = self.getApp().getConfig().getMaxSampleDatabaseSize()
        self._sampleDatabase    = np.empty(shape=(databaseCapacity,),dtype=object)
        self._orderQueue        = simpleQueue.SimpleQueue(databaseCapacity)
        self._sizeDatabaseSize  = 0
        

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getSize(self) -> int:
        """ Return the size of the database """
        return self._size

    def getNumUnreadSamples(self) -> int:
        """ Get the number of samples that have not been sent off """
        return self._orderQueue.size()

    def getNumReadSamples(self) -> int:
        """ Get the number of samples that have been sent off """
        return (self._size - self._orderQueue.getSize())

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Populate Sample Databse 
        self.__loadSamplesFromInputFile()
        self.__buildOrderQueue()

        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def nextBatch(overrideBatchSize=None):
        """ Get the next batch of samples """
        # TODO: Implement this
        return None

    # Public Class

    class LabeledSample:
        """ Stores a Labeled Sample Instance """

        def __init__(self,
                     filePath: str,
                     classInt: int):
            """ Constructor """
            self.filePath   = filePath
            self.classStr   = classStr
            self.classInt   = classInt

        def __del__(self):
            """ Destructor """
            pass

        def __repr__(self) -> str:
            """ Debug representation of instance """
            return "Labeled Sample: {0} -> {1} @ {2}".format(self.classStr,self.classInt,hex(id(self)))

    # Private Interface 

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
        # All Done
        msg = "Finished reading all input file. Sample Database has {0} items".format(self._sampleDatabase.getSize())
        self.logMessage(msg)
        return self

    def __readInputFile(self,inputFilePath: str) -> None:
        """ Read the Contents of an input file, and use it to enqueue new samples """
        inputFrame = pd.read_csv(inputFilePath,index_col=None)
        msg = "\tFound {0} samples in file".format(len(inputFrame))
        self.logMessage(msg)

        for ii,data in inputFrame.iterrows():
            if (self._sampleDatabase.isFull() == True):
                msg = "Sample Database is full. Not enqueing any new samples"
                self.logMessage(msg)
                return self

            # Make a new Sample Instance
            sample = SampleManager.LabeledSample(
                data.filePath,
                data.classInt)

            self.__enqueueSample(sample)
        return self

    def buildOrderQueue(self):
        """ Build the Queue of sample indexes to use """


"""
    Author:         Landon Buell
    Date:           May 2023
"""