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

import commonEnumerations
import imageTools

import manager
import crossValidationFold
import batch

        #### FUNCTION DEFINITIONS ####


        #### CLASS DEFINTIONS ####

class SampleManager(manager.Manager):
    """
        SampleManager is a database of all input samples
    """

    __NAME = "SampleManager"
    

    def __init__(self,
                 app): #imageClassifierApp.ImageClassifierApp
        """ Constructor """
        super().__init__(app,SampleManager.__NAME)
        databaseCapacity        = int(1e5) # TEMP HARD-CODE TO FIX IMPORT ERRORS 
        
        self._sampleDatabase    = np.empty(shape=(databaseCapacity,),dtype=object)
        self._databaseSize      = 0
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getSize(self) -> int:
        """ Return the size of the database """
        return self._databaseSize

    def isFull(self) -> bool:
        """ Return T/F if the sample Database is full """
        return (self._databaseSize >= self._sampleDatabase.size - 1)

    def isEmpty(self) -> bool:
        """ Return T/F is the sample Database Is empty """
        return (self._databaseSize == 0)
 
    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Populate Sample Databse 
        self.__initSampleDatabase()
        
        
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def getNextBatch(self,listOfIndexes: list) -> batch.SampleBatch:
        """ Get a Batch rom a list of indexes """
        batchCounter = batch.SampleBatch.getBatchCounter()
        numSamplesInBatch = len(listOfIndexes)
        msg = "\t\tRetreving batch #{0} w/ {1} samples".format(batchCounter,numSamplesInBatch)
        self.logMessage(msg)

        # Create + Populate Sample Batch Structure
        batchData = batch.SampleBatch(numSamplesInBatch,batch.SHAPE_CHANNELS_FIRST)
        for ii,idx in enumerate(listOfIndexes):
            labeledSample = self[idx]
            X = imageTools.ImageIO.loadImageAsTorchTensor(labeledSample.filePath)
            y = labeledSample.classInt   
            batchData[ii] = (X,y)

        # Finished collecting batch - return
        return batchData

    def getSample(self,key: int):
        """ Get sample at specified int key """
        return self._sampleDatabase[key]

    # Public Class

    class LabeledSample:
        """ Stores a Labeled Sample Instance """

        def __init__(self,
                     filePath: str,
                     classInt: int,
                     classStr: str):
            """ Constructor """
            self.filePath   = filePath
            self.classInt   = classInt
            self.classStr   = classStr

        def __del__(self):
            """ Destructor """
            pass

        def __repr__(self) -> str:
            """ Debug representation of instance """
            return "Labeled Sample: {0} -> {1} @ {2}".format(self.classStr,self.classInt,hex(id(self)))

    # Private Interface 

    def __initSampleDatabase(self) -> None:
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
        msg = "Finished reading all input files. Sample Database has {0} items".format(
            self.getSize())
        self.logMessage(msg)
        return self

    def __readInputFile(self,inputFilePath: str) -> bool:
        """ Read the Contents of an input file, and use it to enqueue new samples """
        inputFrame = pd.read_csv(inputFilePath,index_col=None)
        msg = "\tFound {0} samples in file".format(len(inputFrame))
        self.logMessage(msg)

        for ii,data in inputFrame.iterrows():
            if (self.isFull() == True):
                msg = "Sample Database is full. Not enqueing any new samples"
                self.logMessage(msg)
                return False

            # Make a new Sample Instance & enqueue it
            sample = SampleManager.LabeledSample(
                data.filePath,
                data.classInt,
                data.classStr)
            self.__enqueueSample(sample)
            self.__registerWithDataManager(sample)
        # Finished w/ provided input file
        return True

    def __enqueueSample(self,labeledSample: LabeledSample) -> None:
        """ Add a Sample the the database, return T/F if successful """
        self._sampleDatabase[self._databaseSize] = labeledSample
        self._databaseSize += 1
        return None

    def __registerWithDataManager(self,labeledSample: LabeledSample) -> None:
        """ Register this class w/ Class Database """
        dataMgr = self.getApp().getDataManager()
        dataMgr.registerClassWithDatabase(
            labeledSample.classInt,
            labeledSample.classStr)
        return self

    # Magic Methods

    def __getitem__(self,key: int):
        """ Get sample at specified int key """
        return self._sampleDatabase[key]

    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "{0} w/ size = {1}".format(self.getName(),self.getSize())

    def __iter__(self):
        """ Iterate through the samples """
        for ii in range(self._databaseSize):
            yield self._sampleDatabase[ii]
        return None

"""
    Author:         Landon Buell
    Date:           May 2023
"""