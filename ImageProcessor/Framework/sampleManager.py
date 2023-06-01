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

import manager
import crossValidationFold

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
        databaseCapacity        = int(1e5) # TEMP HARD-CODE TO FIX IMPORT ERRORS     
        self._sampleDatabase    = np.empty(shape=(databaseCapacity,),dtype=object)
        self._databaseSize      = 0
        self._folds             = list([])
        self._callbackInitFolds = SampleManager.callbackInitTrainTestFolds

        if (self.getApp().crossValEnabled() == True):
            # Cross Validation is Enabled
            self._callbackInitFolds = SampleManager.callbackInitCrossValFolds

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

    def getNumFolds(self) ->int:
        """ Return the number of cross validation folds in use """
        return len(self._folds)

    def getFold(self,index: int):
        """ Get the Fold at the supplied index """
        return self._folds[index]
        
    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Populate Sample Databse 
        self.__initSampleDatabase()
        self.__initSampleFolds()
        
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def registerFold(self, newFold) -> bool:
        """ Register Fold w/ Sample Manager """
        self._folds.append(newFold)
        return True
        

    def nextBatch(overrideBatchSize=None):
        """ Get the next batch of samples """
        # TODO: Implement this
        return None

    def getSample(self,key: int):
        """ Get sample at specified int key """
        return self._sampleDatabase[key]

    # Public Class

    class LabeledSample:
        """ Stores a Labeled Sample Instance """

        def __init__(self,
                     filePath: str,
                     classInt: int):
            """ Constructor """
            self.filePath   = filePath
            self.classInt   = classInt

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
        msg = "Finished reading all input file. Sample Database has {0} items".format(
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
                data.classInt)
            self.__enqueueSample(sample)
            self.__registerWithClassDatabase(data.classStr,data.classInt)

        # Finished w/ provided input file
        return True

    def __enqueueSample(self,sample: LabeledSample) -> None:
        """ Add a Sample the the database, return T/F if successful """
        self._sampleDatabase[self._databaseSize] = sample
        self._databaseSize += 1
        return None

    def __registerWithClassDatabase(self,classStr: str, classInt: int) -> None:
        """ Register Class Int/Str pair with ClassDatabase in DataManager """
        # TODO: Implement this w/ Data Manager
        return None

    def __initSampleFolds(self) -> None:
        """ Construct the Order Queue for Each Fold """
        if (self._callbackInitFolds is None):
            msg = "Cannot initialize folds if _callbackInitFolds is set to None"
            self.logMessage(msg)
            raise RuntimeError(msg)
        # Invoke the Callback
        self._callbackInitFolds.__call__(self)  
        return None

    # Magic Methods

    def __getitem__(self,key: int):
        """ Get sample at specified int key """
        return self._sampleDatabase[key]

    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "{0} w/ size = {1}".format(self.getName(),self.getSize())

    # Static Interface

    @staticmethod
    def callbackInitTrainTestFolds(sampleMgr) -> None:
        """ Register a Train/Test pair of folds (NON-X-Validation) """
        allSamples      = np.arange(sampleMgr.getSize(),dtype=np.int32)
        np.random.shuffle(allSamples)
        numTestSamples  = int(allSamples.size * sampleMgr.getApp().getConfig().getTestSplitRatio())
        numTrainSamples = int(allSamples.size - numTestSamples)
        # Make the Train Fold
        trainFold = crossValidationFold.CrossValidationFold(
            foldIndex=0,
            samplesInFold=allSamples[:numTrainSamples])
        sampleMgr.registerFold(trainFold)
        # Make the Test Fold
        testFold = crossValidationFold.CrossValidationFold(
            foldIndex=1,
            samplesInFold=allSamples[numTrainSamples:])
        sampleMgr.registerFold(testFold)
        return None

    @staticmethod
    def callbackInitCrossValFolds(sampleMgr) -> None:
        """ Register a set of Cross Validation Folds """
        allSamples  = np.arange(sampleMgr.getSize(),dtype=np.int32)
        np.random.shuffle(allSamples)
        numFolds    = sampleMgr.getApp().getConfig().getNumCrossValFolds()
        foldSize    = int(allSamples.size / numFolds)
        foldStart   = 0
        for foldIdx in range(numFolds):
            # Create the Fold + Register it
            foldSamples = allSamples[foldStart:foldStart + foldSize]
            fold = crossValidationFold.CrossValidationFold(
                foldIndex=foldIdx,
                samplesInFold=foldSamples)
            sampleMgr.registerFold(fold)
            # Increment the start point
            foldStart += foldSize
        return None


"""
    Author:         Landon Buell
    Date:           May 2023
"""