"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           dataManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os
import numpy as np

import commonEnumerations
import runInfo

import manager
import crossValidationFold

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class DataManager(manager.Manager):
    """
        DataManager stores important runtime information
    """

    __NAME = "DataManager"
    __MAX_NUM_CLASSES = 32

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,DataManager.__NAME)
        
        self._classDatabase = [None] * DataManager.__MAX_NUM_CLASSES
        self._classSet      = set()
        self._foldDatabase  = list([])

        self._runInfo           = runInfo.RunInfo(app)
        self._callbackInitFolds = DataManager.callbackInitTrainTestFolds

        if (self.getApp().crossValEnabled() == True):
            # Cross Validation is Enabled
            self._callbackInitFolds = DataManager.callbackInitCrossValFolds

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    def getRunInfo(self):
        """ Return Reference to RunInfo Structure """
        return self._runInfo

    def getClassName(self, classInt: int) -> str:
        """ Return the Name of the class based on the integer index """
        return self._classDatabase[classInt].className

    def getClassInt(self, className: str) -> int:
        """ Return the integer index of the class based on the name """
        for ii,item in enumerate(self._classDatabase):
            if (item.className == className):
                return ii
        # Not found:
        msg = "\tA class by the name of {0} was not found in the database".format(className)
        self.logMessage(msg)
        return -1

    def getNumClasses(self) -> int:
        """ Get the Number of classes in Use """
        if (len(self._classSet) == 0):
            # No classes yet!
            return 0
        return max(self._classSet)

    def getClassesInUse(self) -> list:
        """ Return a list of the class ints in use """
        return list(self._classSet)

    def getNumFolds(self) ->int:
        """ Return the number of cross validation folds in use """
        return len(self._foldDatabase)

    def getFold(self,index: int) -> crossValidationFold.CrossValidationFold:
        """ Get the Fold at the supplied index """
        if (index >= len(self._foldDatabase)):
            msg = "Fold index at {0} is out of range for {1} number of folds".format(
                index,len(self._foldDatabase))
            self.logMessage(msg)
            return None
        return self._foldDatabase[index]

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Initialize the Folds for Train/Test
        self.__initSampleFolds()

        # Populate Sample Databse 
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


        self.__exportRunInfo()
        self._setShutdownFinished(True)
        return self._status

    def registerFold(self, newFold) -> bool:
        """ Register Fold w/ Sample Manager """
        self._foldDatabase.append(newFold)
        return True

    def registerClassWithDatabase(self,classInt: int, className: str) -> bool:
        """ Register a class w/ the Class Database """
        if (classInt >= DataManager.__MAX_NUM_CLASSES):
            msg = "Class database w/ cap={0} cannot store info about class {1}".format(
                DataManager.__MAX_NUM_CLASSES,classInt)
            self.logMessage(msg)
            return False

        # Otherwise, store the info
        self._classSet.add(classInt)
        if (self._classDatabase[classInt] is None):
            self._classDatabase[classInt] = DataManager.ClassDataStruct(
                classInt,className)
            self._classDatabase[classInt].expectedCount = 1
        else:
            self._classDatabase[classInt].expectedCount += 1
        return True

    # Public Struct

    class ClassDataStruct:
        """ Structure to store info about each class """

        def __init__(self,
                     className: str,
                     classInt: int):
            """ Constructor """
            self.className      = className
            self.classInt       = classInt
            self.expectedCount  = 0
            self.classifierCount = 0
            self.segmenterCount  = 0

        def __del__(self):
            """ Destructor """
            pass

    # Private Interface 

    def __initSampleFolds(self) -> None:
        """ Construct the Order Queue for Each Fold """
        if (self._callbackInitFolds is None):
            msg = "Cannot initialize folds if _callbackInitFolds is set to None"
            self.logMessage(msg)
            raise RuntimeError(msg)
        # Invoke the Callback
        sampleMgr   = self.getApp().getSampleManager()
        dataMgr     = self
        self._callbackInitFolds.__call__(sampleMgr,dataMgr)  
        return None

    def __getIndexesForNextBatch(self,foldIndex: int) -> np.ndarray:
        """ Get the sample Index's for the next Batch of """
        batchSize = self.getApp().getConfig().getBatchSize()
        batchIndexes = self._folds[foldIndex].getNextBatch(batchSize)
        return batchIndexes

    def __plotClassDistrobution(self) -> None:
        """ Plot the Distrobution of classes in the data set """
        msg = "WARNING: Plotting distrobution of class data is not yet implemented"
        self.logMessage(msg)
        return None


    def __exportRunInfo(self) -> None:
        """ Export the runInfo class to disk """
        msg = "WARNING: Exporting RunInfo instance to dosl os not yet implemented"
        self.logMessage(msg)

        exportPath = os.path.join(self.getApp().getConfig().getOutputPath(),"runInfo.txt")
        self._runInfo.toDisk(exportPath)
        return None

    # Static Interface

    @staticmethod
    def callbackInitTrainTestFolds(sampleMgr,dataMgr) -> None:
        """ Register a Train/Test pair of folds (NON-X-Validation) """
        allSamples      = np.arange(sampleMgr.getSize(),dtype=np.int32)
        np.random.shuffle(allSamples)
        numTestSamples  = int(allSamples.size * sampleMgr.getApp().getConfig().getTestSplitRatio())
        numTrainSamples = int(allSamples.size - numTestSamples)
        # Make the Train Fold
        trainFold = crossValidationFold.CrossValidationFold(
            foldIndex=0,
            samplesInFold=allSamples[:numTrainSamples])
        dataMgr.registerFold(trainFold)
        # Make the Test Fold
        testFold = crossValidationFold.CrossValidationFold(
            foldIndex=1,
            samplesInFold=allSamples[numTrainSamples:])
        dataMgr.registerFold(testFold)
        return None

    @staticmethod
    def callbackInitCrossValFolds(sampleMgr,dataMgr) -> None:
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
            dataMgr.registerFold(fold)
            # Increment the start point
            foldStart += foldSize
        return None

"""
    Author:         Landon Buell
    Date:           May 2023
"""
