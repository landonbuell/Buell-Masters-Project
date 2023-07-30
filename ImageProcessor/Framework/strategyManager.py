"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           preprocessManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import numpy as np

import commonEnumerations
import batch

import manager


        #### CLASS DEFINITIONS ####

class StrategyManager(manager.Manager):
    """ Governs the App's Current Execution Strategy """

    __NAME = "StrategyManager"

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,StrategyManager.__NAME)
        self._currentFold = 0


    def __del__(self):
        """ Destructor """
        super().__del__()

    # Accessors

    def crossValEnabled(self) -> bool:
        """ Return T/F if Cross Validation Mode is enabled """
        return (self.getApp().getConfig().getNumCrossValFolds() > 1 )

    def getCurrentFold(self) -> int:
        """ Return the Current Fold In Use """
        return self._currentFold

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        if (self.crossValEnabled() == True):
            # Cross-Validation is enabled
            numFolds = self.getApp().getConfig().getNumCrossValFolds()
            msg = "Cross Validation is enabled w/ {0} folds".format(numFolds)
        else:
            # Basic Train-Test is enabled
            sizeTestRatio = self.getApp().getConfig().getTestSplitRatio() * 100.0
            msg = "Basic Train-Test is enabled w/ {0}% test ratio".format(sizeTestRatio)
        self.logMessage(msg)

        self._setInitFinished(True)
        return self._status

    def call(self) -> commonEnumerations.Status:
        """ Run the appropriate Strategy """
        
        if (self.crossValEnabled() == True):
            # Cross-Validation is enabled
            msg = "Running Cross Validation"
            self.logMessage(msg)
            self.__runCrossValidation()
        else:
            # Basic Train-Test is enabled
            msg = "Running Train/Test Split"
            self.logMessage(msg)
            self.__runTrainTest()

        return self._status


    # Private Interface

    def __runTrainTest(self):
        """ Run the app in Train-Test mode """

        # Train the Model on the 0-th Fold
        indexTrainFold = 0
        self._currentFold = indexTrainFold
        for ii in range(self.getConfig().getNumEpochsPerFold()):
            self.__runTrainOnFold(indexTrainFold,False)
            self.__exportClassificiationTrainHistoryAndModel(indexTrainFold)
        self.__resetClassificationAndLoadModel(indexTrainFold)
        
        # Test the Model on the 1-th Fold
        indexTestFold = 1
        self._currentFold = indexTestFold     
        self.__runTestOnFold(indexTestFold)
        self.__exportClassificiationTestHistory(indexTestFold)
        return None

    def __runCrossValidation(self):
        """ Run the app in Cross Validation Mode """
        numFolds = self.getConfig().getNumCrossValFolds()
        foldIndexes = np.arange(numFolds)

        for foldIndex in foldIndexes:
            msg = "Performing Cross Validation on Fold #{0}".format(foldIndex)
            self.logMessage(msg)

            testFold    = foldIndex
            trainFolds  = np.delete(foldIndexes,testFold)

            # Train on each of the training Folds
            for x in trainFolds:
                msg = "\tTraining on Fold #{0}".format(x)
                self.logMessage(msg)
           
                for ii in range(self.getConfig().getNumEpochsPerFold()):
                    self.__runTrainOnFold(x,False)
                self.__exportClassificiationTrainHistoryAndModel(foldIndex)

            # Test in the remaining test fold
            self.__resetClassificationAndLoadModel(foldIndex)
            msg = "\tTesting on Fold #{0}".format(testFold)
            self.logMessage(msg)
            
            self.__runTestOnFold(testFold)
            self.__exportClassificiationTestHistory(testFold)
            # Cleanup After Each Fold 
            batch.SampleBatch.resetBatchCounter()
        
        # Cleanup
        return None

    
    def __runTrainOnFold(self,
                         foldIndex: int,
                         resetBatchCounter=True):
        """ Run the Training Sequence on the chosen Fold """
        batchSize = self.getConfig().getBatchSize() 
        fold = self.getApp().getDataManager().getFold(foldIndex)
        loop = (fold is not None)
        
        while (loop == True):

            batchIndexes    = fold.getNextBatchIndexes(batchSize)
            batchData       = self.getApp().getSampleManager().getNextBatch(batchIndexes)

            self.getApp().getPreprocessManager().processBatch(batchData)
            # TODO: call augmentation manager on batch 

            self.getApp().getClassificationManager().trainOnBatch(batchData)
            # TODO: invoke segmentation manager
            
            # Check if there is any samples left in this fold
            if (fold.isFinished() == True):
                loop = False
                fold.resetIterator()
                fold.shuffle()

        # Cleanup
        if (resetBatchCounter == True):
            batch.SampleBatch.resetBatchCounter()          
        return None

    def __runTestOnFold(self,
                        foldIndex: int,
                        resetBatchCounter=True):
        """ Run the App in Test-only mode """
        batchSize = self.getConfig().getBatchSize() 
        fold = self.getApp().getDataManager().getFold(foldIndex)
        loop = (fold is not None)
        
        while (loop == True):

            batchIndexes    = fold.getNextBatchIndexes(batchSize)
            batchData       = self.getApp().getSampleManager().getNextBatch(batchIndexes)

            # TODO: call preprocess manager on batch
            self.getApp().getPreprocessManager().processBatch(batchData)

            self.getApp().getClassificationManager().testOnBatch(batchData)
            # TODO: invoke segmentation Manager
            
            # Check if there is any samples left in this fold
            if (fold.isFinished() == True):
                loop = False
                fold.resetIterator()
                fold.shuffle()

        # Cleanup
        if (resetBatchCounter == True):
            batch.SampleBatch.resetBatchCounter()
        return None

    def __exportClassificiationTrainHistoryAndModel(self,foldIndex: int) -> None:
        """ Export the Training History & Model using the fold index as the name """
        self.getApp().getClassificationManager().exportTrainingHistory(
                "trainingHistoryFold{0}.csv".format(foldIndex))
        self.getApp().getClassificationManager().exportModel(
                "modelFold{0}.hdf5".format(foldIndex))
        return None

    def __exportClassificiationTestHistory(self,foldIndex: int) -> None:
        """ Export the Testing History & Model using the fold index as the name """
        self.getApp().getClassificationManager().exportTestingHistory(
                "testingHistoryFold{0}.csv".format(foldIndex))
        self.getApp().getClassificationManager().exportClassificationReport(
            self.getConfig().getOutputPath(),foldIndex)
        return None

    def __resetClassificationAndLoadModel(self,foldIndex: int) -> None:
        """ Reset the state of the manager + reload the model """
        self.getApp().getClassificationManager().resetState()
        self.getApp().getClassificationManager().loadModel(
                "modelFold{0}.hdf5".format(foldIndex))
        return None


"""
    Author:         Landon Buell
    Date:           May 2023
"""
