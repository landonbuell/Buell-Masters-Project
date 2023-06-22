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
import datetime
import numpy as np

import commonEnumerations


import appConfig
import textLogger
import batch

import sampleManager
import dataManager
import augmentationManager
import preprocessManager
import torchManager

        #### CLASS DEFINITIONS ####

class ImageProcessingApp:
    """
        Image Processing Application Instance
    """

    # Static Memebers
    __instance = None

    # Constructors

    def __init__(self, 
                 config: appConfig.AppConfig):
        """ Constructor """
        if (ImageProcessingApp.__instance is not None):
            msg = "Instance of {0} already exists!".format(self.__class__)
            raise RuntimeError(msg)
        ImageProcessingApp.__instance = self

        self._config        = config
        self._logger        = textLogger.TextLogger.fromConfig(appConfig)
        self._exitStatus    = commonEnumerations.Status.SUCCESS
        np.random.seed(config.getShuffleSeed()) # Set the numpy Random Seed

        self._sampleManager = sampleManager.SampleManager(self)
        self._dataManager   = dataManager.DataManager(self)

        self._augmentationManager   = augmentationManager.AugmentationManager(self)
        self._preprocessManager     = preprocessManager.PreprocessManager(self)

        self._classificationManager = torchManager.ClassificationManager(self)
        self._segmentationManager   = torchManager.SegmentationManager(self)

    def __del__(self):
        """ Destructor """
        ImageProcessingApp.__instance = None

    @staticmethod
    def getInstance():
        """ Return or Generate the Singleton Instance """
        if (ImageProcessingApp.__instance is None):
            msg = "Instance of {0} does not exist - call ImageProcessingApp(<args>) to create it"
            raise RuntimeError(msg)
        return ImageProcessingApp.__instance

    # Accessors

    def getConfig(self) -> appConfig.AppConfig:
        """ Return the App's Configuration Structure """
        return self._config

    def getStatus(self) -> commonEnumerations.Status:
        """ Return the App's Status """
        return self._exitStatus

    def getSampleManager(self) -> sampleManager.SampleManager:
        """ Return the sample manager """
        return self._sampleManager

    def getDataManager(self) -> dataManager.DataManager:
        """ Return the Data Manager """
        return self._dataManager

    def getAugmentationManager(self):
        """ Return the Augmentation Manager """
        return self._augmentationManager

    def getPreprocessManager(self):
        """ Return the Prepreocessing Manager """
        return self._preprocessManager

    def getClassificationManager(self):
        """ Return the Classification Manager """
        return self._classificationManager

    def getSegmentationManager(self):
        """ Return the Segmentation Manager """
        return self._segmentationManager

    def crossValEnabled(self) -> bool:
        """ Return T/F if Cross Validation Mode is enabled """
        return (self.getConfig().getNumCrossValFolds() > 1 )

    # Public Interface

    def logMessage(self,message: str) -> None:
        """ Log Message to Logger / Console """
        self._logger.logMessage(message)
        return None

    def startup(self) -> int:
        """ Run App Startup """
        self._sampleManager.init()
        self._dataManager.init()

        self._augmentationManager.init()
        self._preprocessManager.init()

        self._classificationManager.init()
        self._segmentationManager.init()

        return self._exitStatus

    def execute(self) -> int:
        """ Run App Execution """

        if (self.crossValEnabled() == True):
            # Cross-Validation is enabled
            numFolds = self.getConfig().getNumCrossValFolds()
            msg = "Cross Validation is enabled w/ {0} folds".format(numFolds)
            self.logMessage(msg)
            self.__runCrossValidation()
        else:
            # Basic Train-Test is enabled
            sizeTestRatio = self.getConfig().getTestSplitRatio() * 100.0
            msg = "Basic Train-Test is enabled w/ {0}% test ratio".format(sizeTestRatio)
            self.logMessage(msg)
            self.__runTrainTest()

        # Cleanup?

        return self._exitStatus

    def shutdown(self) -> int:
        """ Run App shutdown """

        return self._exitStatus

    # Private Interface

    def __runTrainTest(self):
        """ Run the app in Train-Test mode """

        # Train the Model on the 0-th Fold
        indexTrainFold = 0
        self.__runTrainOnFold(indexTrainFold)

        # Test the Model on the 1-th Fold
        indexTestFold = 1
        self.__runTestOnFold(indexTestFold)

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

                self.__runTrainOnFold(x,False)
           

            # Test in the remaining test fold
            msg = "\tTesting on Fold #{0}".format(testFold)
            self.logMessage(msg)
            self.__runTestOnFold(testFold)

            # Cleanup After Each Fold 
            # TODO: Export Classifier Model
            # TODO: Export Segmentation Model 
            batch.SampleBatch.resetBatchCounter()
        
        # Cleanup
        return None

    def __runTrainOnFold(self,
                         foldIndex: int,
                         resetBatchCounter=True):
        """ Run the Training Sequence on the chosen Fold """
        batchSize = self.getConfig().getBatchSize() 
        fold = self._dataManager.getFold(foldIndex)
        loop = (fold is not None)
        
        while (loop == True):

            batchIndexes    = fold.getNextBatchIndexes(batchSize)
            batchData       = self._sampleManager.getNextBatch(batchIndexes)

            # TODO: call preprocess manager on batch
            self._preprocessManager.processBatch(batchData)
            # TODO: call augmentation manager on batch 

            self._classificationManager.trainOnBatch(batchData)
            # TODO: invoke segmentation manager
            
            # Check if there is any samples left in this fold
            if (fold.isFinished() == True):
                loop = False
                fold.resetIterator()

        # Cleanup
        self._classificationManager.exportTrainingHistory(foldIndex)

        if (resetBatchCounter == True):
            batch.SampleBatch.resetBatchCounter()
        return None

    def __runTestOnFold(self,
                        foldIndex: int,
                        resetBatchCounter=True):
        """ Run the App in Test-only mode """
        batchSize = self.getConfig().getBatchSize() 
        fold = self._dataManager.getFold(foldIndex)
        loop = (fold is not None)
        
        while (loop == True):

            batchIndexes    = fold.getNextBatchIndexes(batchSize)
            batchData       = self._sampleManager.getNextBatch(batchIndexes)

            # TODO: call preprocess manager on batch
            self._preprocessManager.processBatch(batchData)

            self._classificationManager.testOnBatch(batchData)
            # TODO: invoke segmentation Manager
            
            # Check if there is any samples left in this fold
            if (fold.isFinished() == True):
                loop = False
                fold.resetIterator()

        # Cleanup
        if (resetBatchCounter == True):
            batch.SampleBatch.resetBatchCounter()
        return None

    # Private Callbacks

    # Static Interface

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

"""
    Author:         Landon Buell
    Date:           May 2023
"""