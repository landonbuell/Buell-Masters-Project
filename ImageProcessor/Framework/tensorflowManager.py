"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           tensorflowManager.py
    Author:         Landon Buell
    Date:           July 2023
"""

        #### IMPORTS ####

import os
import numpy as np
import tensorflow as tf

import commonEnumerations

import modelHistoryInfo
import classificationReport
import callbackTools
import tensorflowModels

import manager
import batch

        #### CLASS DEFINITIONS ####

        
class TensorflowManager(manager.Manager):
    """
        Parent class to store & run w/ Tensorflow Models & Operations
    """

    def __init__(self,
                 app,   #: imageProcessingApp.ImageProcessingApp,
                 name: str):
        """ Constructor """
        super().__init__(app,name)
        self._inputShape        = (64,64,3)
        self._numClasses        = self.getApp().getConfig().getNumClasses()

        self._callbackGetModel  = None
        self._model             = None
         
        self._optimizer         = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self._objective         = tf.keras.losses.CategoricalCrossentropy()

        self._trainHistory      = modelHistoryInfo.ModelTrainHistoryInfo()
        self._evalHistory       = modelHistoryInfo.ModelTestHistoryInfo(self._numClasses)

        self._trainCallbacks    = callbackTools.TensorflowModelTrain(self)
        self._testCallbacks     = callbackTools.TensorflowModelTest(self)

        self._epochCounter      = 0
        self._metrics           = [tf.keras.metrics.Accuracy(),
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),]
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getModel(self) -> tf.keras.Sequential:
        """ Get the active torch Model """
        return self._model

    def getObjective(self) -> tf.keras.losses.Loss:
        """ Return the objective function """
        return self._objective

    def getOptimizer(self) -> tf.keras.optimizers.Optimizer:
        """ Return the optimizer strategy """
        return self._optimizer

    def getEpochsPerBatch(self) -> int:
        """ Return the Number of epochs to use per batch """
        return self.getApp().getConfig().getNumEpochsPerBatch()

    def getTrainingHistory(self):
        """ Return the Training History Instance """
        return self._trainHistory

    def getEvaluationHistory(self):
        """ Return the Evaluation History Instance """
        return self._evalHistory

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status
        self._initModel()
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status
        self._setShutdownFinished(True)
        return self._status

    def registerGetModelCallback(self,callback) -> None:
        """ Register the callback that returns a pytorch model """
        self._callbackGetModel = callback
        return None

    def trainOnBatch(self,batchData: batch.SampleBatch) -> None:
        """ Train the model on the batch of data provided """
        self.__verifyModelExists(True)
        self.__trainOnBatchHelper(batchData)
        return None

    def testOnBatch(self, batchData: batch.SampleBatch) -> None:
        """ Test the model on the batch of data provided """
        self.__verifyModelExists(True)
        self.__testOnBatchHelper(batchData)
        return None

    def exportTrainingHistory(self,fullOutputPath: str) -> None:
        """ Export the training history """
        msg = "Exporting training history to: {0}".format(fullOutputPath)
        self.logMessage(msg)
        self.getTrainingHistory().export(fullOutputPath)
        return None

    def exportTestingHistory(self,fullOutputPath: str) -> None:
        """ Export the training history """
        msg = "Exporting testing history to: {0}".format(fullOutputPath)
        self.logMessage(msg)
        self.getEvaluationHistory().export(fullOutputPath)
        return None

    def exportModelToHdf5(self,fullOutputPath: str) -> None:
        """ Export the model to specified path """
        self.__verifyModelExists(True)
        msg = "Exporting model to: {0}".format(fullOutputPath)
        self.logMessage(msg)
        self._model.save(fullOutputPath,overwrite=True,save_format="h5")
        return None

    def loadModel(self,fullInputPath: str) -> None:
        """ Import the model from specified path """
        self.__verifyModelExists(True)
        msg = "Importing model from: {0}".format(fullInputPath)
        self.logMessage(msg)
        if (os.path.exists(fullInputPath) == False):
            msg = "Cannot load model from {0} because it does not exist".format(fullInputPath)
            self.logMessage(msg)
            raise RuntimeError(msg)
        self._model = tf.keras.models.load_model(fullInputPath)
        self._model.compile(optimizer=self._optimizer,
                            loss=self._objective,
                            metrics=self._metrics,
                            steps_per_execution=1)
        return None

    def resetState(self) -> None:
        """ Reset the Classifier Manager """
        self._initModel()
        return None

    def getModelNameForFold(self,foldIndex: int) -> str:
        """ get the Model name given the current foldIndex """
        return "{0}Fold{1}".format(self._name,foldIndex)


    # Protected Interface

    def _initModel(self) -> None:
        """ VIRTUAL: Initialize the Model for this manager """
        self._model = self.__invokeGetModel()
        self._model.compile(optimizer=self._optimizer,
                            loss=self._objective,
                            metrics=self._metrics,
                            steps_per_execution=1)
        #self._model.summary(line_length=72,print_fn=self.logMessage)
        return None

    # Private Interface 

    def __verifyModelExists(self,throwErr=False) -> bool:
        """ Verify that the model associated with this instance exists """
        if (self._model is None):
            msg = "\t{0} does not contain a registered model",format(repr(self))
            if (throwErr == True):
                raise RuntimeError(msg)
            return False
        return True

    def __invokeGetModel(self) -> tf.keras.Sequential:
        """ Invoke the callback that returns a new classifier Model """
        if (self._callbackGetModel is None):
            msg = "No callback is defined to fetch a neural network model"
            self.logMessage(msg)
            raise RuntimeError(msg)
        currentFold = self.getApp().getStrategyManager().getCurrentFold()
        modelName = self.getModelNameForFold(currentFold)
        model = self._callbackGetModel.__call__(self._inputShape,self._numClasses,modelName)
        return model

    def __trainOnBatchHelper(self,batchData: batch.SampleBatch) -> None:
        """ Helper Function to Train the model on the batch of data provided """
        features = batchData.getX() 
        labels = batchData.getOneHotY(self._numClasses).astype(np.float32)
        batchLog = self._model.fit(x=features,y=labels,
                        batch_size=self.getConfig().getBatchSize(),
                        epochs=self.getConfig().getNumEpochsPerBatch(),
                        initial_epoch=0,
                        callbacks=self._trainCallbacks)
        self._epochCounter += self.getConfig().getNumEpochsPerBatch()
        return None
       
    def __testOnBatchHelper(self,batchData: batch.SampleBatch) -> None:
        """ Helper function to test the model n the batch of provided data """
        features = batchData.getX()
        labels = batchData.getY()
        predictions = self._model.predict(x=features,
                        batch_size=self.getConfig().getBatchSize())
        # Store the Predictions & Truths
        self._evalHistory.updateFromPredictions(labels,predictions)
        return None

class ClassificationManager(TensorflowManager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "ClassificationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,ClassificationManager.__NAME)
        self.registerGetModelCallback( tensorflowModels.getMultiTierImageClassifier )

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 

    def exportClassificationReport(self,outputPath: str) -> None:
        """ Export a classification report based on evaluation history """
        classNames = self.getApp().getDataManager().getClassNames()
        report = classificationReport.ClassificationReport(classNames)
        report.update( self._evalHistory )
        report.export(outputPath)
        return None

    # Protected Interface


class SegmentationManager(TensorflowManager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "SegmentationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,SegmentationManager.__NAME,)
        self.registerGetModelCallback( tensorflowModels.getAffineModel )

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 

"""
    Author:         Landon Buell
    Date:           June 2023
"""