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
import tensorflow as tf

import commonEnumerations

import modelHistoryInfo
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
        self._randomSeed        = tf.random.set_seed(self.getApp().getConfig().getShuffleSeed())
        
        self._inputShape        = (64,64,3)
        self._numClasses        = self.getApp().getConfig().getNumClasses()

        self._callbackGetModel  = None
        self._model             = None

        self._optimizer         = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._objective         = tf.keras.losses.CategoricalCrossentropy()

        self._trainHistory      = modelHistoryInfo.ModelTrainHistoryInfo()
        self._testHistory       = modelHistoryInfo.ModelTestHistoryInfo()

        self._trainCallbacks    = [callbackTools.TensorflowModelTrain(),]
        self._testCallbacks     = [callbackTools.TensorflowModelTest(),]

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

    def exportTrainingHistory(self,outputFileName: str) -> None:
        """ Export the training history """
        outputPath = os.path.join(self.getOutputPath(),outputFileName)
        msg = "Exporting training history to: {0}".format(outputPath)
        self.logMessage(msg)
        self._trainHistory.export(outputPath)
        return None

    def exportTestingHistory(self,outputFileName: str) -> None:
        """ Export the training history """
        outputPath = os.path.join(self.getOutputPath(),outputFileName)
        msg = "Exporting testing history to: {0}".format(outputPath)
        self.logMessage(msg)
        self._testHistory.export(outputPath)
        return None

    def exportModel(self,outputPathName: str) -> None:
        """ Export the model to specified path """
        self.__verifyModelExists(True)
        outputPath = os.path.join(self.getOutputPath(),outputPathName)
        msg = "Exporting model to: {0}".format(outputPath)
        self.logMessage(msg)
        self._model.save(   outputPath,
                            overWrite=True,
                            save_format="h5")
        return None

    def resetState(self) -> None:
        """ Reset the Classifier Manager """
        self._model     = self.__invokeGetModel()
        self._initOptimizer()
        return None

    def loadModel(self,importPathName: str) -> None:
        """ Import the model from specified path """
        self.__verifyModelExists(True)
        importPath = os.path.join(self.getOutputPath(),importPathName)
        msg = "Importing model from: {0}".format(importPath)
        self.logMessage(msg)
        if (os.path.exists(importPath) == False):
            msg = "Cannot load model from {0} because it does not exist".format(importPath)
            self.logMessage(msg)
            raise RuntimeError(msg)
        self._model = tf.keras.models.load_model(importPathName)
        return None

    # Protected Interface

    def _initModel(self) -> None:
        """ VIRTUAL: Initialize the Model for this manager """
        self._model = self.__invokeGetModel()
        self._model.compile(optimizer=self._optimizer,
                            loss=self._objective,
                            metrics=self._metrics)
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
        model = self._callbackGetModel.__call__(self._inputShape,self._numClasses)
        return model

    def __trainOnBatchHelper(self,batchData: batch.SampleBatch) -> None:
        """ Helper Function to Train the model on the batch of data provided """
        X = batchData.getX()
        Y = batchData.getOneHotY(self._numClasses).type(tf.float32)
        self._model.fit(x=X,y=Y,
                        batch_size=batchData.getNumSamples(),
                        epochs=self.getConfig().getNumEpochsPerBatch(),
                        initial_epoch=self._epochCounter)
        self._epochCounter += self.getConfig().getNumEpochsPerBatch()
        return None
       

    def __testOnBatchHelper(self,batchData: batch.SampleBatch) -> None:
        """ Helper function to test the model n the batch of provided data """
        X = batchData.getX()
        Y = batchData.getY().type(tf.int16)
        self._model.predict(x=X,
                            batch_size=batchData.getNumSamples())
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
        self.registerGetModelCallback( tensorflowModels.getAffineModel )

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 

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