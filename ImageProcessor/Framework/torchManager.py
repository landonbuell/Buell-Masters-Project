"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           torchManager.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import os
import torch
from torch.optim.adam import Adam

import commonEnumerations

import convolutionalNeuralNetworks
import modelHistoryInfo

import manager
import batch

        #### CLASS DEFINITIONS ####

class TorchManager(manager.Manager):
    """
        Parent class to store & run w/ PyTorch Models & Operations
    """

    def __init__(self,
                 app,   #: imageProcessingApp.ImageProcessingApp,
                 name: str):
        """ Constructor """
        super().__init__(app,name)
        self._randomSeed        = torch.randint(0,999999,size=(1,))
        self._numClasses        = self.getApp().getConfig().getNumClasses()

        self._callbackGetModel  = None
        self._model             = None

        self._optimizer         = None
        self._objective         = torch.nn.CrossEntropyLoss()  

        self._trainHistory      = modelHistoryInfo.ModelHistoryInfo()
        self._evalHistory       = modelHistoryInfo.ModelHistoryInfo()
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getModel(self) -> torch.nn.Module:
        """ Get the active torch Model """
        return self._model

    def getObjective(self) -> torch.nn.Module:
        """ Return the objective function """
        return self._objective

    def getOptimizer(self) -> torch.nn.Module:
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
        self._initOptimizer()

        # Populate Sample Databse 
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
        self.__verifyOptimizerExists(True)
        
        self._model.train()
        self.__trainOnBatchHelper(batchData)
        return None

    def testOnBatch(self, batchData: batch.SampleBatch) -> None:
        """ Test the model on the batch of data provided """
        self.__verifyModelExists(True)
        return None

    def exportTrainingHistory(self,foldIndex: int) -> None:
        """ Show + Export Training History """
        outputFileName = "trainingHistoryFold{0}.csv".format(foldIndex)
        outputPath = os.path.join(self.getOutputPath(),outputFileName)
        self._trainHistory.export(outputPath)
        return None

    def exportModel(self,modelName: str) -> bool:
        """ Export the current classifier Model to the outputs folder """
        self.__verifyModelExists(True)
        outPath = os.path.join( self.getApp().getConfig().getOutputPath(), modelName ) 
        torch.save(self._model , outPath)

        return False

    def resetState(self) -> None:
        """ Reset the Classifier Manager """
        self._model     = self.__invokeGetModel()
        self._initOptimizer()
        return None

    # Protected Interface

    def _initModel(self) -> None:
        """ VIRTUAL: Initialize the Model for this manager """
        self._model = self.__invokeGetModel()
        toDevice = self.getApp().getConfig().getTorchConfig().getActiveDevice()
        self._model.to(device=toDevice)
        return None

    def _initOptimizer(self) -> None:
        """ VIRTUAL: Initialize the Optimizer for this Model """     
        self.__verifyModelExists()
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

    def __verifyOptimizerExists(self,throwErr=False) -> bool:
        """ Verify that the optimizer associated with this instance exists """
        if (self._optimizer is None):
            msg = "\t{0} does not contain a registered optimizer",format(repr(self))
            if (throwErr == True):
                raise RuntimeError(msg)
            return False
        return True

    def __invokeGetModel(self) -> torch.nn.Module:
        """ Invoke the callback that returns a new classifier Model """
        if (self._callbackGetModel is None):
            msg = "No callback is defined to fetch a neural network model"
            self.logMessage(msg)
            raise RuntimeError(msg)
        model = self._callbackGetModel.__call__(self._numClasses)
        return model

    def __trainOnBatchHelper(self,batchData: batch.SampleBatch) -> None:
        """ Helper Function to Train the model on the batch of data provided """
                # Isolate X + Y Data
        X = batchData.getX()
        Y = batchData.getOneHotY(self._numClasses).type(torch.float32)

        for epoch in range(self.getEpochsPerBatch()):
            self._optimizer.zero_grad()

            # Forward Pass + Compute cost of batch
            outputs = self._model(X)
            cost = self._objective(outputs,Y)

            # Backwards pass + update the weights          
            cost.backward()
            self._optimizer.step()

            # Update the weights + Log cost
            self._trainHistory.appendLossScore( cost.item() )

        return None

    def __predictOnBatch(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Execute a forward pass using the provided inputs """
        outputs = inputs
        return outputs

    # Static Interface


class ClassificationManager(TorchManager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "ClassificationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,ClassificationManager.__NAME)
        self.registerGetModelCallback( convolutionalNeuralNetworks.getInspiredVisualGeometryGroup )

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 

    # Protected Interface

    def _initOptimizer(self) -> None:
        """ VIRTUAL: Initialize the Optimizer for this Model """     
        super()._initOptimizer()
        self._optimizer = torch.optim.Adam(
                            params=self._model.parameters(),
                            lr=0.001,
                            betas=(0.9,0.999),
                            eps=1e-6)
        return None

class SegmentationManager(TorchManager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "SegmentationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,SegmentationManager.__NAME,)
        self.registerGetModelCallback( convolutionalNeuralNetworks.getAffineModel )

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 



"""
    Author:         Landon Buell
    Date:           June 2023
"""