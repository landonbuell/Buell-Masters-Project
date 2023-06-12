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

import torch
import torchinfo

import manager
import batch

        #### CLASS DEFINITIONS ####

class TorchManager(manager.Manager):
    """
        Parent class to store & run w/ PyTorch Models & Operations
    """

    def __init__(self,
                 app,   #: imageProcessingApp.ImageProcessingApp,
                 name: str,
                 objective: torch.nn.Module,
                 optimizer: torch.nn.Module):
        """ Constructor """
        super().__init__(app,name)
        self._randomSeed        = torch.randint(0,999999,size=(1,))
        self._callbackGetModel  = None
        self._model             = None
        self._objective         = objective
        self._optimizer         = optimizer

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Generate the Model
        self._model = self.__invokeGetModel()

        self.__showModelInfo()

        # Populate Sample Databse 
        self._setInitFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    def trainOnBatch(self,batchData: batch.SampleBatch) -> None:
        """ Train the model on the batch of data provided """
        self.__verifyModelExists(True)



        return None

    def testOnBatch(self, batchData: batch.SampleBatch) -> None:
        """ Test the model on the batch of data provided """
        self.__verifyModelExists(True)

        return None

    def exportModel(self,modelName: str) -> bool:
        """ Export the current classifier Model to the outputs folder """
        self.__verifyModelExists()
        outPath = os.path.join(self.getApp().getConfig().getOutputPath(),modelName)
        torch.save(self._model,outPath)

        return False

    def resetState(self) -> None:
        """ Reset the Classifier Manager """
        self._model = self.__invokeGetModel()
        
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

    def __invokeGetModel(self) -> torch.nn.Module:
        """ Invoke the callback that returns a new classifier Model """
        if (self._callbackGetModel is None):
            msg = "No callback is defined to fetch a neural network model"
            self.logMessage(msg)
            raise RuntimeError(msg)
        model = self._callbackGetModel.__call__()
        return model

    def __predictOnBatch(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Execute a forward pass using the provided inputs """
        outputs = inputs
        return outputs


class ClassificationManager(manager.ModelManager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "ClassificationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,ClassificationManager.__NAME,
                         objective=torch.nn.CrossEntropyLoss(),
                         optimizer=torch.optim.Adam())


    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface 


"""
    Author:         Landon Buell
    Date:           June 2023
"""