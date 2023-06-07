"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           classificationManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import commonEnumerations

import manager

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class ClassificationManager(manager.Manager):
    """
        ClassificationManager handles all image-classification related tasks
    """

    __NAME = "ClassificationManager"

    def __init__(self,
                 app):  # imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,ClassificationManager.__NAME)
        self._callbackGetModel  = None      # Callback that returns a "new" Classification Conv Neural Net
        self._model             = None  

    def __del__(self):
        """ Destructor """
        pass

    # Accessors


    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

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

        self._setShutdownFinished(True)
        return self._status

    def exportClassifierModel(self) -> None:
        """ Export the current classifier Model to the outputs folder """

        return None

    # Private Interface 

    def __getNewClassifierModel(self):
        """ Invoke the callback that returns a new classifier Model """
        if (self._callbackGetModel is None):
            msg = "No callback is defined to fetch a classifier neural network model"
            self.logMessage(msg)
            raise RuntimeError(msg)
        model = self._callbackGetModel.__call__()
        return model





"""
    Author:         Landon Buell
    Date:           May 2023
"""
