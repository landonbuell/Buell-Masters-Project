"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           augmentationManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import commonEnumerations

import manager

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class AugmentationManager(manager.Manager):
    """
        AugmentationManager augments an existing dataset through a chosen strategy
    """

    __NAME = "AugmentationManager"

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,AugmentationManager.__NAME)

    def __del__(self):
        """ Destructor """
        pass

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

    # Private Interface 


"""
    Author:         Landon Buell
    Date:           May 2023
"""

