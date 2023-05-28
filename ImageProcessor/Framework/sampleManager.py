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
import queue

import manager
import imageProcessingApp
import appConfig

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class SampleManager(manager.Manager):
    """
        SampleManager is a database of all input samples
    """

    __NAME = "SampleManager"

    def __init__(self,
                 app: imageProcessingApp.ImageProcessingApp):
        """ Constructor """
        super().__init__(app,SampleManager.__NAME)

        self._database = queue.Queue(app.getConfig().getMaxSampleDatabseSize())


    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def init(self) -> appConfig.Status:
        """ Initialize this Manager """
        if (super().init() == appConfig.Status.ERROR):
            return self._status

        # Populate Sample Databse 
        self._setInitFinished(True)
        return self._status

    def call(self) -> appConfig.Status:
        """ Run this manager """
        if (super().call() == appConfig.Status.ERROR):
            return self._status

        self._setExecuteFinished(True)
        return self._status

    def cleanup(self) -> appConfig.Status:
        """ Cleanup this manager """
        if (super().cleanup() == appConfig.Status.ERROR):
            return self._status

        self._setShutdownFinished(True)
        return self._status

    # Private Interface 


"""
    Author:         Landon Buell
    Date:           May 2023
"""