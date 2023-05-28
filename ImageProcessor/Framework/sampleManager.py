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
        if (super().getStatus() == appConfig.Status.ERROR):
            return self._status

        # Populate Sample Databse 

        return self._status

    def loadNextBatch(self) -> list:
        """ Return the next batch """ 
        return []

    # Private Interface 


"""
    Author:         Landon Buell
    Date:           May 2023
"""