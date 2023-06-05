"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           runInfo.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import os
import numpy as np

        #### CLASS DEFINITIONS ####

class RunInfo:
    """ Stores important information for runtime """

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        self._startupPath   = app.getConfig().getStartupPath()
        self._inputPaths    = app.getConfig().getInputPaths()
        self._outputPath    = app.getConfig().getOutputPath()

        self._shuffleSeed   = app.getConfig().getShuffleSeed()
        self._numFolds      = app.getConfig().getNumFolds()

        self._numClasses    = 0


    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface

    def toDisk(self,
               exportPath: str):
        """ Export this instance to file """
        return self



"""
    Author:         Landon Buell
    Date:           June 2023
"""
