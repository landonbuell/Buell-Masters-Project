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
import enum
        #### CLASS DEFINITIONS ####

class Status(enum.IntEnum):
        """ Stores Exist Status for Application """
        SUCCESS     = 0
        WARNING     = 1
        ERROR       = 2

class AppConfig:
    """ 
        Stores configuration Information for ImageProcessingApplication Instance
    """

    def __init__(self):
        """ Constructor """

        self._pathStartup   = os.getcwd()
        self._pathInputs    = set()
        self._pathOutput    = None
        self._isSerialized  = False

        self._logToConsole  = True
        self._logToFile     = True

        self._batchSize     = 128
        self._shuffleSeed   = 123456789

"""
    Author:         Landon Buell
    Date:           May 2023
"""

