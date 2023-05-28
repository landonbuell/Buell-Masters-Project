"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           commonEnumerations.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import enum

        #### CLASS DEFINITIONS ####

class Status(enum.IntEnum):
    """ Stores Exist Status for Application """
    SUCCESS     = 0
    WARNING     = 1
    ERROR       = 2


"""
    Author:         Landon Buell
    Date:           May 2023
"""