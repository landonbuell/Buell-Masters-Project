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
import torch

import commonEnumerations

import manager
import batch

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

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
    Date:           May 2023
"""
