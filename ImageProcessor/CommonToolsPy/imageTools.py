"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           imageIO.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import os

import torch
import torchvision


        #### CLASS DEFINITIONS ####

class ImageIO:
    """ Static class of method to load and save various image formats """

    def __init__(self):
        """ Dummy Constructor - Throws Error """
        msg = "{0} is a static class. Instances are not allowed".format(self.__class__)
        raise RuntimeError(msg)

    # Image Loaders

    def loadImageAsTorchTensor(imagePath: str) -> torch.Tensor:
        """ Load a JPG image as a torch tensor """
        torchImage = torchvision.io.read_image(imagePath)
        return torchImage




"""
    Author:         Landon Buell
    Date:           June 2023
"""