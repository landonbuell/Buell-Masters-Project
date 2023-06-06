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
import matplotlib.pyplot as plt
import matplotlib.image as img

        #### CLASS DEFINITIONS ####

class ImageIO:
    """ Static class of method to load and save various image formats """

    def __init__(self):
        """ Dummy Constructor - Throws Error """
        msg = "{0} is a static class. Instances are not allowed".format(self.__class__)
        raise RuntimeError(msg)

    # Image Loaders

    def loadImage(imagePath: str):
        """ Load a JPG image as a torch tensor """
        imageData = img.imread(imagePath)
        plt.imshow(imageData)
        return imageData





"""
    Author:         Landon Buell
    Date:           June 2023
"""