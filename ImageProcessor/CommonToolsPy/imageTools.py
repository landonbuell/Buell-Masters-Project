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
import tensorflow as tf

        #### CLASS DEFINITIONS ####

class ImageIO:
    """ Static class of method to load and save various image formats """

    def __init__(self):
        """ Dummy Constructor - Throws Error """
        msg = "{0} is a static class. Instances are not allowed".format(self.__class__)
        raise RuntimeError(msg)

    # Image Loaders

    def loadImageAsArray(imagePath: str) -> tf.Tensor:
        """ Load a JPG image as a tensorflow tensor """
        pilImage = tf.keras.utils.load_img(imagePath)
        npImage = tf.keras.utils.img_to_array(pilImage)
        return npImage




"""
    Author:         Landon Buell
    Date:           June 2023
"""