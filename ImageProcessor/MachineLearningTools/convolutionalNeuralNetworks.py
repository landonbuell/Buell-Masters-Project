"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Classification
    Namespace:      N/A
    File:           convolutionalNeuralNetworks.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import torch

        #### FUNCTION DEFINITIONS ####

class Conv2dVGG16(torch.nn.Module):
    """ 2D Convolutional Neural Network inspired from the VGG-16 architecture """

    def __init__(self):
        """ Constructor """
        super().__init__()
        self._conv1A  = torch.nn.Conv2d()


    def __del__(self):
        """ Destructor """
        super().__del__()

    # Public Interface

    def forward(self,x):
        """ Module's forward-pass mechanism """



        return x


"""
    Author:         Landon Buell
    Date:           June 2023
"""