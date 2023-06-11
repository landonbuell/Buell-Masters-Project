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


        #### CLASS DEFINTIONS ####

class Conv2dVGG16(torch.nn.Module):
    """ 2D Convolutional Neural Network inspired from the VGG-16 architecture """

    def __init__(self,
                 numClasses: int):
        """ Constructor """
        super().__init__()
        # 1st layer group
        self._conv1A  = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())      
        self._conv1B  = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._pool01 = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(1,1)))
        # 2nd Layer Group
        self._conv2A  = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())      
        self._conv2B  = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._pool02 = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(1,1)))
        # Flatten + Dense Layers
        self._flatten = torch.nn.Flatten()


    def __del__(self):
        """ Destructor """
        super().__del__()

    # Public Interface

    def forward(self,x):
        """ Module's forward-pass mechanism """
        y = x
        # 1st layer group
        y = self._conv1A(y)
        y = self._conv1B(y)
        y = self._pool01(y)
        # 2nd layer group
        y = self._conv2A(y)
        y = self._conv2B(y)
        y = self._pool02(y)


        return x


"""
    Author:         Landon Buell
    Date:           June 2023
"""