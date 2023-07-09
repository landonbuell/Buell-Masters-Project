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

        #### FUNCTION DEFINTIONS ####

def getAffineModel(numClasses: int):
    """ Callback does Nothing """
    return AffineModel(numClasses)

def getMultiTierImageClassifer(numClasses: int):
    """ Return an instance of a basic CNN """
    return MultiTierImageClassifer(numClasses)

    
        #### CLASS DEFINTIONS ####

class AffineModel(torch.nn.Module):
    """ Represents a Dummy (does nothing) Neural Network """

    def __init__(self,numClasses:int):
        """ constructor """
        super().__init__()
        self._numClasses = numClasses
        self.params = torch.nn.ParameterList([torch.nn.Linear(10,numClasses,True)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Define forward pass behavior for this model """
        outputs = torch.clone(inputs)
        for ii,item in self.params:
            outputs = item(outputs)
        return outputs

class MultiTierImageClassifer(torch.nn.Module):
    """ Multiple-Tiered Convolutional Nueral Network classifier """

    def __init__(self,numClasses: int):
        """ Constructor """
        super().__init__()
        self._numClasses = numClasses
        self._maxPool   = torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self._conv1A    = torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=(1,1))
        self._conv1B    = torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=(1,1))

        self._conv2A    = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1))
        self._conv2B    = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=(1,1))

        self._conv3A    = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1))
        self._conv3B    = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1))

        self._conv4A    = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1))
        self._conv4B    = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1))


    def __del__(self):
        """ Destructor """
        super().__del__()

    def forward(self,x: torch.tensor) -> torch.Tensor:
        """ Forward pass behavior """
        x   = torch.nn.functional.relu( self._conv1A(x) )
        x   = torch.nn.functional.relu( self._conv1B(x) )
        x   = self._maxPool(x)
        print(x.shape)

        x   = torch.nn.functional.relu( self._conv2A(x) )
        x   = torch.nn.functional.relu( self._conv2B(x) )
        x   = self._maxPool(x)
        print(x.shape)

        x   = torch.nn.functional.relu( self._conv3A(x) )
        x   = torch.nn.functional.relu( self._conv3B(x) )
        x   = self._maxPool(x)
        print(x.shape)

        x   = torch.nn.functional.relu( self._conv4A(x) )
        x   = torch.nn.functional.relu( self._conv4B(x) )
        x   = self._maxPool(x)
        print(x.shape)

        return x

"""
    Author:         Landon Buell
    Date:           June 2023
"""