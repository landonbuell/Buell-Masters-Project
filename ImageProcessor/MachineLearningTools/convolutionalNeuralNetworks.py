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

def getInspiredVisualGeometryGroup(numClasses: int):
    """ Return an Instance of this model """
    return InspiredVisualGeometryGroup(numClasses)


    
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

class InspiredVisualGeometryGroup(torch.nn.Module):
    """ Class for a Model inspired by Visual Geometry Group (2014) """
    
    def __init__(self,numClasses: int):
        """ Constructor """
        super().__init__()
        self._numClasses = numClasses
        self.layers  = torch.nn.ParameterList([None] * 16)
        
        self.__initLayerGroup01()
        self.__initLayerGroup02()
        self.__initLayerGroup03()
        self.__initLayerGroup04()
        self.__initDenseLayers()
        
    def __del__(self):
        """ Destructor """
        pass

        # Public Interface

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for  Forward pass mechanism """
        x = self.layers[0].__call__(x)
        x = self.layers[1].__call__(x)
        x = self.layers[2].__call__(x)
        x = self.layers[3].__call__(x)
        x = self.layers[4].__call__(x)
        x = self.layers[5].__call__(x)
        x = self.layers[6].__call__(x)
        x = self.layers[7].__call__(x)
        x = self.layers[8].__call__(x)
        x = self.layers[9].__call__(x)
        x = self.layers[10].__call__(x)
        x = self.layers[11].__call__(x)
        x = self.layers[12].__call__(x)
        x = self.layers[13].__call__(x)
        x = self.layers[14].__call__(x)
        x = self.layers[15].__call__(x)
        return x

    # Private Interface
    
    def __initLayerGroup01(self):
        """ Initialize Layer Chain """
        self.layers[0] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[1] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[2] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)))
        return None
    
    def __initLayerGroup02(self):
        """ Initialize Layer Chain """
        self.layers[3] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[4] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[5] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)))
        return None
    
    def __initLayerGroup03(self):
        """ Initialize Layer Chain """
        self.layers[6] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[7] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[8] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)))
        return None
    
    def __initLayerGroup04(self):
        """ Initialize Layer Chain """
        self.layers[9] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[10] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self.layers[11] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2)))
        return None
    
    def __initDenseLayers(self):
        """ Initialize Dense Layers """
        self.layers[12] = torch.nn.Sequential(
            torch.nn.Flatten(
                start_dim=1,
                end_dim=-1))
        self.layers[13] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=3136,   # (64 x 7 x 7)
                out_features=128),
            torch.nn.ReLU())
        self.layers[14] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=128,
                out_features=64),
            torch.nn.ReLU())
        self.layers[15] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=64,
                out_features=self._numClasses),
            torch.nn.Softmax())
        return None



        

"""
    Author:         Landon Buell
    Date:           June 2023
"""