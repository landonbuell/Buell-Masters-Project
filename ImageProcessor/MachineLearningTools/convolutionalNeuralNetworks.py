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

def getInspiredVisualGeometryGroup(numClasses: int):
    """ Return an Instance of this model """
    return InspiredVisualGeometryGroup(numClasses)
    
        #### CLASS DEFINTIONS ####

class InspiredVisualGeometryGroup(torch.nn.Module):
    """ Class for a Model inspired by Visual Geometry Group (2014) """
    
    def __init__(self,numClasses: int):
        """ Constructor """
        super().__init__()
        self._numClasses = numClasses
        self._layers  = [None] * 18
        
        self.__initLayerGroup01()
        self.__initLayerGroup02()
        self.__initLayerGroup03()
        self.__initLayerGroup04()
        self.__initDenseLayers()
        
    def __del__(self):
        """ Destructor """
        pass

        # Public Interface

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Define Forward pass mechanism """
        x = torch.clone(inputs)
        for ii,layer in enumerate(self._layers):
            x = layer(x)
        return x

    # Private Interface
    
    def __initLayerGroup01(self):
        """ Initialize Layer Chain """
        self._layers[0] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[1] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[2] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(2,2)))
        return None
    
    def __initLayerGroup02(self):
        """ Initialize Layer Chain """
        self._layers[3] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[4] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[5] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(2,2)))
        return None
    
    def __initLayerGroup03(self):
        """ Initialize Layer Chain """
        self._layers[6] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[7] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[8] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(2,2)))
        return None
    
    def __initLayerGroup04(self):
        """ Initialize Layer Chain """
        self._layers[9] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[10] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1)),
            torch.nn.ReLU())
        self._layers[11] = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3,3),
                stride=(1,1)))
        return None
    
    def __initDenseLayers(self):
        """ Initialize Dense Layers """
        self._layers[12] = torch.nn.Sequential(
            torch.nn.Flatten(
                start_dim=1,
                end_dim=-1))
        self._layers[13] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=6272,
                out_features=4096),
            torch.nn.ReLU())
        self._layers[14] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=4096,
                out_features=2048),
            torch.nn.ReLU())
        self._layers[15] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2048,
                out_features=1024),
            torch.nn.ReLU())
        self._layers[16] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=1024,
                out_features=512),
            torch.nn.ReLU())
        self._layers[17] = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512,
                out_features=self._numClasses),
            torch.nn.Softmax())
        return None



        

"""
    Author:         Landon Buell
    Date:           June 2023
"""