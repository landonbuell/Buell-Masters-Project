"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           batch.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import numpy as np

        #### FUNCTION DEFINITONS ####

def oneHotEncode(labels:np.ndarray,
                 numClasses: int):
    """ One-Hot encode a vector of labels """
    if (labels.ndim > 1):
        msg = "Cannot encode labels w/ ndim={0}. Expected ndim=1".format(labels.ndim)
        raise RuntimeError(msg)
    oneHot = np.zeros(shape=(labels.shape[0],numClasses),dtype=labels.dtype)
    for ii,tgt in labels:
        oneHot[ii,tgt] = 1
    return oneHot

        #### CLASS DEFINITIONS ####

class SampleBatch:
    """ Batch of Samples to handoff to a nueral network """

    __batchCounter = 0

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 isOneHot):
        """ Constructor """
        self._X = X
        self._y = X

        self._batchIndex    = SampleBatch.__batchCounter
        self._isOneHot      = isOneHot

         SampleBatch.__batchCounter += 1

    def __del__(self):
        """ Destructor """
        self._X = None
        self._y = None

    # Accessors

    def getX(self) -> np.ndarray
        """ Return Features """
        return self._X

    def getY(self) -> np.ndarray
        """ Return Y """
        return self._y

    def getBatchIndex(self) -> int:
        """ Get the batch Index """
        return self._batchIndex

    def getIsOneHot(self) -> bool:
        """ Return T/F if batch is one hot encoded """
        return self._isOneHot

    def getSize(self):
        """ Return the size of the batch """
        return self._X.size

    # Public Interface

    def oneHotEncode(self,numClasses: int):
        """ One-hot-encode this batch's labels """
        if (self._isOneHot == True):
            return self
        self._y = oneHotEncode(self._y,numClasses)
        self._isOneHot = True
        return self

    # Magic Methods

    def __str__(self) -> str:
        """ Return string representaion of instance """
        return "Batch#{0} w/ size={1}".format(self._batchIndex,self.getSize())

    def __repr__(self) -> str:
        """ Return debug representation of batch """
        return "Batch#{0} @ {1}".format(self._batchIndex,hex(id(self)))

"""
    Author:         Landon Buell
    Date:           June 2023
"""