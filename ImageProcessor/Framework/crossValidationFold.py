"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           sampleManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import numpy as np

import simpleQueue
import appConfig

        #### CLASS DEFINITIONS ####

class CrossValidationFold:
    """ Represents a collection of samples use for a Cross Validation Fold """

    def __init__(self,
                 foldIndex: int,
                 samplesInFold: np.ndarray):
        """ Constructor """
        self._foldIndex     = foldIndex 
        self._sampleQueue   = samplesInFold
        self._queueIter     = 0

    def __del__(self):
        """ Destructor """
        self._seenSamples.clear()
        self._usedSamples.clear()

    # Accessors

    def getFoldIndex(self) -> int:
        """ Return the Fold Index """
        return self._foldIndex

    def getSize(self) -> int:
        """ Return the size of this fold """
        return self._sampleQueue.size

    def getAll(self) -> np.ndarray:
        """ Get all indexes in this fold """
        return self._sampleQueue

    def getNext(self) -> int:
        """ Get the next sample Index """
        nextIndex = self._sampleQueue[self._queueIter]
        self._queueIter += 1
        return nextIndex

    # Public Interface

    # Magic Methods

    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Fold #{0} w/ size = {1} @ {2}".format(
            self._foldIndex,self.getSize(),hex(id(self)))

    def __len__(self) -> int:
        """ Return the size of this instance """
        return self._sampleQueue.size

        


 

