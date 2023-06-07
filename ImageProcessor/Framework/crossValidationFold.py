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

import appConfig

        #### CLASS DEFINITIONS ####

class CrossValidationFold:
    """ Represents a collection of samples use for a Cross Validation Fold """

    def __init__(self,
                 foldIndex: int,
                 samplesInFold: np.ndarray):
        """ Constructor """
        self._foldIndex     = foldIndex 
        self._sampleQueue   = samplesInFold.astype(np.int32)
        self._queueIter     = 0

    def __del__(self):
        """ Destructor """

    # Accessors

    def getFoldIndex(self) -> int:
        """ Return the Fold Index """
        return self._foldIndex

    def getSize(self) -> int:
        """ Return the size of this fold """
        return self._sampleQueue.size

    def getRemaining(self) -> int:
        """ Return the number of samples remaining """
        return (self._sampleQueue.size - self._queueIter)

    def getAll(self) -> np.ndarray:
        """ Get all indexes in this fold """
        return self._sampleQueue

    def getNext(self) -> int:
        """ Get the next sample Index """
        if (self.isFinished() == True):
            # Iterated through all items
            self.resetIterator()
            return -1

        nextIndex = self._sampleQueue[self._queueIter]
        self._queueIter += 1
        return nextIndex

    def getNextBatchIndexes(self,batchSize: int) -> np.ndarray:
        """ Get the indexes for the next batch of samples """
        batchSize = min(batchSize,self.getRemaining())
        batchSamples = np.empty(shape=(batchSize,),dtype=np.int32)
        for ii in range(batchSize):
            batchSamples[ii] = self._sampleQueue[self._queueIter]
            self._queueIter += 1
        return batchSamples

    # Public Interface

    def isFinished(self) -> bool:
        """ Return T/F if this fold has iterated through all samples """
        return (self._queueIter >= self._sampleQueue.size)

    def resetIterator(self) -> None:
        """ Resest the internal iterator """
        msg = "\tReseting the iterator on fold #{0}".format(str(self))
        # TODO: Log this message
        self._queueIter = 0
        return None

    # Magic Methods

    def __str__(self) -> str:
        """ Return string representation of instance """
        return "Fold #{0} w/ size={1}".format(
            self._foldIndex,self._sampleQueue.size)

    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Fold #{0} w/ size={1} @ {2}".format(
            self._foldIndex,self.getSize(),hex(id(self)))

    def __len__(self) -> int:
        """ Return the size of this instance """
        return self._sampleQueue.size

        


 

