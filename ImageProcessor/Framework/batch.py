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

import tensorflow as tf

import imageProcessingApp

        #### CONSTANTS ####

SHAPE_CHANNELS_FIRST    = (3,200,200)
SHAPE_CHANNELS_LAST     = (200,200,3)

        #### FUNCTION DEFINITONS ####

def oneHotEncode(labels: tf.Tensor,
                 numClasses: int):
    """ One-Hot encode a vector of labels """
    if (labels.ndim > 1):
        msg = "Cannot encode labels w/ ndim={0}. Expected ndim=1".format(labels.ndim)
        raise RuntimeError(msg)
    oneHot = tf.zeros(shape=(labels.shape[0],numClasses),dtype=tf.int32)
    for ii,tgt in enumerate(labels):
        oneHot[ii,tgt] = 1
    return oneHot

        #### CLASS DEFINITIONS ####

class SampleBatch:
    """ Batch of Samples to handoff to a nueral network """

    __batchCounter = 0

    def __init__(self,
                 numSamples: int,
                 sampleShape: tuple,
                 dataTypeX=tf.uint8,
                 dataTypeY=tf.int16):
        """ Constructor """
        shapeX = (numSamples,) + sampleShape
        shapeY = (numSamples,)

        self._X = tf.zeros(shape=shapeX,dtype=dataTypeX)
        self._y = tf.zeros(shape=shapeY,dtype=dataTypeY)

        self._batchIndex    = SampleBatch.__batchCounter
        SampleBatch.__batchCounter += 1

    def __del__(self):
        """ Destructor """
        self._X = None
        self._y = None

    # Accessors

    def getDataTypeX(self):
        """ Return the data type for features """
        return self._X.dtype

    def getDataTypeY(self):
        """ Return the data type for labels """
        return self._y.dtype

    def setDataTypeX(self,dataType):
        """ Set the data type for the features """
        self._X = self._X.cast(dataType)
        return self

    def setDataTypeY(self,dataType):
        """ Set the data dtype for labels """
        self._y = self._y.cast(dataType)
        return self

    def getX(self) -> tf.Tensor:
        """ Return Features """
        return self._X

    def getY(self) -> tf.Tensor:
        """ Return Y """
        return self._y

    def setX(self, newX: tf.Tensor) -> None:
        """ Set the Tensor for the features """
        if (newX.shape[0] != self._X.shape[0]):
            msg = "Expected new features to have {0} samples but got {1}".format(
                self._X.shape[0],newX.shape[0])
            raise RuntimeError(msg)
        self._X = newX
        return None

    def getBatchIndex(self) -> int:
        """ Get the batch Index """
        return self._batchIndex

    def getNumSamples(self) -> int:
        """ Return the size of the batch """
        return self._y.shape[0]

    def getSampleShape(self) -> tuple:
        """ Return the Shape of each item in the design matrix """
        return self._X.shape[1:]

    # Public Interface

    def getOneHotY(self,numClasses: int) -> tf.Tensor:
        """ One-hot-encode this batch's labels """
        return oneHotEncode(self._y,numClasses)

    # Magic Methods

    def __getitem__(self,key: int):
        """ Return the (X,y) pair at specified index """
        return (self._X[key],self._y[key],)

    def __setitem__(self,key: int, val: tuple):
        """ Set the (X,y) pair at specified index """
        x = val[0].type(self.getDataTypeX())
        y = tf.Tensor(val[1],dtype=self.getDataTypeY())
        self._X[key] = x
        self._y[key] = y
        return self

    def __str__(self) -> str:
        """ Return string representaion of instance """
        return "Batch#{0} w/ size={1}".format(self._batchIndex,self.getNumSamples())

    def __repr__(self) -> str:
        """ Return debug representation of batch """
        return "Batch#{0} @ {1}".format(self._batchIndex,hex(id(self)))

    # Static Interface

    @staticmethod
    def getBatchCounter() -> int:
        """ Get the current batch counter """
        return SampleBatch.__batchCounter

    @staticmethod
    def resetBatchCounter() -> None:
        """ Return the batch counter to zero """
        SampleBatch.__batchCounter = 0
        return None

"""
    Author:         Landon Buell
    Date:           June 2023
"""