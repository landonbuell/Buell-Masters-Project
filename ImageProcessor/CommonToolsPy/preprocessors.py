"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           processors.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import batch

import numpy as np

        #### CLASS DEFINITIONS ####

class BasePreprocessor:
    """ Abstract Base Class for all Preprocessors """

    def __init__(self,
                 sampleShape: tuple,
                 name: str):
        """ Constructor """
        self._sampleShape   = sampleShape
        self._name          = name

    def __del__(self):
        """ Destructor """

    # Accessors

    def getSampleShape(self) -> tuple:
        """ Return the expected shape of each sample """
        return self._sampleShape

    def getNumFeatures(self) -> int:
        """ Return the number of features in each sample """
        result = 1
        for axisSize in self._sampleShape:
            result = (result * axisSize)
        return result

    def getName(self) -> str:
        """ Return the Name of this preprocessor """
        return self._name

    # Public Interface

    def fit(self):
        """ Fit this preprocessor to the provided data """

        return None


    def applyFit(self,batchData):
        """ Apply the fitted data to the provided batch of data """

        return batchData



class CustomStandardScaler(BasePreprocessor):
    """ Scale input tensors to have mean = 0, var = 1 """

    __NAME = "CustomStandardScaler"

    def __init__(self):
        """ Constructor """
        super().__init__(CustomStandardScaler.__NAME)
        self._means = np.array([],dtype=np.float32)
        self._varis = np.array([],dtype=np.float32)

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Public Interface

    
    def fitToDatabase(self,
                      sampleMgr): #sampleManger.SampleManager
        """ Fit Sample Manager to Every sample in the sample Database """
        msg = "WARNING: CustomStandardScaler is not yet implemented."
        sampleMgr.logMessage()
        #self.__fitToDatabase(sampleMgr)
        return self

    # Private Interface

    def __fitToDatabase(self,sampleMgr):
        """ Helper to fit every sample in the database """
        numFeatures = self.getNumFeatures()
        samplesToFitAtOnce = 128
        totalNumSamplesInDatabase = sampleMgr.getSize()

        featureStartIndex = 0
        featureStopIndex = min(featureStartIndex + samplesToFitAtOnce,numFeatures)

        while (featureStartIndex < numFeatures):
            featuresToProcessMask = np.arange(featureStartIndex,featureStopIndex,dtype=np.int16)
            sampleData = np.empty(
                shape=(totalNumSamplesInDatabase,featuresToProcessMask.size),
                dtype=np.float32)
            self.__storeFeatureGroup(
                sampleData,
                featufeaturesToProcessMask,
                numFeatures)


        return None

    def __storeFeatureGroup(self,
                            sampleData: np.ndarray,
                            featureMask: np.ndarray,
                            sampleMgr):
        """ Store a group of features across all samples """
        pass 

"""
    Author:         Landon Buell
    Date:           June 2023
"""