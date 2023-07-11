"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           torchManager.py
    Author:         Landon Buell
    Date:           June 2023
"""

        #### IMPORTS ####

import pandas as pd
import numpy as np
import torch

import callbackTools

        #### CLASS DEFINITIONS ####

class ModelTrainHistoryInfo:
    """ Stores Historical information from a model's train or test process """

    def __init__(self):
        """ Constructor """
        self._losses    = np.array([],dtype=np.float32)
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getLossHistory(self) -> np.ndarray:
        """ Return the Loss History """
        return self._losses

    # Public Interface

    def appendLoss(self, batchLoss: np.float32) -> None:
        """ Append a Loss Item to Array of Losses """
        self._losses    = np.append(self._losses,batchLoss)
        return None

    def reset(self) -> None:
        """ Reset the state of instance to construction """
        self._losses    = np.array([],dtype=np.float32)
        self._precision = np.array([],dtype=np.float32)
        self._recalls   = np.array([],dtype=np.float32)
        return None

    def plotAll(self,show=True,save=None) -> None:
        """ Generate and optionally show and save history of all scores """
        # TODO: Implement this
        return None

    def toDataFrame(self) -> pd.DataFrame:
        """ Return history data as a pandas dataframe """
        data = {"Loss"      : self._losses}
        frame = pd.DataFrame(data=data,index=None)
        return frame

    def export(self,outputPath) -> bool:
        """ Write history info to specified path. Return T/F if successful """
        frame = self.toDataFrame()
        frame.to_csv(outputPath,index=False,mode="w")
        self._exportCounts += 1
        return True

    # Private Interface

class ModelTestHistoryInfo:
    """ Stores Historical information from a model's train or test process """

    def __init__(self):
        """ Constructor """
        self._truths    = np.array([],dtype=np.int16)
        self._outputs   = np.array([],dtype=np.int16)
        self._scores    = np.array([],dtype=np.float32)
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface

    def appendLoss(self, batchLoss: np.float32) -> None:
        """ Append a Loss Item to Array of Losses """
        self._losses    = np.append(self._losses,batchLoss)
        return None

    def reset(self) -> None:
        """ Reset the state of instance to construction """
        self._truths    = np.array([],dtype=np.int16)
        self._outputs   = np.array([],dtype=np.int16)       
        self._scores    = np.array([],dtype=np.float32)
        return None

    def plotAll(self,show=True,save=None) -> None:
        """ Generate and optionally show and save history of all scores """
        # TODO: Implement this
        return None

    def toDataFrame(self) -> pd.DataFrame:
        """ Return history data as a pandas dataframe """
        data = {"truth"     : self._truths,
                "outputs"   : self._outputs,
                "scores"    : self._scores}
        frame = pd.DataFrame(data=data,index=None)
        return frame

    def export(self,outputPath) -> bool:
        """ Write history info to specified path. Return T/F if successful """
        frame = self.toDataFrame()
        frame.to_csv(outputPath,index=False,mode="w")
        self._exportCounts += 1
        return True

    # Private Interface

"""
    Author:         Landon Buell
    Date:           June 2023
"""