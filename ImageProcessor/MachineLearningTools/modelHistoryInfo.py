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

import callbacks

        #### CLASS DEFINITIONS ####

class ModelHistoryInfo:
    """ Stores Historical information from a model's train or test process """

    def __init__(self):
        """ Constructor """
        self._losses    = np.array([],dtype=np.float32)
        self._precision = np.array([],dtype=np.float32)
        self._recalls   = np.array([],dtype=np.float32)
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getLossHistory(self) -> np.ndarray:
        """ Return the Loss History """
        return self._losses

    def getPrecisionHistory(self) -> np.ndarray:
        """ Return the precision history """
        return self._precision

    def getRecallHistory(self) -> np.ndarray:
        """ Return the recall history """
        return self._recalls

    def getF1History(self) -> np.ndarray:
        """ Return the F1-Score history """
        return 2 * (self._precision * self._recalls) / (self._precision + self._recalls)

    # Public Interface

    def updateWithTrainBatch(self,preds,truth,numClasses) -> None:
        """ Update state w/ outputs + truth of a training batch """
        lossScore           = torch.
        precisionScore      = callbacks.multiclassPrecisionScore(preds,truth)
        recallScore         = callbacks.multiclassRecallScore(preds,truth)

        self._losses    = np.append(self._losses,cost)
        self._precision = np.append(self._precision,precisionScore)
        self._recalls   = np.append(self._recalls,recallScore)

        return None



    def plotAll(self,show=True,save=None) -> None:
        """ Generate and optionally show and save history of all scores """
        # TODO: Implement this
        return None

    def toDataFrame(self) -> pd.DataFrame:
        """ Return history data as a pandas dataframe """
        data = {"Loss"      : self._losses,
                "Precision" : self._precision,
                "Recall"    : self._recalls,
                "F1"        : self.getF1History()}
        frame = pd.DataFrame(data=data,index=None)
        return frame

    def export(self,outputPath) -> bool:
        """ Write history info to specified path. Return T/F if successful """
        frame = self.toDataFrame()
        frame.to_csv(outputPath,index=False)
        return True

    # Private Interface

"""
    Author:         Landon Buell
    Date:           June 2023
"""