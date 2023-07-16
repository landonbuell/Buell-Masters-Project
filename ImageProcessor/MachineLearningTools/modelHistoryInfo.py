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
import matplotlib.pyplot as plt

import callbackTools

        #### CLASS DEFINITIONS ####

class ModelTrainHistoryInfo:
    """ Stores Historical information from a model's train or test process """

    def __init__(self):
        """ Constructor """
        self._exportCounts = 0
        self._epochs        = np.array([],dtype=np.float32)
        self._losses        = np.array([],dtype=np.float32)
        self._accuracies    = np.array([],dtype=np.float32)
        self._precisions    = np.array([],dtype=np.float32)
        self._recalls       = np.array([],dtype=np.float32)
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getLossHistory(self) -> np.ndarray:
        """ Return the Loss History """
        return self._losses

    # Public Interface

    def updateFromBatchLog(self,batchLog: dict):
        """ Update Instance w/ data from a batch Log """
        self._losses        = np.append(self._losses,batchLog["loss"])
        self._accuracies    = np.append(self._accuracies,batchLog["accuracy"])
        self._precisions    = np.append(self._precisions,batchLog["precision"])
        self._recalls       = np.append(self._recalls,batchLog["recall"])
        return None

    def reset(self) -> None:
        """ Reset the state of instance to construction """
        self._epochs        = np.array([],dtype=np.float32)
        self._losses        = np.array([],dtype=np.float32)
        self._accuracies    = np.array([],dtype=np.float32)
        self._precisions    = np.array([],dtype=np.float32)
        self._recalls       = np.array([],dtype=np.float32)
        return None

    def plotAll(self,show=True,save=None) -> None:
        """ Generate and optionally show and save history of all scores """
        plt.figure(figsize=(16,12))
        plt.title("Training History",size=32,weight='bold')
        plt.xlabel("Epoch Index",size=24,weight='bold')
        plt.ylabel("Metric Score",size=24,weight='bold')

        plt.plot(self._losses,label="Loss")
        plt.plot(self._precisions,label="Precision")
        plt.plot(self._recalls,label="Recall")

        plt.grid()
        plt.legend()
        if (show == True):
            plt.show()
        plt.close()
        return None

    def toDataFrame(self) -> pd.DataFrame:
        """ Return history data as a pandas dataframe """
        data = {"loss"      : self._losses,
                "accuracy"  : self._accuracies,
                "preicsion" : self._precisions,
                "recalls"   : self._recalls}
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
        self._exportCounts = 0
        self._truths    = np.array([],dtype=np.int16)
        self._outputs   = np.array([],dtype=np.int16)
        self._scores    = np.array([],dtype=np.float32)
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    # Public Interface

    def updatedWithBatch(self,
                         truth: np.int16,
                         output: np.int16,
                         score: np.float32) -> None:
        """ Update Instance w/ a set of outputs """
        self._truths    = np.append(self._truths,truth)
        self._outputs   = np.append(self._outputs,output)
        self._scores    = np.append(self._scores,score)
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