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

    def __init__(self,numClasses:int):
        """ Constructor """
        self._numClasses    = numClasses
        self._exportCounts  = 0       
        self._groundTruths  = np.array([],dtype=np.float32)
        self._predictions   = np.array([],dtype=np.float32)
   
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getNumClasses(self) -> int:
        """ Return the number of classes """
        return self._numClasses

    def getNumSamples(self) -> int:
        """ Return the number of samples """
        return self._groundTruths.size

    def getGroundTruths(self) -> np.ndarray:
        """ Return an array of truths """
        return self._groundTruths

    def getPredictions(self) -> np.ndarray:
        """ Return a 2D array of all predictions by class """
        return self._predictions

    def getClassPredictions(self) -> np.ndarray:
        """ Return an array of class predictions """
        return np.argmax(self._predictions,axis=1,dtype=np.int16)

    def getConfidences(self) -> np.ndarray:
        """ Return an array of predictions confidences """
        return np.max(self._predictions,axis=1,dtype=np.float32)

    # Public Interface

    def updateFromPredictions(self,truths,predictions) -> None:
        """ Update Instance w/ a set of outputs """
        numSamples = truths.shape[0]
        if (predictions.shape[0] != truths.shape[0]):
            msg = "Got truth labels w/ {0} samples and predictions with {1} samples".format(
                predictions.shape[0],truths.shape[0])
            raise RuntimeError(msg)
        if (numSamples == 0):
            msg = "Got {0} samples to store".format(msg)
            raise RuntimeError(msg)
        # Update as needed
        self._groundTruths  = np.append(self._groundTruths,truths)
        self._predictions   = np.append(self._predictions,predictions)
        totalNumSamples     = self._groundTruths.size
        self._predictions   = np.reshape(self._predictions,newshape=(totalNumSamples,self._numClasses))
        return None

    def reset(self) -> None:
        """ Reset the state of instance to construction """
        self._exportCounts  = 0     
        self._groundTruths  = np.array([],dtype=np.float32)
        self._predictions   = np.array([],dtype=np.float32)
        return None

    def plotAll(self,show=True,save=None) -> None:
        """ Generate and optionally show and save history of all scores """
        # TODO: Implement this
        return None

    def toDataFrame(self) -> pd.DataFrame:
        """ Return history data as a pandas dataframe """
        data = {"truth"     : self.getGroundTruths(),
                "predict"   : self.getPredictions()}
        for ii in range(self._numClasses):
            newKey = "class{0}".format(ii)
            newVal = self._classProbabilities[:,ii]
            data[newKey] = newVal
        frame = pd.DataFrame(data=data,index=None)
        return frame

    def export(self,outputPath: str) -> bool:
        """ Write history info to specified path. Return T/F if successful """
        frame = self.toDataFrame()
        frame.to_csv(outputPath,index=False,mode="w")
        self._exportCounts += 1
        return True

    @staticmethod
    def importFromFile(importPath: str):
        """ Read history info from specified path """
        frame = pd.read_csv(importPath,header=0,index_col=None)
        numSamples = frame.shape[0]
        numClasses = frame.shape[1] - 2 # account for 'truth' & 'prediction' columns
        truths = frame["truth"] 
        predictions = np.empty(shape=(numSamples,numClasses),dtype=np.float32)
        for ii in range(numClasses):
            key = "class{0}".format(ii)
            predictions[:,ii] = frame[key]
        modelTestHistoryInfo = ModelTestHistoryInfo(numClasses)
        modelTestHistoryInfo.updateFromPredictions(truths,predictions)
        return modelTestHistoryInfo

    # Private Interface





"""
    Author:         Landon Buell
    Date:           June 2023
"""