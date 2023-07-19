"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        MachineLearningTools
    Namespace:      N/A
    File:           confusionMatrix.py
    Author:         Landon Buell
    Date:           July 2023
"""

        #### IMPORTS ####

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import modelHistoryInfo

        #### CLASS DEFINTIONS ####

class ClassificationReport:
    """ Generate + Export a report of classification scores """

    def __init__(self,numClasses: int):
        """ Constructor """
        self._numClasses        = numClasses
        self._confusionMatrix   = ConfusionMatrix(numClasses)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getNumClasses(self) -> int:
        """ Return the number of classes """
        return self._numClasses

    def getConfusionMatrix(self):
        """ Return the underlying Confusion Matrix """
        return self._confusionMatrix

    # Public Interface

    def update(self,evaluationHistory: modelHistoryInfo.ModelTestHistoryInfo) -> None:
        """ Update the state of this report from ModelTestHistoryInfo instance """
        truths      = evaluationHistory.getGroundTruths()   # 1D array
        predictions = evaluationHistory.getPredictions()    # 2D array
        # Store the results
        self._confusionMatrix.updateFromPredictions(truths,predictions)

    def export(self,
               outputFolder: str,
               currentFoldIndex: int):
        """ Write all classification Report Data to Disk """

        # Export the Standard Confusion Matrix
        standardConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixStandard{0}.csv".format(currentFoldIndex))
        self._confusionMatrix.exportStandardMatrixAsCsv(standardConfusionMatrixCsvPath)
        standardConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixStandard{0}.png".format(currentFoldIndex))
        self._confusionMatrix.exportStandardMatrixAsPng(standardConfusionMatrixPngPath)

        # Export the Weighted Confusion Matrix
        weightedConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixWeighted{0}.csv".format(currentFoldIndex))
        self._confusionMatrix.exportWeightedMatrixAsCsv(weightedConfusionMatrixCsvPath)
        weightedConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixWeighted{0}.png".format(currentFoldIndex))
        self._confusionMatrix.exportWeightedMatrixAsPng(weightedConfusionMatrixPngPath)

        # All done!
        return None

class ConfusionMatrix:
    """ Represents a Confusion Matrix """

    def __init__(self,
                 numClasses:int):
        """ Constructor """
        self._numClasses        = numClasses
        self._numTruthSamples   = np.zeros(shape=(numClasses,),dtype=np.uint16)    
        self._matrixStandard    = np.zeros(shape=(numClasses,numClasses),dtype=np.uint16)
        self._matrixWeighted    = np.zeros(shape=(numClasses,numClasses),dtype=np.float32)

    def __del__(self):
        """ Destructor """
        pass

    # Getters and Setters

    def getNumClasses(self) -> int:
        """ Return the number of classes """
        return self._numClasses

    def getNumSamplesInClass(self,classIndex: int) -> int:
        """ Return the number of truth instances of a class """
        return self._numTruthSamples[classIndex]

    def getTotalNumSamples(self) -> int:
        """ Return the total Number of samples seen """
        return np.sum(self._numTruthSamples)

    def getStandardMatrix(self) -> np.ndarray:
        """ Return the current standard confusion matrix """
        return self._matrixStandard

    def getWeightedMatrix(self) -> np.ndarray:
        """ Return the current weighted confusion matrix """
        return self._matrixWeighted

    # Getters and Setters 

    def updateFromPredictions(self,
                              grouthTruths: np.ndarray,
                              predictions: np.ndarray) -> None:
        """ Update Instance w/ a set of truths & predictions """
        numSamples = grouthTruths.shape[0]
        if (predictions.shape[0] != grouthTruths.shape[0]):
            msg = "Got truth labels w/ {0} samples and predictions with {1} samples".format(
                predictions.shape[0],grouthTruths.shape[0])
            raise RuntimeError(msg)
        if (numSamples == 0):
            msg = "Got {0} samples to store".format(numSamples)
            raise RuntimeError(msg)
        # Update the matrices
        classPredicitons        = np.argmax(predictions,axis=1)
        predictionConfidences   = np.max(predictions,axis=1)
        for ii in range(numSamples):
            trueClass = grouthTruths[ii]
            predClass = classPredicitons[ii]
            self._numTruthSamples[trueClass] += 1   # Increment the truth counter
            self._matrixStandard[trueClass,predClass] += 1                      # Prediction
            self._matrixStandard[trueClass,predClass] += predictionConfidences  # confidence
        return None

    def exportStandardMatrixAsCsv(self,fullOutputPath: str) -> None:
        """ Export the Confusion matrix as a CSV file """
        frame = pd.DataFrame(data=self._matrix,columns=None,index=None)
        frame.to_csv(fullOutputPath,columns=None,index=False,mode="w")
        return None

    def exportStandardMatrixAsPng(self,fullOutputPath: str,show=False) -> None:
        """ Export the Confusion Matrix as a PNG file """
        self.__plotStandardConfusionMatrix(show,fullOutputPath)
        return None

    def exportWeightedMatrixAsCsv(self,fullOutputPath: str) -> None:
        """ Export the Confusion matrix as a CSV file """
        frame = pd.DataFrame(data=self._matrix,columns=None,index=None)
        frame.to_csv(fullOutputPath,columns=None,index=False,mode="w")
        return None

    def exportWeightedMatrixAsPng(self,fullOutputPath: str,show=False) -> None:
        """ Export the Confusion Matrix as a PNG file """
        self.__plotWeightedConfusionMatrix(show,fullOutputPath)
        return None

    # Private Interface
        
    def __plotStandardConfusionMatrix(self,show=True,savePath=None):
        """ Plot the Standard Confusion Matrix """
        plt.figure(figsize=(16,12))
        plt.xlabel("Predicted Label",size=24,weight='bold')
        plt.ylabel("Actual Label",size=24,weight='bold')
        plt.imshow(self._matrixStandard,cmap=plt.cm.viridis)
        if (savePath is not None):
            plt.savefig(savePath)
        if (show == True):
            plt.show()
        plt.close()
        return None

    def __plotWeightedConfusionMatrix(self,show=True,savePath=None):
        """ Plot the Weighted Confusion Matrix """
        plt.figure(figsize=(16,12))
        plt.xlabel("Predicted Label",size=24,weight='bold')
        plt.ylabel("Actual Label",size=24,weight='bold')
        plt.imshow(self._matrixWeighted,cmap=plt.cm.viridis)
        if (savePath is not None):
            plt.savefig(savePath)
        if (show == True):
            plt.show()
        plt.close()
        return None

"""
    Author:         Landon Buell
    Date:           June 2023
"""

        
