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

import modelHistoryInfo

        #### CLASS DEFINTIONS ####

class ClassificationReport:
    """ Generate + Export a report of classification scores """

    def __init__(self,numClasses: int):
        """ Constructor """
        self._numClasses        = numClasses
        self._standardConfMat   = ConfusionMatrix(numClasses,False)
        self._weightedConfMat   = ConfusionMatrix(numClasses,True)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getNumClasses(self) -> int:
        """ Return the number of classes """
        return self._numClasses

    # Public Interface

    def update(self,evaluationHistory: modelHistoryInfo.ModelTestHistoryInfo) -> None:
        """ Update the state of this report from ModelTestHistoryInfo instance """
        truths      = evaluationHistory.getGroundTruths()   # 1D array
        predictions = evaluationHistory.getPredictions()    # 2D array
        # Store the results
        self._standardConfMat.updateFromPredictions(truths,predictions)
        self._weightedConfMat.updateFromPredictions(truths,predictions)

    def export(self,
               outputFolder: str,
               currentFoldIndex: int):
        """ Write all classification Report Data to Disk """

        # Export the Standard Confusion Matrix
        standardConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixStandard{0}.csv".format(currentFoldIndex))
        self._standardConfMat.exportMatrixAsCsv(standardConfusionMatrixCsvPath)
        standardConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixStandard{0}.png".format(currentFoldIndex))
        self._standardConfMat.exportMatrixAsPng(standardConfusionMatrixPngPath)

        # Export the Weighted Confusion Matrix
        weightedConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixWeighted{0}.csv".format(currentFoldIndex))
        self._standardConfMat.exportMatrixAsCsv(weightedConfusionMatrixCsvPath)
        weightedConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixWeighted{0}.png".format(currentFoldIndex))
        self._standardConfMat.exportMatrixAsPng(weightedConfusionMatrixPngPath)

        # All done!
        return None

class ConfusionMatrix:
    """ Represents a Confusion Matrix """

    def __init__(self,
                 numClasses:int,
                 weighted=False):
        """ Constructor """
        self._numClasses        = numClasses
        self._numTruthSamples   = np.zeros(shape=(numClasses,),dtype=np.uint16)
        self._isWeighted        = weighted

        if (self._isWeighted == True):
            # Weighted Confusion Matrix
            self._updateCallback = self.__updateWeightedConfusionMatrix
            dataType = np.float32
        else:
            # Standard Confusion Matrix
            self._updateCallback = self.__updateStandardConfusionMatrix
            dataType = np.uint16
            
        self._matrix            = np.zeros(shape=(numClasses,numClasses),dtype=dataType)

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

    def isWeighted(self) -> bool:
        """ Return T/F if indexes are weighted by confidence """
        return self._isWeighted

    def getMatrix(self) -> np.ndarray:
        """ Return the current confusion matrix """
        return self._matrix

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
        # Update as needed
        self._updateCallback.__call__(grouthTruths,predictions)
        return None



    def exportMatrixAsCsv(self,fullOutputPath: str) -> None:
        """ Export the Confusion matrix as a CSV file """
        frame = pd.DataFrame(data=self._matrix,columns=None,index=None)
        frame.to_csv(fullOutputPath,columns=None,index=False,mode="w")
        return None

    def exportMatrixAsPng(self,fullOutputPath: str,show=False) -> None:
        """ Export the Confusion Matrix as a PNG file """
        self.__plotMatrix(show,fullOutputPath)
        return None

    # Private Interface

    def __updateStandardConfusionMatrix(self,
                                        grouthTruths: np.ndarray,
                                        predictions: np.ndarray) -> None:
        """ Update the matrix entries for a standard confusion matrix """
        numSamples = grouthTruths.shape[0]
        classPredicitons = np.argmax(predictions,axis=1)
        for ii in range(numSamples):
            trueClass = grouthTruths[ii]
            predClass = classPredicitons[ii]
            self._numTruthSamples[trueClass] += 1   # Increment the truth counter
            self._matrix[trueClass,predClass] += 1  # Prediction
        return None

    def __updateWeightedConfusionMatrix(self,
                                        grouthTruths: np.ndarray,
                                        predictions: np.ndarray) -> None:
        """ Update the matrix entries for a standard confusion matrix """
        numSamples = grouthTruths.shape[0]
        classPredicitons = np.argmax(predictions,axis=1)
        predictionConfidences = np.max(predictions,axis=1)
        for ii in range(numSamples):
            trueClass = grouthTruths[ii]
            predClass = classPredicitons[ii]
            self._numTruthSamples[trueClass] += 1   # Increment the truth counter
            self._matrix[trueClass,predClass] += predictionConfidences  # Prediction
        return None
        
    def __plotMatrix(self,show=True,savePath=None):
        """ Plot the Matrix """
        plt.figure(figsize=(16,12))
        plt.xlabel("Predicted Label",size=24,weight='bold')
        plt.ylabel("Actual Label",size=24,weight='bold')
        plt.imshow(self._matrix,cmap=plt.cm.viridis)
        if (savePath is not None):
            plt.savefig(savePath)
        if (show == True):
            plt.show()
        plt.close()
        