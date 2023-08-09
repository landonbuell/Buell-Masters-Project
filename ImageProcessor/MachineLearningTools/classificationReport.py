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

    def __init__(self,classNames: list):
        """ Constructor """
        self._classNames        = classNames
        self._confusionMatrix   = ConfusionMatrix(len(self._classNames))
        self._groundTruths      = np.array([],dtype=np.uint16)
        self._predictions       = np.array([],dtype=np.uint16)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getClassNames(self) -> list:
        """ Return the list of class Names """
        return self._classNames

    def getNumClasses(self) -> int:
        """ Return the number of classes """
        return len(self._classNames)

    def getConfusionMatrix(self):
        """ Return the underlying Confusion Matrix """
        return self._confusionMatrix

    def getPrecisionScores(self) -> np.ndarray:
        """ Return an array of precisions by class """
        precisionMetric = tf.keras.metrics.Precision()
        result = np.zeros(shape=(self.getNumClasses(),),dtype=np.float32)
        for ii in range(self.getNumClasses()):
            maskTruths  = (self._groundTruths == ii)
            maskPreds   = (self._predictions == ii)
            precisionMetric.update_state(maskTruths,maskPreds)
            result[ii]  = precisionMetric.result()
            precisionMetric.reset_state()
        return result

    def getRecallScores(self) -> np.ndarray:
        """ Return an array of precisions by class """
        recallMetric = tf.keras.metrics.Recall()
        result = np.zeros(shape=(self.getNumClasses(),),dtype=np.float32)
        for ii in range(self.getNumClasses()):
            maskTruths  = (self._groundTruths == ii)
            maskPreds   = (self._predictions == ii)
            recallMetric.update_state(maskTruths,maskPreds)
            result[ii]  = recallMetric.result()
            recallMetric.reset_state()
        return result

    def getF1Scores(self) -> np.ndarray:
        """ Return an array of precisions by class """
        precisions  = self.getPrecisionScores()
        recalls     = self.getRecallScores()
        result  = 2 * (precisions * recalls) / (precisions + recalls)
        return result

    def getAccuraryScores(self) -> np.ndarray:
        """ Return an array of precisions by class """
        accuracyMetric = tf.keras.metrics.Accuracy()
        result = np.zeros(shape=(self.getNumClasses(),),dtype=np.float32)
        for ii in range(self.getNumClasses()):
            maskTruths  = (self._groundTruths == ii)
            maskPreds   = (self._predictions == ii)
            accuracyMetric.update_state(maskTruths,maskPreds)
            result[ii]  = accuracyMetric.result()
            accuracyMetric.reset_state()
        return result

    # Public Interface

    def update(self,evaluationHistory: modelHistoryInfo.ModelTestHistoryInfo) -> None:
        """ Update the state of this report from ModelTestHistoryInfo instance """
        if (evaluationHistory.getNumClasses() != self.getNumClasses()):
            msg = "Got an evaluation history w/ {0} classes, but expected {1}".format(
                evaluationHistory.getNumClasses(),self.getNumClasses())
            raise RuntimeError(msg)

        truths      = evaluationHistory.getGroundTruths()   # 1D array
        predictions = evaluationHistory.getPredictions()    # 2D array
        # Store the results
        self._confusionMatrix.updateFromPredictions(truths,predictions)

        # Store the class predictions too
        classPredictions = evaluationHistory.getClassPredictions()
        self._groundTruths  = np.append(self._groundTruths,truths)
        self._predictions   = np.append(self._predictions,classPredictions)
        return None

    def export(self,
               outputFolder: str):
        """ Write all classification Report Data to Disk """
        frameDict = {   "class"        : self._classNames,
                        "precision"    : self.getPrecisionScores(),
                        "recall"       : self.getRecallScores(),
                        "f1-score"     : self.getF1Scores(),
                        "accuracy"     : self.getAccuraryScores()}
        frame = pd.DataFrame(data=frameDict)
        reportPath = os.path.join(outputFolder,"classificationReport.csv")
        frame.to_csv(reportPath,index=True,mode="w")

        # Export the Standard Confusion Matrix
        standardConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixStandard.csv")
        self._confusionMatrix.exportStandardMatrixAsCsv(standardConfusionMatrixCsvPath)
        standardConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixStandard.png")
        self._confusionMatrix.exportStandardMatrixAsPng(standardConfusionMatrixPngPath)

        # Export the Weighted Confusion Matrix
        weightedConfusionMatrixCsvPath = os.path.join(outputFolder,"confusionMatrixWeighted.csv")
        self._confusionMatrix.exportWeightedMatrixAsCsv(weightedConfusionMatrixCsvPath)
        weightedConfusionMatrixPngPath = os.path.join(outputFolder,"confusionMatrixWeighted.png")
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
            self._matrixStandard[trueClass,predClass] += predictionConfidences[trueClass]  # confidence
        return None

    def exportStandardMatrixAsCsv(self,fullOutputPath: str) -> None:
        """ Export the Confusion matrix as a CSV file """
        frame = pd.DataFrame(data=self._matrixStandard,columns=None,index=None)
        frame.to_csv(fullOutputPath,columns=None,index=False,header=False,mode="w")
        return None

    def exportStandardMatrixAsPng(self,fullOutputPath: str,show=False) -> None:
        """ Export the Confusion Matrix as a PNG file """
        self.__plotStandardConfusionMatrix(show,fullOutputPath)
        return None

    def exportWeightedMatrixAsCsv(self,fullOutputPath: str) -> None:
        """ Export the Confusion matrix as a CSV file """
        frame = pd.DataFrame(data=self._matrixWeighted,columns=None,index=None)
        frame.to_csv(fullOutputPath,columns=None,index=False,header=False,mode="w")
        return None

    def exportWeightedMatrixAsPng(self,fullOutputPath: str,show=False) -> None:
        """ Export the Confusion Matrix as a PNG file """
        self.__plotWeightedConfusionMatrix(show,fullOutputPath)
        return None

    # Private Interface
        
    def __plotStandardConfusionMatrix(self,show=True,savePath=None):
        """ Plot the Standard Confusion Matrix """
        plt.figure(figsize=(16,12))
        plt.xlabel("Predicted Label",size=32,weight='bold')
        plt.ylabel("Actual Label",size=32,weight='bold')
        plt.xticks(ticks=range(self._numClasses),size=16)
        plt.yticks(ticks=range(self._numClasses),size=16)
        plt.imshow(self._matrixStandard,cmap=plt.cm.gray)
        if (savePath is not None):
            plt.savefig(savePath)
        if (show == True):
            plt.show()
        plt.close()
        return None

    def __plotWeightedConfusionMatrix(self,show=True,savePath=None):
        """ Plot the Weighted Confusion Matrix """
        plt.figure(figsize=(16,12))
        plt.xlabel("Predicted Label",size=32,weight='bold')
        plt.ylabel("Actual Label",size=32,weight='bold')
        plt.xticks(ticks=range(self._numClasses),size=16)
        plt.yticks(ticks=range(self._numClasses),size=16)
        plt.imshow(self._matrixWeighted,cmap=plt.cm.gray)
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

        
