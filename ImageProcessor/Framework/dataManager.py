"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           dataManager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import commonEnumerations
import runInfo

import manager

        #### FUNCTION DEFINITIONS ####

        #### CLASS DEFINTIONS ####

class DataManager(manager.Manager):
    """
        DataManager stores important runtime information
    """

    __NAME = "DataManager"
    __MAX_NUM_CLASSES = 32

    def __init__(self,
                 app): #imageProcessingApp.ImageProcessingApp
        """ Constructor """
        super().__init__(app,DataManager.__NAME)
        
        self._classDatabase = [None] * DataManager.__MAX_NUM_CLASSES
        self._runInfo       = runInfo.RunInfo(self)

    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getRunInfo(self):
        """ Return Reference to RunInfo Structure """
        return self._runInfo

    def getClassName(self, classInt: int) -> str:
        """ Return the Name of the class based on the integer index """
        return self._classDatabase[classInt].className

    def getClassInt(self, className: str) -> int:
        """ Return the integer index of the class based on the name """
        for ii,item in enumerate(self._classDatabase):
            if (item.className == className):
                return ii
        # Not found:
        msg = "\tA class by the name of {0} was not found in the database".format(className)
        self.logMessage(msg)
        return -1

    def getNumClasses(self) -> int:
        """ Get the Number of classes in Use """
        numClasses = 0
        for item in self._classDatabase:
            if (item is not None):
                numClasses += 1
        return numClasses

    def getClassesInUse(self) -> list:
        """ Return a list of the class ints in use """
        classes = []
        for item in self._classDatabase:
            if (item is not None):
                classes.append( item.classInt )
        return classes

    # Public Interface

    def init(self) -> commonEnumerations.Status:
        """ Initialize this Manager """
        if (super().init() == commonEnumerations.Status.ERROR):
            return self._status

        # Plot Distrobution of Classes

        # Populate Sample Databse 
        self._setInitFinished(True)
        return self._status

    def call(self) -> commonEnumerations.Status:
        """ Run this manager """
        if (super().call() == commonEnumerations.Status.ERROR):
            return self._status

        self._setExecuteFinished(True)
        return self._status

    def cleanup(self) -> commonEnumerations.Status:
        """ Cleanup this manager """
        if (super().cleanup() == commonEnumerations.Status.ERROR):
            return self._status


        self.__exportRunInfo()
        self._setShutdownFinished(True)
        return self._status

    def registerClassWithDatabase(self,classInt: int, className: str) -> bool:
        """ Register a class w/ the Class Database """
        if (classInt >= DataManager.__MAX_NUM_CLASSES):
            msg = "Class database w/ cap={0} cannot store info about class {1}".format(
                DataManager.__MAX_NUM_CLASSES,classInt)
            self.logMessage(msg)
            return False

        # Otherwise, store the info
        if (self._classDatabase[classInt] is None):
            self._classDatabase[classInt] = DataManager.ClassDataStruct(
                classInt,className)
            self._classDatabase[classInt].expectedCount = 1
        else:
            self._classDatabase[classInt].expectedCount += 1
        return True

    # Public Struct

    class ClassDataStruct:
        """ Structure to store info about each class """

        def __init__(self,
                     className: str,
                     classInt: int):
            """ Constructor """
            self.className      = className
            self.classInt       = classInt
            self.expectedCount  = 0
            self.classifierCount = 0
            self.segmenterCount  = 0

        def __del__(self):
            """ Destructor """
            pass

    # Private Interface 

    def __plotClassDistrobution(self) -> None:
        """ Plot the Distrobution of classes in the data set """
        msg = "WARNING: Plotting distrobution of class data is not yet implemented"
        self.logMessage(msg)
        return None


    def __exportRunInfo(self) -> None:
        """ Export the runInfo class to disk """
        msg = "WARNING: Exporting RunInfo instance to dosl os not yet implemented"
        self.logMessage(msg)

        exportPath = os.path.join(self.getApp().getConfig().getOutputPath(),"runInfo.txt")
        self._runInfo.toDisk(exportPath)
        return None

    


"""
    Author:         Landon Buell
    Date:           May 2023
"""
