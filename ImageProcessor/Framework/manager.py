"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           manager.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import appConfig
import imageProcessingApp

        #### CLASS DEFINITIONS ####

class Manager:
    """
        Parent Class for all Managers
    """

    def __init__(self,
                 app: imageProcessingApp.ImageProcessingApp,
                 name: str):
        """ Constructor """
        self._app       = app
        self._name      = name

        self._progress  = [False] * 6
        self._status    = appConfig.Status.SUCCESS
        
    def __del__(self):
        """ Destructor """
        pass

    # Accessors

    def getStatus(self) -> appConfig.Status:
        """ Return the Status of this manager """
        return self._status

    def getApp(self):
        """ Return Reference to Parent Application """
        return self._app

    def getName(self) -> str:
        """ Get the Name of this Manager """
        return self._name

    def getInitStarted(self) -> bool:
        """ Return T/F if initialization has started """
        return self._progress[0]

    def getInitFinished(self) -> bool:
        """ Return T/F if initialization has finished """
        return self._progress[1]

    def getExecuteStarted(self) -> bool:
        """ Return T/F if execution has started """
        return self._progress[2]

    def getExecutionFinished(self) -> bool:
        """ Return T/F if execution has finished """
        return self._progress[3]

    def getShutdownStarted(self) -> bool:
        """ Return T/F if shutdown has started """
        return self._progress[4]

    def getShutdownFinished(self) -> bool:
        """ Return T/F if shutdown has finished """
        return self._progress[5]

    # Public Interface

    def logMessage(self, message : str) -> None:
        """ Log Message to App """
        self._app.logMessage(message)
        return None

    def updateStatus(self, newStatus: appConfig.Status) -> bool:
        """ Update the status of this manager if newer status is more sever """
        intOldStatus = int(self._status)
        intNewStatus = int(newStatus)
        if (intNewStatus > intOldStatus):
            msg = "{0} status: {1} -> {2}".format(
                self._name,str(self._status),str(newStatus))
            self.logMessage(msg)
            self._status = intNewStatus
            return True
        return False

    def init(self) -> appConfig.Status:
        """ Initialize this Manager """
        self._setInitStarted(True)
        msg = "Intializing {0} ... ".format(self._name)
        self.logMessage(msg)

        return self._status

    def call(self) -> appConfig.Status:
        """ Execute this Manager """
        self._setExecuteStarted(True)

        return self._status

    def cleanup(self) -> appConfig.Status:
        """ Cleanup this Manager """
        msg = "Cleaning {0} ... ".format(self._name)
        self.logMessage(msg)

        return self._status

    # Proctected Interface

    def _overrideStatus(self, newStatus: appConfig.Status) -> None:
        """ Force Override the status of this Manager """
        self._status = newStatus
        return self

    def _setInitStarted(self,status: bool) -> None:
        """ Set T/F if initialization has started """
        self._progress[0] = status
        return None

    def _setInitFinished(self,status: bool) -> None:
        """ Set T/F if initialization has finished """
        self._progress[1] = status
        return None

    def _setExecuteStarted(self,status: bool) -> None:
        """ Set T/F if initialization has started """
        self._progress[2] = status
        return None

    def _setExecuteFinished(self,status: bool) -> None:
        """ Set T/F if initialization has finished """
        self._progress[3] = status
        return None

    def _setShutdownStarted(self,status: bool) -> None:
        """ Set T/F if initialization has started """
        self._progress[4] = status
        return None

    def _setShutdownFinished(self,status: bool) -> None:
        """ Set T/F if initialization has finished """
        self._progress[5] = status
        return None

    # Static Interface
    
"""
    Author:         Landon Buell
    Date:           May 2023
"""
