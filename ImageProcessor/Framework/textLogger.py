"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Framework
    Namespace:      N/A
    File:           TextLogger.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os

import imageProcessingApp

        #### CLASS DEFINITIONS ####

class TextLogger:
    """ 
        TextLogger all runtime Logging 
    """

    def __init__(self,
                 outputPath: str,
                 fileName: str,
                 toConsole=True,
                 toFile=True):
        """ Constructor for Logger Instance """ 
        self._outPath       = os.path.join(outputPath,fileName)
        self._outStream     = None
        self._toConsole     = toConsole
        self._toFile        = toFile
        
        if (self._toFile):
            self._outFile = open(self._outPath,"w")
        self.__writeHeader()

    def __del__(self):
        self.__writeFooter()
        """ Destructor for Logger Instance """
        if (self._outFile is not None):
            if (self._outFile.closed() == False):
                self._outFile.close()
        self._outFile = None

    @staticmethod
    def fromConfig(config):
        """ Initialize TextLogger Instance from AppConfig Instance """
        app = imageProcessingApp.ImageProcessingApp.getInstance()
        logger = TextLogger(
            outputPath=app.getConfig().getOutputPath(),
            fileName="textLogger.txt",
            toConsole=app.getConfig().getLogToConsole(),
            toFile=app.getConfig().getLogToFile())
        return logger

    # Getters and Setters

    def getOutputPath(self):
        """ Return the Path to the logger text output file """
        return self._outPath

    # Public Interface

    def logMessage(self,message:str,timeStamp=True):
        """ Log Message to Console or Text """
        if (timeStamp == True):
            # Log Message w/ a TimeStamp
            now = TextLogger.getDateTime()
        else:
            # Log Message w/o a TimeStamp
            now = ""
        formattedMessage = "\t{0:<32}\t{1}".format(now,message)

        # Write the Message to Console and/or to File
        if (self._toConsole == True):
            print(formattedMessage)

        if (self._toFile == True):
            self._outStream.write(formattedMessage + "\n")
        return None

    # Private Interface

    def __writeHeader(self):
        """ Write Header To Logger """
        header = [
            self.__spacer(),
            "ImageProcessingApp",
            TextLogger.getDateTime(),
            self.__spacer()
            ]
        # Log Each Line of the Header
        for msg in header:
            self.logMessage(msg,False)
        return self

    def __writeFooter(self):
        """ Write Footer To Logger """
        footer = [
            self.__spacer(),
            "ImageProcessingApp",
            TextLogger.getDateTime(),
            self.__spacer()
            ]
        # Log Each Line of the Header
        for msg in footer:
            self.logMessage(msg,False)
        return self

    def __spacer(self,numChars=64):
        """ Get a Spacer String """
        return "\n" + ("-" * numChars) + "\n"

    # Static Interface

    @staticmethod
    def getDatetime():
        """ Return the current Datetime as a string """
        return imageProcessingApp.ImageProcessingApp.getDateTime()



"""
    Author:         Landon Buell
    Date:           May 2023
"""