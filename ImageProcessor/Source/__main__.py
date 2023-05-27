"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        Source
    Namespace:      N/A
    File:           __main__.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os
import sys

import appConfig
import imageProcessingApplication

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Process Command Line Inputs
    # TODO

    # Initialize a Setting / Configurations Struct
    config = appConfig.AppConfig()

    # Create & Run the Image Processing App
    app = imageProcessingApplication.ImageProcessingApp(config)
    app.startup()
    app.execute()
    app.shutdown()
    
    exitStatus = app.getStatus()
    sys.exit(exitStatus)

"""
    Author:         Landon Buell
    Date:           May 2023
"""
