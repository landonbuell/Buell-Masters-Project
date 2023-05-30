"""
    Repository:     Buell-MSES-Project
    Solution:       ImageProcessing
    Project:        CommonToolsPy
    Namespace:      N/A
    File:           simpleQueue.py
    Author:         Landon Buell
    Date:           May 2023
"""

        #### IMPORTS ####

import os
import sys

import numpy as np

        #### CLASS DEFINITIONS ####

class SimpleQueue:
    """ Simple FIFO Queue implementation through a numpy array """

    def __init__(self,
                 maxCapacity=int(2**16)):
        """ Constructor """
        self._size      = 0
        self._data      = np.empty(shape=(maxCapacity,),dtype=object)
        self._front     = -1
        self._rear      = -1

    def __del__(self):
        """ Destructor """
        self.clear()

    # Accessors

    def getSize(self) -> int:
        """ Return the size of this queue """
        return self._size

    def getCapacity(self) -> int:
        """ Return the capacity of this queue """
        return self._data.shape[0]

    def isEmpty(self) -> bool:
        """ Return T/F if this queue is empty """
        return (self._size == 0)

    def isFull(self) -> bool:
        """ Return T/F if this queue is at it's maximum capacity """
        return (self._size >= self.getCapacity() - 1)

    # Public Interface

    def enqueue(self,item) -> None:
        """ Add an item to the back of the queue """
        if (self.isFull() == True):
            msg = "{0} is full, cannot enqueue new item".format(self)
            raise RuntimeError(msg)

        if (self.isEmpty() == True):
            self._front = 0
            self._rear = 0
        else:
            self._rear = (self._rear + 1) % self.getCapacity()

        self._data[self._rear] = item
        self._size += 1
        return None

    def dequeue(self) -> None:
        """ Remove an item from the front of the queue """
        if (self.isEmpty() == True):
            msg = "{0} is empty, cannot dequeue an item".format(self)
            raise RuntimeError(msg)
        
        if (self._front == self._rear):
            self._front = 0
            self._rear  = 0
        else:
            self._front = (self._front + 1) % self.getCapacity()

        self._size -= 1 
        return None

    def front(self) -> object:
        """ Access the item at the front of the queue """
        return self._data[self._front]
    
    def clear(self):
        """ Clean all items from this queue """
        self._size = 0
        self._front = -1
        self._rear  = -1
        return self

    # Private interface


    # Magic methods

    def __len__(self) -> int:
        """ Return the size of the queue """
        return self._size

    def __bool__(self) -> bool:
        """ Return T/F if queue is non-empty """
        return (self.isEmpty() == False)

    def __str__(self) -> str:
        """ Return string representation of this instance """
        return "{0} @ {1}".format(self.__class__,hex(id(self)))


"""
    Author:         Landon Buell
    Date:           May 2023
"""