"""
Collection of functions required for Bbox path counting algorithm
"""
import numpy as np


class Stats:
    def __init__(self, door_count):
        """
        Create a counter for doors using numpy
        """
        self.Doors = np.zeros(int(door_count)).astype(np.int32)