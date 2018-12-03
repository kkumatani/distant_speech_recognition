# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.utils
#   Purpose: Utility routines.
#   Author:  Fabian Jakobs

from __future__ import generators
from _utils import *
import time

class FGet:
    """Replicates the Janus 'fgets' functionality."""
    def __init__(self, fileName):
        """Initialize an 'FGets' object."""
        self.__fname = fileName

    def __iter__(self):
        """Iterate over a a list of speakers in a file."""
        while(1):
            tag = fgets(self.__fname)
            if tag == '':
                raise StopIteration
            yield tag

def fopen(filename, mode = 'r', bufsize = -1, trials = 100, secs = 5):
    """
    A file open routine that copes with network delays by attempting to open a file several times
    filename, mode, bufsize: see builtin function open() for description
    trials: number of attempts that will be made to open the file
    secs: number of seconds the routine sleeps until it retries
    """
    while (trials > 1):
        try:
            res = open(filename, mode, bufsize)
            return res
        except IOError:
            trials -= 1
            print "fopen(): Failed to open file %s in mode %s, will retry %d times." % (filename, mode, trials)
            time.sleep(secs)
    return open(filename, mode, bufsize)


def safeIO(func, filehandle, trials = 100, secs = 5):
    """
    A routine to cope with network delays by repeatedly attempting to perform func on filehandle

    func: function that has one single free paramter (consider using lambda form), that is a filename or descriptor
    filehandle: filename or descriptor for a function to operate on
    trials: number of attempts that will be made to perform func
    secs: number of seconds the routine sleeps until it retries
    """
    while (trials > 1):
        try:
            res = func(filehandle)
            return res
        except IOError:
            trials -= 1
            print "safeIO(): Failed to use file %s, will retry %d times." % (filehandle, trials)
            time.sleep(secs)
    return func(filehandle)

