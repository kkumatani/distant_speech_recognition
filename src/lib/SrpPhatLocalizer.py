# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.SrpPhatLocalizer
#   Purpose: Source localization.
#   Author:  Ulrich Klee
# 
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or (at
#  your option) any later version.
#  
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.


from btk.utils import *
from btk import dbase
from btk.localization import *
from btk.cepstralFrontend import *

from Numeric import *

class SrpPhatLocalizer:
    """
    Class to perform SourceLocalization
    """
    
    def __init__(self, arrgeom, fftLen=512, nBlocks=4, subSampRate=2, sampleRate=16000.0, beta=0.06):
       
        self.arrgeom = arrgeom
    
        # Specify filter bank parameters.
        self.nChan       = len(arrgeom)
        self.fftLen      = fftLen
        self.nBlocks     = nBlocks
        self.subSampRate = subSampRate
        self.sampleRate  = sampleRate
        self.Delta_f     = sampleRate / fftLen
        self.beta        = beta
        self.q           = [2000.0, 0.0, 0.0]
        

        # Build analysis chain
        self.analysisFBs = []
        for i in range(self.nChan):
            self.analysisFBs.append(AnalysisFB(self.fftLen, self.nBlocks, self.subSampRate))

    def __iter__(self):
        pass

    def getPostitionEstimation(self, spectralPerChan, sampleNr, searchRange = [-800.0, 800.0, 20.0]):
        """Return PositionEstimation for given soundSource"""       
        
        # Try to localize speaker
        argMax   = 0.0
        maxPos   = -10000
        
        mFramePerChan = []

        for l in range(self.nChan):
            help = spectralPerChan[l]
            mFramePerChan.append(help[sampleNr])

        mFramePerChan = array(mFramePerChan)
        #print mFramePerChan.shape
        
        for position in range(searchRange[0],searchRange[1],searchRange[2]):
            # print "Trying position %d" %position
            self.q[1] = position
            delays     = calcDelays(self.q[0], self.q[1], self.q[2], self.arrgeom)
            
            sum = getSrpPhat(delays, self.Delta_f, mFramePerChan)
            
            if abs(sum) > argMax:
                argMax = abs(sum)
                maxPos = position

        positionEst = array([self.q[0], maxPos, self.q[2]])
        return positionEst
            
