#
#                           Beamforming Toolkit
#                                  (btk)
#
#   Module:  correlate.py
#   Purpose: Correlate the beamformed chirp signal with the original
#            waveform to extract the impulse response of the room + beamformer.
#   Date:    December 2, 2005
#   Author:  John McDonough

from btk20.stream import *
from btk20.common import *
from btk20.feature import *
from btk20.convolution import *

import copy
import wave

def corr():
    M = 2048

    chirpFile    = './chirp_stetch7000.wav'
    sampleFile   = './beamformer-output.wav'
    outDataFile  = './impulse-response.wav'

    chirpFeature = SampleFeaturePtr(blockLen=M, shiftLen=M, padZeros=True)
    chirpFeature.read(chirpFile)
    impulseResponse = copy.deepcopy(chirpFeature.next(0))

    sampleFeature = SampleFeaturePtr(blockLen=M, shiftLen=M/2, padZeros=1)
    overlapAdd    = OverlapAddPtr(sampleFeature, chirpSignal, fftLen=M)
    print('\n data \n')

    sampleFeature.read(sampleFile)
    wavebuffer = []
    for fc in overlapAdd:
        print(fc)

corr()
