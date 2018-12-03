#!/usr/bin/python

import os.path
import pickle
import wave
import getopt, sys
from types import FloatType

from btk.common import *
from btk.stream import *
from btk.feature import *
from btk.utils import *

from btk.modulated import *
import copy

def test( M, m, r ):

    R    = 2**r
    D    = M / R
    fftLen = M
    sampleRate = 16000
    samplePath   = 'Headset1.wav'
    filename = './wav.PR/M=%d-m=%d_oversampled.wav' %(M, m)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    print samplePath, filename

    # Read analysis prototype 'h'
    protoFile  = 'prototype_M%d_m%d' %(M, m)
    fp = open(protoFile, 'r')
    prototype = pickle.load(fp)
    fp.close()

    sampleFeature = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True)
    analysisFB  = PerfectReconstructionFFTAnalysisBankPtr( sampleFeature, prototype, M, m, r )
    synthesisFB = PerfectReconstructionFFTSynthesisBankPtr( PyVectorComplexFeatureStreamPtr(analysisFB), prototype, M, m, r )
    sampleFeature.read( samplePath, sampleRate )
    dumpeSampleN = 0
    wavebuffer  = []
    for b in synthesisFB:
        #if dumpeSampleN >= M:
        wavebuffer.extend(copy.deepcopy(b))
        dumpeSampleN += len(b)
        
    storewave = array(wavebuffer, Float)
    wavefile = wave.open(filename, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(sampleRate))
    #ampVal = 16383.0 / max( abs( storewave ) )
    #print 'multiply %f ...' %(ampVal)
    #storewave *= 3
    wavefile.setnframes(len(storewave))
    wavefile.writeframes(storewave.astype('s').tostring())
    wavefile.close()

try:
    opts, args = getopt.getopt(sys.argv[1:], "hM:m:r:", ["help", "nSubband=", "cutoff=", "decimation="])
except getopt.GetoptError:
    # print help information and exit:
    sys.exit(2)

Ms  = [256,512] #,128,256,512,1024]
ms   = [2, 4 ]
r   = 0

for M in Ms:
    for m in ms:
        test(M, m, r)
