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

def test( M, m, r, v ):

    R    = 2**r
    D    = M / R
    fftLen = M
    sampleRate = 16000
    samplePath   = 'Headset1.wav'
    filename = './wav.deHaan/M=%d-m=%d-r=%d-v=%0.6f_oversampled.wav' %(M, m, r, v)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    print samplePath, filename
    protoPath    = 'prototypes.wp1'
    # Read analysis prototype 'h'
    protoFile  = '%s/h-M=%d-m=%d-r=%d-v=%0.6f.txt' %(protoPath, M, m, r, v)
    print 'Loading analysis prototype from \'%s\'' %protoFile
    fp = open(protoFile, 'r')
    h_fb = pickle.load(fp)
    fp.close()

    # Read synthesis prototype 'g'
    protoFile = '%s/g-M=%d-m=%d-r=%d-v=%0.6f.txt' %(protoPath, M, m, r, v)
    print 'Loading synthesis prototype from \'%s\'' %protoFile
    fp = open(protoFile, 'r')
    g_fb = pickle.load(fp)
    fp.close()

    sampleFeature = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True)
    analysisFB = OverSampledDFTAnalysisBankPtr(sampleFeature, prototype = h_fb, M = M, m = m, r = r, filterBankType=1 )
    synthesisFB = OverSampledDFTSynthesisBankPtr(PyVectorComplexFeatureStreamPtr(analysisFB), prototype = g_fb, M = M, m = m, r = r, filterBankType=1 )       
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
    storewave *= pi
    wavefile.setnframes(len(storewave))
    wavefile.writeframes(storewave.astype('s').tostring())
    wavefile.close()

try:
    opts, args = getopt.getopt(sys.argv[1:], "hM:m:r:", ["help", "nSubband=", "cutoff=", "decimation="])
except getopt.GetoptError:
    # print help information and exit:
    sys.exit(2)

Ms  = [256,512] #,128,256,512,1024]
ms  = [2]
rs  = [1,2,3,4]
v = 0.01

for M in Ms:
    for m in ms:
	for r in rs:	
	    test(M, m, r, v)
