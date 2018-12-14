#!/usr/bin/python
#
# 			          Millenium
#                     Automatic Speech Recognition System
#                                   (asr)
#
#   Module:  recognize.py
#   Purpose: Recognize and dump phone lattices on 'wsj0' data.
#   Date:    May 12, 2007
#   Author:  John McDonough

import os.path
import Numeric
import pickle
import sys
from math import *

from btk.modulated import *

def designSynthesisFilter( M, m, r, v, wpFactor, h, synthesisFileName ):

    if os.path.exists( synthesisFileName ):
        print '%s exists' %( synthesisFileName )
    else:
        # Create and write synthesis prototype
        synthesis = SynthesisOversampledDFTDesignPtr(h, M = M, m = m, r = r, v = v, wpFactor = wpFactor)
        g = synthesis.design()
        err = synthesis.calcError()
        print 'g = '
        print g
        fp = open(synthesisFileName, 'w')
        pickle.dump(g, fp, True)
        fp.close()


def design( M, m, r, v, wp ):

    R    = 2**r
    D    = M / R
    wpFactor = wp * M
    print 'D = %f, wp = pi/%f' %(D, wpFactor)
    
    protoPath    = './prototypes.wp%d' %(wp)
    analysisFileName = '%s/h-M=%d-m=%d-r=%d-v=%0.6f.txt' %(protoPath, M, m, r, v)
    if not os.path.exists(os.path.dirname(analysisFileName)):
        os.makedirs(os.path.dirname(analysisFileName))

    synthesisFileName = '%s/g-M=%d-m=%d-r=%d-v=%0.6f.txt' %(protoPath, M, m, r, v)
    if not os.path.exists(os.path.dirname(synthesisFileName)):
        os.makedirs(os.path.dirname(synthesisFileName))

    if os.path.exists( analysisFileName ):
        print '%s exists' %( analysisFileName )
        fp = open(analysisFileName, 'r')
        h = pickle.load(fp)
        fp.close()
        designSynthesisFilter(M, m, r, v, wpFactor, h, synthesisFileName )
    else:
        # Create and write analysis prototype
        analysis = AnalysisOversampledDFTDesignPtr(M = M,  m = m, r = r, wpFactor = wpFactor)
        h = analysis.design()
        analysis.calcError()
        print 'h = ' 
        print h
        fp = open(analysisFileName, 'w')
        pickle.dump(h, fp, True)
        fp.close()
        designSynthesisFilter(M, m, r, v, wpFactor, h, synthesisFileName )
                

Ms  = [256,512]
ms  = [2]
rs  = [1]
wps = [1]
vs  = [1.0]

for M in Ms:
    for m in ms:
        for r in rs:
            for wp in wps:
                for v in vs:
                    print 'M=%d, m=%d, r=%d, v=%f wp=%f\n' %(M, m, r, v, wp)
                    design(M, m, r, v, wp)
