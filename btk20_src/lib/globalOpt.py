# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.globalOpt
#   Purpose: Global optimization of sensor weights for maximum
#            likelihood beamforming.
#   Author:  John McDonough

from btk.subbandBeamforming import *
from btk.hmmBeamforming import *

gblBeamformer = 0

def initBeamformer(spectralSources, dspath, inDataDir, vitPathDir, meanDir,
                   plotting = 0, cepLen = 13, gsc = 1):
    global gblBeamformer
    if gsc == 1:
        gblBeamformer = HMMSubbandBeamformerGSC(spectralSources, dspath,
                                                inDataDir, vitPathDir, meanDir,
                                                plotting, cepLen)
    else:
        gblBeamformer = HMMSubbandBeamformer(spectralSources, dspath,
                                             inDataDir, vitPathDir, meanDir,
                                             plotting, cepLen)
        
def setInterference(itfDir, itfName, dBItfLevel):
    global gblBeamformer
    gblBeamformer.setInterference(itfDir, itfName, dBItfLevel)

def saveWeights(fileName):
    global gblBeamformer
    gblBeamformer.saveWeights(fileName)

def calcArrayManifoldVectors(sampleRate, delays):
    global gblBeamformer
    gblBeamformer.calcArrayManifoldVectors(sampleRate, delays)

def fun(x, (utt, vitpathName, cmsName)):
    global gblBeamformer
    calcGrad = 0
    return gblBeamformer.logLhoodGrad(x, utt, vitpathName, cmsName, calcGrad)

def dfun(x, (utt, vitpathName, cmsName)):
    global gblBeamformer
    calcGrad = 1
    (f, df) = gblBeamformer.logLhoodGrad(x, utt, vitpathName, cmsName, calcGrad)
    return df

def fdfun(x, (utt, vitpathName, cmsName)):
    global gblBeamformer
    calcGrad = 1
    return gblBeamformer.logLhoodGrad(x, utt, vitpathName, cmsName, calcGrad)
