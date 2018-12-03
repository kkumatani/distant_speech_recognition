# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.globalBeamformer
#   Purpose: Global optimization of sensor weights for maximum
#            likelihood beamforming.
#   Author:  Ulrich Klee
#
#
#


from btk.mlBeamforming import *

gblBeamformer = 0

def initBeamformer(nChan, fftLen, ndim, nBlocks, sampleRate,
                   subSampRate, prototype, M, m, dspath, inDataDir,
                   vitPathDir, meanDir, cepLen = 13, gsc = 0):
    global gblBeamformer
    if gsc == 0:
        gblBeamformer = HMMSubbandBeamformer(nChan, fftLen, ndim, nBlocks,
                                             sampleRate, subSampRate, prototype,
                                             M, m, dspath, inDataDir, vitPathDir, meanDir, cepLen)
        
def calcArrayManifoldVector(delays):
    global gblBeamformer
    return gblBeamformer.calcArrayManifoldVector(delays)

def saveWeights(fileName):
    global gblBeamformer
    gblBeamformer.saveWeights(fileName)

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
