#
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.beamforming
#   Purpose: HMM and ML subband beamforming.
#   Author:  Ulrich Klee
#
#


from __future__ import generators

import os.path

from sfe.common import *
from sfe.feature import *
from sfe.matrix import *

from btk.beamformer import *
from btk.subbandBeamforming import *
from btk.cepstralFrontend import *
from btk.modulated import *
from btk.maximumLikelihood import *

from asr.gaussian import *

from MLab import *


class HMMSubbandBeamformer(SubbandBeamformer):
    """
    Unconstrained Hidden Markov model beamformer. Provides functionality
    for performing global optimization.
    """

    def __init__(self, nChan, fftLen, ndim, nBlocks, sampleRate, subSampRate,
                 prototype, M, m, dspath, inDataDir,
                 vitPathDir, meanDir, cepLen = 13):
        """Initialize the HMM subband LMS beamformer."""

        # File paths
        self.__inDataDir    = inDataDir
        self.__vitPathDir   = vitPathDir
        self.__meanDir      = meanDir

        # Signal parameters
        self._nChan         = nChan
        self._fftLen        = fftLen
        self._fftLen2       = fftLen/2
        self._nDim          = ndim
        self._nBlocks       = nBlocks
        self._sampleRate    = sampleRate
        self._subSampRate   = subSampRate
        
        # Filter bank parameters
        self.__prototype    = prototype
        self.__m            = m
        self._M             = M
        self._snapShotArray = SnapShotArrayPtr(2*self._M, self._nChan)
        SpectralSource.__init__(self, 2*self._M, self._nBlocks, self._subSampRate)
        
        # Asr parameters 
        self.__dct          = dctMatrix
        self.__mel          = melMatrix
        print self.__mel.shape
        self.__cepLen       = cepLen
        self.__dspath       = dspath
        self.__cms          = zeros(self.__cepLen, Float)
        self.__windowLen    = 320
        self.__HammingWin   = zeros(self._M, Complex)
        self.__HammingWin[:self.__windowLen] = hamming(self.__windowLen)
        print self.__HammingWin.shape
        self._KmH     = zeros([len(self.__mel), self._M * self._M], Complex)
        # Initialize 'MaximumLikelihood' and 'CalculateLntPtr' objects
        self.__maximumLikelihood = MaximumLikelihoodPtr(prototype, M, m, self._nChan)
        self.__calculateLnt      = CalculateLntPtr(cepLen, len(self.__mel), M)
        self.__calculateLnt.calculateKmHm(self.__mel, self.__HammingWin, self._KmH)
        
        # Init beamforming weights and grad
        self._waHK          = zeros([2*self._M, self._nChan], Complex)
        self.__grad_waHK    = zeros([2*self._M, self._nChan], Complex)
        self.__gsWpast      = zeros([2*self._M, 2*self._M*self._nChan], Complex)


    def calcArrayManifoldVector(self, delays):
	"""Calculate one (conjugate) array manifold vector
        for each sub-band."""
        Delta_f = self._sampleRate / self._M

        print 'Delays: ', delays

        J = (0+1j)
        arrayManifold = zeros(4*self._M*self._nChan, Float)
        
        for fbinX in range(self._M):
            for chanX in range(self._nChan):
                arrayManifold[2*fbinX*self._nChan+chanX]   = (Numeric.exp(J*2.0*Numeric.pi*(0.5+fbinX)*Delta_f*delays[chanX])).real
            for chanX in range(self._nChan):
                arrayManifold[2*fbinX+1*self._nChan+chanX] = (Numeric.exp(J*2.0*Numeric.pi*(0.5-self._M-fbinX)*Delta_f*delays[chanX])).imag 

        return array(arrayManifold) 
        
    def logLhoodGrad(self, x, utt, vitpathName, cmsName, calcGrad = 0):
        """Calculates new weights for a given utterance"""
        
        # Unpack current weights
        for m in range(2*self._M):
            self._waHK[m] = x[(2*m*self._nChan):(2*m*self._nChan+self._nChan)] + 1j * x[(2*m*self._nChan+self._nChan):(2*(m+1)*self._nChan)]

        # Init analysisFB and process utt with it
        sampleFeats = []
        analysisFBs = []
        for i in range(self._nChan):
            sampleFeature = SampleFeaturePtr(blockLen=self._M, shiftLen=self._M, padZeros=1)
            sampleFeats.append(sampleFeature)
            analysisFB = FFTAnalysisBankFloatPtr(sampleFeature, self.__prototype,
                                                 M=self._M, m=self.__m)
            analysisFBs.append(analysisFB)
        for i in range(self._nChan):
            nextFile = ("%s/%s.%s.adc" %(self.__inDataDir, utt.file, (i+1)))
            sampleFeats[i].read(nextFile, format=SF_FORMAT_RAW|SF_FORMAT_PCM_16)

        # Init synthesisFB
        synthesisFB = FFTSynthesisBankPtr(self.__prototype, self._M, self.__m, "Test")

        dsNamePath = ("%s/%s.path" %(self.__vitPathDir, vitpathName))
        self.__dspath.read(dsNamePath)
        dsiter = self.__dspath.__iter__()
        
        cm = FVector(self.__cepLen)
        cmsPath = ("%s%s.mean" %(self.__meanDir, cmsName))
        cm.bload(cmsPath)
        
        # load mean for CMS
        for n in range(self.__cepLen):
            self.__cms[n] = cm.getItem(n)
                
        time         = 0
        firstTime    = 1
        logLhood     = 0.0

        # Init all necessary matrices and vectors for the loop
        snapshot     = zeros(2*self._M*self._nChan, Complex)
        
        Lnt          = zeros([self.__cepLen, self._M * self._M], Complex)
        LntMatrix    = zeros([self._M, self._M], Complex)
        gs_W         = zeros([2*self._M, 2*self._M*self._nChan], Complex)
        gsW_sum      = zeros([self._M, 2*self._M*self._nChan], Complex)
        product      = zeros(2 * self._M * self._nChan, Complex)
        Wl           = zeros(2 * self._M, Complex)
        cn_W          = zeros([self._M, 2 * self._M * self._nChan], Complex)
        
        if calcGrad:
            self.__grad_waHK = zeros([2*self._M, self._nChan], Complex)
            
        #while (1):
        for test in range(5):
            try:
                print "SequenceNr.:", test+1
                sigmaK = self.updateSnapShotArray(analysisFBs)

                # Calculate the frequency-domain output 'Vl' 
                Vl  = zeros(2*self._M, Complex)
                for m in range(2*self._M):
                    # Get next snapshot 
                    XK     = self.getSnapShot(m)
                    snapshot[m*self._nChan:(m+1)*self._nChan] = XK

                    # Do beamforming and calculate the array output.
                    Vl[m] = innerproduct(self._waHK[m], XK)
                    if (m == 100 and firstTime == 1):
                        norm_waK = abs(innerproduct(self._waHK[m], conjugate(self._waHK[m])))
                        print '||waHK||^2  = %8.4e' %norm_waK
                        print '  waHK      = ', abs(self._waHK[m])
                        firstTime = 0

                synthSamples = zeros(self._M, Float)
                synthesisFB.nextSample(Vl, synthSamples)
                synthSamples = self.__HammingWin * synthSamples

                
                # Calculate mel-warped frequency components 'Vtm' and
                # MFCC vector 'v'  !!! uses only upper half of x !!!
                Cm = fft(synthSamples)

                Vt = matrixmultiply(self.__mel, abs(Cm[:self._M/2+1])**2)
                v  = matrixmultiply(dctMatrix, log10(Vt))

                # get next codebook and find most likely Gaussian component
                ds = dsiter.next()
                cb = ds.cbk()
                diff = zeros(self.__cepLen, Float)

                # Do length normalization
                for n in range(self.__cepLen):
                    diff[n] = (v[n] - self.__cms[n])

                lhoodIndex  = cb.logLhood(diff)
                logLhood   += lhoodIndex.lhood()

                if calcGrad == 0:
                    continue
                
                gaussIndex  = lhoodIndex.index()
               
                self.__calculateLnt.nextSample(Vt, self._KmH, Lnt)
                self.__maximumLikelihood.nextSample(snapshot, gs_W)
                               
                gsW_sum = gs_W[self._M:] + self.__gsWpast[:self._M]
                self.__gsWpast = copy.deepcopy(gs_W)

                # Calculate weight update (equ 2.46)
                factor      = 0.0
                for n in range(self.__cepLen):
                    print "CepNr:", n
                    factor = (v[n] - cb.mean(gaussIndex, n)) * cb.invCov(gaussIndex, n)  

                    for row in range(self._M):
                        for col in range(self._M):
                            LntMatrix[row][col] = Lnt[n][row * self._M+col]
                    self.__calculateLnt.calculateCnW(LntMatrix, gsW_sum, cn_W)
                    calcProduct(synthSamples, cn_W, product)

                    for m in range(self._M):
                        self.__grad_waHK[m] += factor * product[m*self._nChan:m*self._nChan+self._nChan]

                print 'next Iteration'
                
 
            except StopIteration:
                print 'logLhood = ', logLhood

                if calcGrad == 0: 
                    return logLhood 

                grad = zeros(self._nDim, Float)
                print grad.shape
                print self.__grad_waHK.shape
                for m in range(2*self._M):
                    grad[(2*m*self._nChan):(2*m*self._nChan+self._nChan)] = self.__grad_waHK[m].real
                    grad[(2*m*self._nChan+self._nChan):(2*(m+1)*self._nChan)] = self.__grad_waHK[m].imag

                print "returning gradient"
                return (logLhood, grad)

        print 'logLhood = ', logLhood

        if calcGrad == 0: 
            return logLhood 
        
        grad = zeros(self._nDim, Float)
        print grad.shape
        print self.__grad_waHK.shape
        for m in range(2*self._M):
            grad[(2*m*self._nChan):(2*m*self._nChan+self._nChan)] = self.__grad_waHK[m].real
            grad[(2*m*self._nChan+self._nChan):(2*(m+1)*self._nChan)] = self.__grad_waHK[m].imag

        print grad.shape
        print grad[10]
        print "returning gradient"
        return (logLhood, grad)






