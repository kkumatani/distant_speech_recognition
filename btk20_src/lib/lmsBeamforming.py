from __future__ import generators

from Numeric import *
from MLab import *
from FFT import *
from LinearAlgebra import *
from RandomArray import *
from btk.plot import *
from btk.sound import *
from btk.digitalFilterBank import *
from btk.subBandBeamforming import *
from btk.cepstralFrontend import *
from asr.common import *
from asr.feature import *
from asr.codebook import *
from asr.path import *
from asr.matrix import *


class SubBandBeamformerLmsHmm(SubbandBeamformerGSC):

    def __init__(self, spectralSources, plotting = 0, beta = .99, gamma = 1e6, bound = .001, taps = 1, dep = 0, bftype = 1, noisy = 0):
        """Initialize the sub-band RLS beamformer."""
        SubBandBeamformer.__init__(self, spectralSources)
        self.__plotFlag      = plotting
        self.taps = taps
        self.dep = dep
        self.noisy = noisy
        self.bftype = bftype
        self.dct = dctMatrix
        self.mel = melMatrix
        self.beta = beta        # a forgetting factor for calculation of average power in a subband beta >= 0.99
        self.gamma = gamma      # a learning rate constant for the gradient descent procedure 0.005 < gamma < 0.05
        self.bound = bound      # a bound on the magnitude of the active weight vector
        self.uttsamp = 0
        self.spksamp = 0

        # for now we place initialization of the codebook infrastructure here:
        descFile = '/project/draub/exp/aux-models01a/clusterLH/desc/codebookSet.2250.gz'
        cbkFile  = '/project/draub/exp/aux-models01a/train/Weights/5i.cbs.gz'
        self.meanDir = '/project/draub/exp/exp01a/test-means/means/'
        fs = FeatureSet()
        print 'cbs'
        cbs = CodebookSetBasicPtr(descFile = descFile, fs = fs)
        print 'load'
        cbs.load(cbkFile)
        self.cbpath = CodebookPathPtr(cbs)

    def calcNoiseManifold(self, sampleRate, delays):
        """Calculate one (conjugate) array manifold vector
        for each sub-band."""
        Delta_f = sampleRate / self._fftLen
        J = (0+1j)
        noiseManifold = []
        for fbinX in range(self._fftLen2 + 1):
            vs = zeros(self._nChan).astype(complex)
            for chanX in range(self._nChan):
                vs[chanX] = exp(J*2.0*pi*fbinX*Delta_f*delays[chanX])
            noiseManifold.append(vs);
        self.noiseManifold = array(noiseManifold)

    def __iter__(self):
        """Return the next spectral sample."""
        self.uttsamp = 0

        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        cbiter = self.cbpath.__iter__()

        u = zeros((self.taps, self._nChan, self._fftLen2 + 1), Complex)

        while (1):
            u[1:] = u[:-1]
            for i in range(self._nChan):
                u[0,i] = (analysisFBs[i].next())[:self._fftLen2 + 1]

            if self.noisy:
                noise = real_fft( normal(0, self.noisy, 320) * hamming(320), 512)
                u[0] += noise * transpose(conjugate(self.noiseManifold))

            wqH = transpose(array(self._arrayManifold[:self._fftLen2 + 1])/self._nChan)

            # Calculate output of upper branch and array output.
            # y : frontend output (mfccs)
            # v : GSC output
            if (self.bftype == 1):
                for f in range(self._fftLen2+1):
                    self.wH[0,:,f] = - matrixmultiply(self.waH[:,f],self._blockingMatrix[f])
                self.wH[0,:,:] += wqH
            elif (self.bftype == 3):
                pass
            else:
                for f in range(self._fftLen2+1):
                    self.wH[:,:,f] = - matrixmultiply(self.waH[:,:,f], self._blockingMatrix[f])
                self.wH[0,:,:] += wqH

#             problem: check that the dimensions of summation are right
            v = sum(sum( self.wH * u, 0), 0)
            y = matrixmultiply(self.dct, log10(matrixmultiply(self.mel, abs(v)**2))) # perhaps it might save time to deposit abs(v)**2 in a new variable


            # get next codebook
            cb = cbiter.next()

            if ((self.spksamp % self._subSampRate == 0) ) :#and (cb.name()[:3] != 'SIL')): # Update active weight vector.
                # obtain inverse covariance matrix and mean vector of current state
                mean = zeros(cb.featLen(), Float)
                invcov = zeros((cb.featLen(),cb.featLen()), Float)
                for i in range(cb.featLen()):
                    mean[i] = cb.mean(0,i)  # we have only one gaussian, hence the 0 for first parameter
                    invcov[i,i] = cb.invCov(0,i)

                # calculate the (only!) frequency dependent (not channel dependent or tap dependent) weighting to be applied to gradient
                M = 2 * matrixmultiply( (y - self.cms - mean) * diagonal(invcov) , matrixmultiply(self.dct, self.mel / outerproduct( matrixmultiply(self.mel, abs(v)**2), ones(self._fftLen2 + 1) ) ))
                # in case cov is non-diagonal, we need (check math again before use !!!!):
                # M = 2 * matrixmultiply( matrixmultiply(y - self.cms - mean, invcov) , matrixmultiply(self.dct, self.mel / outerproduct( matrixmultiply(self.mel, abs(v)**2), ones(self._fftLen2 + 1) ) ))                n_dwaH = M * n_dv_dwaH_T

                # calculate derivative of GSC output and subband power
                if (self.bftype == 1):
                    n_dv_dwaH_T = zeros((self._nChan-1,self._fftLen2 + 1), Complex)
                    dv_dwH_T = zeros((self.taps, self._nChan, self._fftLen2 + 1), Complex)
                    for f in range(self._fftLen2):
                        n_dv_dwaH_T[:,f] = conjugate( v[f] ) * matrixmultiply(self._blockingMatrix[f], u[0,:,f]) # negated and transposed gradient matrix of v
                        dv_dwH_T[1:,:,f]= conjugate( v[f] ) * u[1:,:,f]
                    dwH = M * dv_dwH_T
                    n_dwaH = M * n_dv_dwaH_T
                elif (self.bftype == 3):
                    dv_dwH_T = zeros((self.taps, self._nChan, self._fftLen2 + 1), Complex)
                    for f in range(self._fftLen2):
                        dv_dwH_T[:,:,f]= conjugate( v[f] ) * u[:,:,f]
                    dwH = M * dv_dwH_T
                else:
                    n_dv_dwaH_T = zeros((self.taps, self._nChan-1, self._fftLen2 + 1), Complex)
                    for f in range(self._fftLen2):
                        n_dv_dwaH_T[:,:,f] = conjugate( v[f] ) * matrixmultiply(u[:,:,f], transpose(self._blockingMatrix[f])) # negated and transposed gradient matrix of v
                        n_dwaH = M * n_dv_dwaH_T

                # there is one gradient vector waH for each frequency bin, each has length #channels-1
                # multiply each frequency column of dvT with the (only!) frequency dependent (not channel dependent) factor
                # the pointwise multiplication applies M to each frequency vector (last dimension)

                # calculate learning rates:
                if self.dep:
                    self.sigma2 = self.beta * self.sigma2 + (1 - self.beta) * sum(sum(abs(u[0])**2)) / (self._fftLen2 + 1)
                else:
                    self.sigma2 = self.beta * self.sigma2 + (1 - self.beta) * sum(abs(u[0])**2)
                xi = self.gamma / self.sigma2

                # update the weight vector:
                # before: multiply each frequency column of the gradient matrix by the appropriate frequency dependent learning rate
                # now: use an average learning rate (frequency independent
                if (self.bftype == 1):
                    self.wH = self.wH - conjugate(xi * dwH)  # now it is right I think with '-' instead of '+'
                    self.waH = self.waH + conjugate(xi * n_dwaH)
                    norm_w = max(concatenate([[sum(abs(self.waH)**2, 0)],sum(abs(self.wH[1:])**2, 1)]), 0)
                elif (self.bftype == 3):
                    self.wH = self.wH - conjugate(xi * dwH)  # now it is right I think with '-' instead of '+'
                    norm_w = max(sum(abs(self.wH)**2, 1), 0)
                else:
                    self.waH = self.waH + conjugate(xi * n_dwaH)
                    norm_w = max(sum(abs(self.waH)**2, 1), 0)

                # Still under control?
                # TAKE ANOTHER LOOK AT NORMALIZATION !!!
                if sum(norm_w > self.bound) :
                    print 'Weight vector ||wa||^2 > %2.2e at sample %d, we normalize ... ' %(self.bound, self.spksamp)
                    # normalization routine:
                    if self.dep:
                        if (self.bftype == 1):
                           self.waH = self.waH * (self.bound / max(norm_w))
                           self.wH = self.wH * (self.bound / max(norm_w))
                        elif (self.bftype == 3):
                           self.wH = self.wH * (self.bound / max(norm_w))
                        else:
                           self.waH = self.waH * (self.bound / max(norm_w))
                    else:
                        if (self.bftype == 1):
                            self.waH = self.waH / ((norm_w / self.bound)**(norm_w  > self.bound))
                            self.wH = self.wH / ((norm_w / self.bound)**(norm_w  > self.bound))
                        elif (self.bftype == 3):
                            self.wH = self.wH / ((norm_w / self.bound)**(norm_w  > self.bound))
                        else:
                            self.waH = self.waH / ((norm_w / self.bound)**(norm_w  > self.bound))


                # Dump debugging info.
                if self.spksamp % 50 == 0:
                    f = 60
                    print ''
                    print 'Speaker Sample %d' %(self.spksamp)
                    print 'Utterance Sample %d' %(self.uttsamp)
                    print '||w||^2       = %8.4e' %(norm_w[f])
#                     if self.dep:
#                         print 'sigma^2: %2.3e' % (self.sigma2)
#                     else:
                    print 'sigma^2: %2.3e' % (self.sigma2[f])
                    print 'mean: ', mean
                    print 'mfcc y-offset: ', y-self.cms
                    print 'WARNING: If you are using multitap, the plot will not accurately represent the actual response'
                    print 'waH:'
                    if (self.bftype == 1):
                        print abs(self.waH[:,f])
                        if self.__plotFlag:
                            plotBeamPattern2(wqH[:,f] - matrixmultiply(self.waH[:,f],self._blockingMatrix[f]), wqH[:,f])
                    elif (self.bftype == 3):
                        print abs(self.wH[:,:,f])
                        if self.__plotFlag:
                            plotBeamPattern2(wH[0,:,f], wqH[:,f])
                    else:
                        print abs(self.waH[:,:,f])
                        if self.__plotFlag:
                            plotBeamPattern2(self.wqH[:,f] - matrixmultiply(self.waH[0,:,f],self._blockingMatrix[f]), self.wqH[:,f])

            # Calculate complete array output.
            # output = concatenate( v, conjugate(v[len(y)-2,0,-1]) )
            # output = v

            # For now we output the MFCC since produce them anyway ...
            # this is maybe not nice, since the other beamformers output FFT values (for this use lines above)
            yield y
            self.spksamp += 1
            self.uttsamp += 1

    def nextSpkr(self, spkId):
        """Reset weight vectors for new speaker."""
        self.spksamp = 0
        # reinitialize the weight vectors for the next speaker
        self.wH = zeros((self.taps, self._nChan, self._fftLen2 + 1), Complex)
        if (self.bftype == 1):
            self.waH = zeros((self._nChan-1, self._fftLen2 + 1), Complex)
        elif (self.bftype == 3):
            # initialize with current manifold
            self.wH[0,:,:] = transpose(array(self._arrayManifold[:self._fftLen2 + 1])/self._nChan)
        else:
            self.waH = zeros((self.taps, self._nChan-1, self._fftLen2 + 1), Complex)

        # reset the power average
#         if self.dep:
#             self.sigma2 =  1E10      # exponential average of the subband powers
#         else:
        self.sigma2 =  1E10 * ones((self._fftLen2 + 1), Float)      # exponential average of the subband powers

        # get the cepstral mean for the new speaker
        v = FVector(13)
        v.bload(self.meanDir + '/' + spkId + '.mean')
        self.cms =  zeros(13, Float)    # mean for cepstral mean subtraction
        for i in range(13):
            self.cms[i] = v.getItem(i)

    def nextUtt(self, cbkNamesFile):
        """attach next viterbi path to variables"""
        self.cbpath.read(cbkNamesFile)
        self.uttsamp = 0
