#
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.hmmBeamforming
#   Purpose: HMM subband beamforming.
#   Author:  John McDonough
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


from __future__ import generators

import os.path

from btk.common import *
from btk.feature import *

from btk.subbandBeamforming import *
from btk.cepstralFrontend import *
from btk.squareRoot import *
from btk.beamformer import *

from asr.gaussian import *
from btk.matrix import *

class HMMSubbandBeamformerGriffithsJim(SubbandBeamformerGSC):

    def __init__(self, spectralSources, dspath, plotting = 0, beta = .99, gamma = 1e6, alpha2 = 1.0E-04, cepLen = 13, diagLoad = 1.0E+10 ):
        """Initialize the HMM sub-band LMS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources)
        self.__plotFlag = plotting
        self.__dct      = dctMatrix
        self.__mel      = melMatrix
        self.__beta     = beta
        self.__gamma    = gamma
        self.__alpha2   = alpha2
        self.__cepLen   = cepLen
        self.__dspath   = dspath
        self.__cms      = zeros(self.__cepLen, Float)
        self.__initDiagLoad = diagLoad
        self.__floorSubBandSigmaK = 1.0E+08

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        dsiter = self.__dspath.__iter__()

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs)

            # Calculate the frequency-domain output 'Vl' of the GSC
            Vl = zeros(self._fftLen2+1, Complex)
            for i in range(1,self._fftLen2+1):
                # Get next snapshot and form output of blocking matrix.
                XK = self.getSnapShot(i)
                ZK = matrixmultiply(self._blockingMatrix[i], XK)
                self.__ZK[i] = copy.deepcopy(ZK)

                # Calculate output of upper branch.
                wqH = self._arrayManifold[i]/self._nChan
                YcK = innerproduct(wqH,XK)

                # Calculate complete array output.
                Vl[i] = YcK - innerproduct(self.__waHK[i], ZK)

            # Calculate mel-warped frequency components 'Vtm' and
            # MFCC vector 'v'
            Vt = matrixmultiply(self.__mel, abs(Vl)**2)
            v  = matrixmultiply(self.__dct, log10(Vt))

            assert(len(v) == self.__cepLen)

            # get next codebook
            ds = dsiter.next()
            nm = ds.name()[0:3]
            cb = ds.cbk()

            if (self.__isamp % self._subSampRate == 0) and (nm == 'SIL'):
                print 'SIL: Skipping update on frame %d' %(self.__isamp)

            # Update active weight vector.
            if (self.__isamp % self._subSampRate == 0) and (nm != 'SIL'):

                # Calculate 'rho_nm(t)'
                rho = matrixmultiply(matrixmultiply(self.__dct, diag(1.0 / Vt)), self.__mel)

                # Calculate 'nu_m(t)'
                diff = zeros(self.__cepLen, Float)
                for n in range(self.__cepLen):
                    diff[n] = ( v[n] - self.__cms[n] )
                gaussIndex = ds.logLhood(diff).index()
                for n in range(self.__cepLen):
                    diff[n] = (diff[n] - cb.mean(gaussIndex,n)) * cb.invCov(gaussIndex,n)
                nu = matrixmultiply(diff, rho)

                for i in range(1,self._fftLen2+1):
                    # Get next snapshot and form output of blocking matrix.
                    XK = self.getSnapShot(i)
                    ZK = self.__ZK[i]

                    epa           = Vl[i]
                    subBandSigmaK = self.__subBandSigmaK[i] * self.__beta + (1.0-self.__beta)*abs(innerproduct(conjugate(ZK), ZK))

                    if subBandSigmaK < self.__floorSubBandSigmaK:
                        subBandSigmaK = self.__floorSubBandSigmaK

                    alphaK = self.__gamma / subBandSigmaK
                    watHK  = self.__waHK[i] + alphaK * conjugate(nu[i]) * conjugate(ZK) * epa

                    norm_watK = abs(innerproduct(watHK, conjugate(watHK)))
                    if norm_watK > self.__alpha2:
                        cK = sqrt(self.__alpha2 / norm_watK)
                        waHK = cK * watHK
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if i == 100 and self.__isamp % 50 == 0:
                        print ''
                        print 'Sample %d' %(self.__isamp)
                        print 'Sub-Band SigmaK = %8.4e' %(subBandSigmaK)
                        print 'SigmaK          = %8.4e' %(sigmaK)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        norm_waK = abs(innerproduct(waHK, conjugate(waHK)))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)
                        if self.__plotFlag:
                            from btk.plot import *
                            wqH = self._arrayManifold[i]/self._nChan
                            plotBeamPattern(wqH - matrixmultiply(waHK,self._blockingMatrix[i]))

                    # Store values for next iteration
                    self.__waHK[i]          = copy.deepcopy(waHK)
                    self.__subBandSigmaK[i] = subBandSigmaK

            # Update the average power.
            self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta)*sigmaK

            yield v
            self.__isamp += 1

    def nextSpkr(self, spkId, meanDir):
        """Reset weight vectors for new speaker."""
        self.__sigmaK        = self.__initDiagLoad
        self.__isamp         = 0
        self.__waHK          = []
        self.__subBandSigmaK = []
        for i in range(self._fftLen2+1):
            self.__waHK.append(zeros(self._nChan-1, Complex))
            self.__subBandSigmaK.append(self.__initDiagLoad)

        cm = FVector(self.__cepLen)
        cm.bload(meanDir + '/' + spkId + '.mean')

        # load mean for CMS
        for i in range(self.__cepLen):
            self.__cms[i] = cm.getItem(i)

    def nextUtt(self, cbkNamesFile):
        """attach next viterbi path to variables"""
        self.__dspath.read(cbkNamesFile)
        self.__ZK = []
        for i in range(self._fftLen2+1):
            self.__ZK.append(zeros(self._nChan-1, Complex))


class HMMSubbandBeamformerRLS(SubbandBeamformerGSC):

    def __init__(self, spectralSources, cbpath, plotting = False, beta = 0.98, alpha2 = 1.0E-04, cepLen = 13, diagLoad = 1.0E+10, lam = 0.95, bof = 1000 ):
        """Initialize the HMM sub-band LMS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources)
        self.__plotFlag      = plotting
        self.__dct           = dctMatrix
        self.__mel           = melMatrix
        self.__alpha2        = alpha2
        self.__cepLen        = cepLen
        self.__cbpath        = cbpath
        self.__lambdaInv     = 1.0 / lam
        self.__cms           = zeros(self.__cepLen, Float)
        self.__initDiagLoad  = diagLoad
        self.__beta          = beta
        self.__bailOutFactor = bof

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        cbiter = self.__cbpath.iterator()

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs)

            # Calculate the frequency-domain output 'Vl' of the GSC
            Vl = zeros(self._fftLen2+1, Complex)
            for m in range(1,self._fftLen2+1):
                # Get next snapshot and form output of blocking matrix.
                XK           = self.getSnapShot(m)
                ZK           = matrixmultiply(self._blockingMatrix[m], XK)
                self.__ZK[m] = copy.deepcopy(ZK)

                # Calculate output of upper branch.
                wqH = self._arrayManifold[m]/self._nChan
                YcK = innerproduct(wqH,XK)

                # Calculate complete array output.
                Vl[m] = YcK - innerproduct(self.__waHK[m], ZK)

            # Calculate mel-warped frequency components 'Vtm' and
            # MFCC vector 'v'
            Vt = matrixmultiply(self.__mel, abs(Vl)**2)
            v  = matrixmultiply(self.__dct, log10(Vt))

            assert(len(v) == self.__cepLen)

            # get next codebook
            cb = cbiter.next()
            nm = cb.name()[0:3]
            
            if (self.__isamp % self._subSampRate == 0) and (nm == 'SIL'):
                print 'SIL: Skipping update on frame %d' %(self.__isamp)

            # Update active weight vector.
            if (self.__isamp % self._subSampRate == 0) and (nm != 'SIL'):

                # Calculate 'Crls_nm(t)'
                Crls = - matrixmultiply(matrixmultiply(matrixmultiply(self.__dct, diag(1.0 / Vt)), self.__mel), diag(Vl))

                # Calculate intermediate result needed for Kalman Gain.
                alpha = zeros(self._fftLen2+1, Complex)
                for m in range(1,self._fftLen2+1):
                    ZK       = self.__ZK[m]
                    alpha[m] = innerproduct(conjugate(ZK), matrixmultiply(self.__PzK[m], ZK))

                # Calculate vector describing contribution of HMM
                diff = zeros(self.__cepLen, Float)
                for n in range(self.__cepLen):
                    diff[n] = ( v[n] - self.__cms[n] )
                gaussIndex = cb.logLhood(diff).index()
                for n in range(self.__cepLen):
                    diff[n] = cb.mean(gaussIndex, n) - diff[n]

                # Calculate 'invD'
                D = zeros((self.__cepLen, self.__cepLen), Float)
                vectorMatrixProduct(alpha, Crls, D)
                D *= self.__lambdaInv
                for n in range(self.__cepLen):
                    D[n][n] += 1.0 / cb.invCov(gaussIndex,n)
                invD = inverse(D)
                
                for m in range(1,self._fftLen2+1):
                    temp = outerproduct(matrixmultiply(self.__PzK[m], self.__ZK[m]), conjugate(Crls[:,m]))

                    # Calculate Kalman Gain
                    Gm   = self.__lambdaInv * matrixmultiply(temp, invD)

                    # Update State Error Correlation Matrix
                    PzK  = self.__lambdaInv * (self.__PzK[m] -
                                               outerproduct(matrixmultiply(Gm, Crls[:,m]),
                                                            matrixmultiply(conjugate(self.__ZK[m]), self.__PzK[m])))

                    # Enforce conjugate symmetry
                    makeConjugateSymmetric(PzK)

                    # Update active weight vector.
                    watHK = self.__waHK[m] + conjugate(matrixmultiply(Gm, diff))
                    watK  = conjugate(watHK)

                    # Still in control? Apply quadratic constraint.
                    betaK = 0.0
                    norm_watK = abs(innerproduct(watHK, watK))
                    if norm_watK > self.__bailOutFactor * self.__alpha2:
                        print 'Bailing out at sample %d' %(self.__isamp)
                        waHK = self.__waHK[m]
                        PzK  = identity(self._nChan-1)/self.__initDiagLoad
                    elif norm_watK > self.__alpha2:
                        print 'Bounding ....'
                        va   = matrixmultiply(PzK, watK)
                        a    = innerproduct(va, conjugate(va)).real
                        b    = -2.0 * (innerproduct(conjugate(va), watK)).real
                        if (a < 0.0) or (b > 0.0):
                            print "Shit! This f*cking thing doesn't work!"
                        c    = norm_watK - self.__alpha2
                        arg  = b*b - 4.0*a*c
                        if arg > 0:
                            betaK = - (b + sqrt(arg)) / (2.0 * a)
                        else:
                            betaK = - b / (2.0 * a)
                        waHK = watHK - betaK * conjugate(va)
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if m == 100 and self.__isamp % 5 == 0:
                        print ''
                        #print 'D = '
                        #print D
                        print 'Pm_inv = '
                        print inverse(PzK)
                        print 'Sample %d' %(self.__isamp)
                        print 'SigmaK          = %8.4e' %(sigmaK)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        print '||Z^H P_z Z||^2 = %8.4e' %(abs(alpha[m]))
                        print 'betaK           = %8.4e' %(betaK)
                        print '||watK||^2      = %8.4e' %(norm_watK)
                        norm_waK = abs(innerproduct(conjugate(waHK), waHK))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)
                        if self.__plotFlag:
                            from btk.plot import *
                            wqH = self._arrayManifold[m]/self._nChan
                            plotBeamPattern(wqH - matrixmultiply(waHK,self._blockingMatrix[m]))

                    # Store values for next iteration
                    self.__PzK[m]  = copy.deepcopy(PzK)
                    self.__waHK[m] = copy.deepcopy(waHK)

            # Update the average power.
            self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta)*sigmaK

            yield v
            self.__isamp += 1

    def nextSpkr(self, spkId, meanDir):
        """Reset weight vectors for new speaker."""
        self.__sigmaK     = self.__initDiagLoad
        self.__isamp      = 0
        self.__PzK        = []
        self.__waHK       = []
        for i in range(self._fftLen2+1):
            self.__PzK.append(copy.deepcopy(identity(self._nChan-1,Complex)/self.__initDiagLoad))
            self.__waHK.append(zeros(self._nChan-1, Complex))

        cm = FVector(self.__cepLen)
        cm.bload(meanDir + '/' + spkId + '.mean')

        # load mean for CMS
        for n in range(self.__cepLen):
            self.__cms[n] = cm.getItem(n)

    def nextUtt(self, cbkNamesFile):
        """attach next viterbi path to variables"""
        self.__cbpath.read(cbkNamesFile)
        self.__ZK = []
        for m in range(self._fftLen2+1):
            self.__ZK.append(zeros(self._nChan-1, Complex))

class HMMSubbandBeamformerSqRtCovar(SubbandBeamformerGSC):

    def __init__(self, spectralSources, cbpath, plotting = False, beta = 0.98, alpha2 = 1.0E-04, cepLen = 13, diagLoad = 1.0E+10, lam = 0.95 ):
        """Initialize the HMM sub-band LMS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources)
        self.__plotFlag       = plotting
        self.__dct            = dctMatrix
        self.__mel            = melMatrix
        self.__alpha2         = alpha2
        self.__cepLen         = cepLen
        self.__cbpath         = cbpath
        self.__sqrt_lambdaInv = 1.0 / sqrt(lam)
        self.__cms            = zeros(self.__cepLen, Float)
        self.__initDiagLoad   = diagLoad
        self.__beta           = beta

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        cbiter = self.__cbpath.iterator()

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs)

            # Calculate the frequency-domain output 'Vl' of the GSC
            Vl = zeros(self._fftLen2+1, Complex)
            for m in range(1,self._fftLen2+1):
                # Get next snapshot and form output of blocking matrix.
                XK           = self.getSnapShot(m)
                ZK           = matrixmultiply(self._blockingMatrix[m], XK)
                self.__ZK[m] = copy.deepcopy(ZK)

                # Calculate output of upper branch.
                wqH = self._arrayManifold[m]/self._nChan
                YcK = innerproduct(wqH,XK)

                # Calculate complete array output.
                Vl[m] = YcK - innerproduct(self.__waHK[m], ZK)

            # Calculate mel-warped frequency components 'Vtm' and
            # MFCC vector 'v'
            Vt = matrixmultiply(self.__mel, abs(Vl)**2)
            v  = matrixmultiply(self.__dct, log10(Vt))

            assert(len(v) == self.__cepLen)

            # get next codebook
            cb = cbiter.next()
            nm = cb.name()[0:3]
            
            if (self.__isamp % self._subSampRate == 0) and (nm == 'SIL'):
                print 'SIL: Skipping update on frame %d' %(self.__isamp)

            # Update active weight vector.
            if (self.__isamp % self._subSampRate == 0) and (nm != 'SIL'):

                # Normalize cepstral feature, then calculate log-likelihood
                # and most likely Gaussian
                innovation = zeros(self.__cepLen, Float)
                for n in range(self.__cepLen):
                    innovation[n] = ( v[n] - self.__cms[n] )
                lhoodIndex       = cb.logLhood(innovation)
                gaussIndex       = lhoodIndex.index()
                self.__logLhood += lhoodIndex.lhood()

                # calculate the innovation
                for n in range(self.__cepLen):
                    innovation[n] = cb.mean(gaussIndex, n) - innovation[n]

                # Calculate 'Crls_nm(t)'
                Crls = - matrixmultiply(matrixmultiply(matrixmultiply(self.__dct, diag(1.0 / Vt)), self.__mel), diag(Vl))

                alpha = zeros(self._fftLen2+1, Complex)
                for m in range(1,self._fftLen2+1):
                    ZK           = self.__ZK[m]
                    ZHK_sqrt_PzK = matrixmultiply(conjugate(self.__ZK[m]), self.__sqrt_PzK[m])
                    alpha[m]     = innerproduct(ZHK_sqrt_PzK, conjugate(ZHK_sqrt_PzK))
                    self.__ZHK_sqrt_PzK[m] = copy.deepcopy(ZHK_sqrt_PzK)

                # Calculate 'sqrt_D'
                D = zeros((self.__cepLen, self.__cepLen), Float)
                vectorMatrixProduct(alpha, Crls, D)
                D *= (self.__sqrt_lambdaInv)**2
                for n in range(self.__cepLen):
                    D[n][n] += 1.0 / cb.invCov(gaussIndex,n)
                sqrt_D = cholesky_decomposition(D)

                # Perform back-substitution on the innovation
                choleskyForwardSubComplex(sqrt_D, innovation)

                for m in range(1,self._fftLen2+1):
                    # Propagate 'sqrt_Pm'
                    c_m  = Crls[:,m]
                    A_12 = outerproduct(c_m, self.__sqrt_lambdaInv * self.__ZHK_sqrt_PzK[m])
                    sqrt_Dm = asarray(sqrt_D, Complex)
                    alpha_m = abs(alpha[m])*(self.__sqrt_lambdaInv)**2

                    # This procedure can succumb to round-off error.
                    # Must be ready to punt on the update.
                    try:
                        rankOneUpdateCholeskyFactor(sqrt_Dm, alpha_m, c_m)
                    except ArithmeticError:
                        continue

                    self.__sqrt_PzK[m] *= self.__sqrt_lambdaInv
                    Gm_sqrt_D = zeros((self._nChan-1, self.__cepLen), Complex)
                    propagateCovarSquareRoot(sqrt_Dm, A_12, Gm_sqrt_D, self.__sqrt_PzK[m])

                    # Update the active weight vector.
                    watHK = self.__waHK[m] + conjugate(matrixmultiply(Gm_sqrt_D, innovation))
                    watK  = conjugate(watHK)

                    # Apply quadratic constraint.
                    betaK = 0.0
                    norm_watK = abs(innerproduct(watHK, watK))
                    if norm_watK > self.__alpha2:
                        va   = matrixmultiply(self.__sqrt_PzK[m], matrixmultiply(watK, conjugate(self.__sqrt_PzK[m])))
                        a    = innerproduct(va, conjugate(va)).real
                        b    = -2.0 * (innerproduct(conjugate(va), watK)).real
                        if (a < 0.0) or (b > 0.0):
                            print "Shit! This f*cking thing doesn't work!"
                        c    = norm_watK - self.__alpha2
                        arg  = b*b - 4.0*a*c
                        if arg > 0:
                            betaK = - (b + sqrt(arg)) / (2.0 * a)
                        else:
                            betaK = - b / (2.0 * a)
                        waHK = watHK - betaK * conjugate(va)
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if m == 100 and self.__isamp % 50 == 0:
                        print ''
                        #print 'D = '
                        #print matrixmultiply(sqrt_Dm, conjugate(transpose(sqrt_Dm)))
                        #print 'Pm = '
                        #print matrixmultiply(self.__sqrt_PzK[m], conjugate(transpose(self.__sqrt_PzK[m])))
                        print 'Sample %d' %(self.__isamp)
                        print 'SigmaK          = %8.4e' %(sigmaK)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        print '||Z^H P_z Z||^2 = %8.4e' %(abs(alpha[m]))
                        print 'betaK           = %8.4e' %(betaK)
                        print '||watK||^2      = %8.4e' %(norm_watK)
                        norm_waK = abs(innerproduct(conjugate(waHK), waHK))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)
                        if self.__plotFlag:
                            from btk.plot import *
                            wqH = self._arrayManifold[m]/self._nChan
                            plotBeamPattern(wqH - matrixmultiply(waHK,self._blockingMatrix[m]))

                    # Store values for next iteration
                    self.__waHK[m]     = copy.deepcopy(waHK)

            # Update the average power.
            self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta)*sigmaK

            yield v
            self.__isamp += 1

    def nextSpkr(self, spkId, meanDir):
        """Reset weight vectors for new speaker."""
        self.__sigmaK     = self.__initDiagLoad
        self.__isamp      = 0
        self.__sqrt_PzK   = []
        self.__waHK       = []
        self.__logLhood   = 0.0
        for i in range(self._fftLen2+1):
            self.__sqrt_PzK.append(copy.deepcopy(identity(self._nChan-1,Complex)/sqrt(self.__initDiagLoad)))
            self.__waHK.append(zeros(self._nChan-1, Complex))

        cm = FVector(self.__cepLen)
        cm.bload(meanDir + '/' + spkId + '.mean')

        # load mean for CMS
        for n in range(self.__cepLen):
            self.__cms[n] = cm.getItem(n)

    def nextUtt(self, cbkNamesFile):
        if self.__logLhood > 0.0:
            print 'Utterance %12s : LogLhood = %g' %(os.path.basename(self.__cbNames), self.__logLhood)
            self.__logLhood = 0.0
        """attach next viterbi path to variables"""
        self.__cbNames      = cbkNamesFile
        self.__cbpath.read(cbkNamesFile)
        self.__ZK           = []
        self.__ZHK_sqrt_PzK = []
        for m in range(self._fftLen2+1):
            self.__ZK.append(zeros(self._nChan-1, Complex))
            self.__ZHK_sqrt_PzK.append(zeros(self._nChan-1, Complex))


class HMMSubbandBeamformerSqRtInfo(SubbandBeamformerGSC):
    """
    Hidden Markov model RLS beamformer implemented in generalized
    sidelobe canceller (GSC) configuration.  The Cholesky decomposition
    (i.e., square-root) of the information matrix is propagated with
    every iteration. The diagonal loading of the information matrix is
    recursively updated.
    """
    def __init__(self, spectralSources, dspath, plotting = False, beta = 0.98,
                 cepLen = 13, initPower = 1.0E+08, lam = 0.95,
                 subBandLoad = 0, dBLoadLevel = -20.0, updateCepMean = 0,
                 nLocalItns = 5, staticAfter = 500):
        """Initialize the HMM sub-band LMS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources)
        self.__plotFlag         = plotting
        self.__dct              = dctMatrix
        self.__mel              = melMatrix
        self.__cepLen           = cepLen
        self.__dspath           = dspath
        self.__init_sqrt_lambda = sqrt(lam)
        self.__cms              = zeros(self.__cepLen, Float)
        self.__beta             = beta
        self.__subBandLoad      = subBandLoad
        self.__loadLevel        = 10.0**(dBLoadLevel / 10.0)
        self.__initPower        = initPower
        self.__initDiagLoad     = self.__initPower * self.__loadLevel
        self.__updateCepMean    = updateCepMean
        self.__init_nLocalItns  = nLocalItns
        self.__staticAfter      = staticAfter

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        dsiter = self.__dspath.__iter__()

        while True:
            self.updateSnapShotArray(analysisFBs, chanX = 0)

            # get next distribution
            ds = dsiter.next()
            nm = ds.name()[0:3]
            cb = ds.cbk()

            if self.__isamp == self.__staticAfter:
                print 'Setting lambda = 1.0 after sample %d' %(self.__staticAfter)
                self.__nLocalItns  = 1
                self.__sqrt_lambda = 1.0

            for m in range(1,self._fftLen2+1):
                self.__psiHK[m] = copy.deepcopy(self.__waHK[m])

            for i in range(self.__nLocalItns):

                # Calculate the frequency-domain output 'Vl' of the GSC
                Vl = zeros(self._fftLen2+1, Complex)
                for m in range(1,self._fftLen2+1):
                    # Get next snapshot and form output of blocking matrix.
                    XK           = self.getSnapShot(m)
                    ZK           = matrixmultiply(self._blockingMatrix[m], XK)
                    self.__ZK[m] = copy.deepcopy(ZK)

                    # Calculate output of upper branch.
                    wqH = self._arrayManifold[m]/self._nChan
                    YcK = innerproduct(wqH,XK)

                    # Calculate complete array output.
                    Vl[m] = YcK - innerproduct(self.__psiHK[m] / self.__sqrt_lambda, ZK)

                # Calculate mel-warped frequency components 'Vtm' and
                # MFCC vector 'v'
                Vt = matrixmultiply(self.__mel, abs(Vl)**2)
                v  = matrixmultiply(self.__dct, log10(Vt))

                assert(len(v) == self.__cepLen)

                # Normalize cepstral feature, then calculate log-likelihood
                # and most likely Gaussian
                innovation = v - self.__cms
                if i == 0:
                    lhoodIndex       = ds.logLhood(innovation)
                    gaussIndex       = lhoodIndex.index()
                    self.__logLhood += lhoodIndex.lhood()

                if (self.__isamp % self._subSampRate == 0) and (nm == 'SIL'):
                    print 'SIL: Skipping update on frame %d' %(self.__isamp)

                if (self.__isamp % self._subSampRate != 0) or (nm == 'SIL'):
                    break

                # Calculate 'Crls_nm(t)'
                Crls = - matrixmultiply(matrixmultiply(matrixmultiply(self.__dct, diag(1.0 / Vt)), self.__mel), diag(Vl))

                # calculate 'mu(t)'
                # Note: Rewrite this in 'C'.
                for n in range(self.__cepLen):
                    innovation[n] = cb.mean(gaussIndex, n) - innovation[n]

                innovation += (Crls[:,0] * conjugate(innerproduct(self.__psiHK[0],self.__ZK[0]))).real
                for m in range(1,self._fftLen2):
                    innovation += 2.0 * (Crls[:,m] * conjugate(innerproduct(self.__psiHK[m],self.__ZK[m]))).real
                innovation += (Crls[:,self._fftLen2] * conjugate(innerproduct(self.__psiHK[self._fftLen2],self.__ZK[self._fftLen2]))).real

                # Calculate 'sqrt_Q_inv'
                sqrt_Q_inv = zeros(self.__cepLen, Float)
                for n in range(self.__cepLen):
                    sqrt_Q_inv[n] = sqrt(cb.invCov(gaussIndex,n))
                innovation *= sqrt_Q_inv

                sigmaK = 0.0
                largest_norm_waK   = 0.0
                largest_norm_waK_m = 0
                for m in range(1,self._fftLen2+1):
                    # Propagate 'sqrt_Pm_inv'
                    cH_m = conjugate(Crls[:,m]) * sqrt_Q_inv
                    A_12 = outerproduct(self.__ZK[m], cH_m)
                    a_22 = asarray(innovation, Complex)

                    sqrt_PzK_inv = self.__sqrt_PzK_inv[m] * self.__sqrt_lambda
                    a_21 = matrixmultiply(self.__waHK[m], sqrt_PzK_inv)

                    propagateInfoSquareRoot(sqrt_PzK_inv, A_12, a_21, a_22, rankOneA12 = 1)

                    # Back substitute to calculate 'watK(t)'
                    choleskyBackSubComplex(sqrt_PzK_inv, a_21, self.__psiHK[m])

                    psiHK = self.__psiHK[m]
                    norm_waK = abs(innerproduct(conjugate(psiHK), psiHK))

                    if m == -1:
                        print ''
                        print 'Local Iteration %d:' %(i)
                        print '||psiK||^2 = %8.4e'  %(norm_waK)
                        print 'psiK:'
                        print abs(psiHK)

                    if i == (self.__nLocalItns-1):
                        if norm_waK > largest_norm_waK:
                            largest_norm_waK   = norm_waK
                            largest_norm_waK_m = m

                        # Copy weight vector
                        self.__waHK[m] = copy.deepcopy(psiHK)

                        # Copy Cholesky factor of information matrix
                        self.__sqrt_PzK_inv[m] = copy.deepcopy(sqrt_PzK_inv)

                        # Refresh diagonal loading
                        load = 0.0
                        if self.__subBandLoad:
                            load = self.__sub_sigmaK[m]
                        else:
                            load = self.__sigmaK
                        load *= self.__loadLevel

                        if self.__sqrt_lambda < 1.0:
                            load *= (1.0 - (self.__sqrt_lambda)**(2.0*(self._nChan-1)))
                        else:
                            load *= (self._nChan-1)
                            
                        addDiagonalLoading(self.__sqrt_PzK_inv[m], self.__loadIndex % (self._nChan-1), sqrt(load))

                        # Update subband power
                        subBandPower = (innerproduct(conjugate(self.__ZK[m]), self.__ZK[m])).real / (self._nChan-1)
                        self.__sub_sigmaK[m] = self.__sub_sigmaK[m] * self.__beta + (1.0-self.__beta)*subBandPower
                        sigmaK += subBandPower

            if (self.__isamp % self._subSampRate == 0) and (nm != 'SIL'):
                # Dump largest weight vector
                if self.__isamp % 10 == 0:
                    print ''
                    print 'Sample %d' %(self.__isamp)
                    print 'Avg. SigmaK       = %8.4e' %(self.__sigmaK)
                    print 'Subband SigmaK    = %8.4e' %(self.__sub_sigmaK[largest_norm_waK_m])
                    print 'largest ||waK||^2 = %8.4e' %(largest_norm_waK)
                    print 'largest m         = %d'    %(largest_norm_waK_m)
                    print 'largest waK:'
                    print abs(self.__waHK[largest_norm_waK_m])
                    #print 'cms:'
                    #print array2string(self.__cms, precision = 4, suppress_small = 1)
                    if self.__plotFlag:
                        from btk.plot import *
                        wqH = self._arrayManifold[largest_norm_waK_m]/self._nChan
                        plotBeamPattern2(wqH, wqH - matrixmultiply(self.__waHK[largest_norm_waK_m], self._blockingMatrix[largest_norm_waK_m]))

                self.__loadIndex += 1

                # Update average power.
                self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta) * (sigmaK/(self._fftLen2+1))

                # Update cepstral mean
                if self.__updateCepMean:
                    self.__cms = self.__cms * 0.995 + (1.0-0.995) * v

            yield v
            self.__isamp += 1

    def nextSpkr(self, spkId, meanDir):
        """Reset weight vectors for new speaker."""
        self.__sqrt_lambda  = self.__init_sqrt_lambda
        self.__nLocalItns   = self.__init_nLocalItns
        self.__sigmaK       = self.__initPower
        self.__sub_sigmaK   = []
        self.__isamp        = 0
        self.__sqrt_PzK_inv = []
        self.__waHK         = []
        self.__psiHK        = []
        self.__logLhood     = 0.0
        self.__loadIndex    = 0
        for m in range(self._fftLen2+1):
            self.__sqrt_PzK_inv.append(copy.deepcopy(identity(self._nChan-1,Complex)*sqrt(self.__initDiagLoad)))
            self.__waHK.append(zeros(self._nChan-1, Complex))
            self.__psiHK.append(zeros(self._nChan-1, Complex))
            self.__sub_sigmaK.append(self.__initPower)

        cm = FVector(self.__cepLen)
        cm.bload(meanDir + '/' + spkId + '.mean')

        # load mean for CMS
        for n in range(self.__cepLen):
            self.__cms[n] = cm.getItem(n)

    def nextUtt(self, dsNamesFile):
        if self.__logLhood > 0.0:
            print 'Utterance %12s : LogLhood = %g' %(os.path.basename(self.__dsNames), self.__logLhood)
            self.__logLhood = 0.0
        """attach next viterbi path to variables"""
        self.__dsNames      = dsNamesFile
        self.__dspath.read(dsNamesFile)
        self.__ZK           = []
        for m in range(self._fftLen2+1):
            self.__ZK.append(zeros(self._nChan-1, Complex))


class HammingSpectra(SpectralSource):
    """
    Apply a 'Hamming window' to a spectral sample.
    """
    def __init__(self, specSrc, windowLen = 320, fftLen = 512):
        """Initialize the sub-band beamformer."""
        assert (windowLen <= specSrc._fftLen)
        SpectralSource.__init__(self, specSrc._fftLen, specSrc._nBlocks,
                                specSrc._subSampRate)
        self._spectralSource = specSrc
        self._windowLen      = windowLen
        self._windowLen2     = windowLen / 2
        self._outFFTLen      = fftLen

        # Set up time-domain window
        self._hamming = zeros(self._fftLen, Float)
        window        = zeros(self._windowLen2, Float)
        window[0]     = 1.0
        for n in range(1, self._windowLen2):
            val  = (0.54 + 0.46 * cos (pi * n / (self._windowLen2)))
            val /= sin (pi * n / self._fftLen) / (pi * n / self._fftLen)
            val /= (0.54 + 0.46 * cos (pi * n / (self._fftLen * self._nBlocks2)))
            window[n] = val

        # print 'window:'
        # print window

        self._hamming[0:self._windowLen2] = window
        self._hamming[self._fftLen-self._windowLen2:] = window[::-1]

        # print 'self._hamming:'
        # print self._hamming

    def __iter__(self):
        """Return the next cepstra feature."""
        for spectralSample in self._spectralSource:
            samples = (inverse_fft(spectralSample)).real * self._hamming
            yield fft(samples)

    def nextSpkr(self):
        pass


class SingleChannelCepstra:
    """
    Convert a frame of spectral samples to a cepstral feature.
    """
    def __init__(self, spectralSource, cepLen = 13, fftLen = 512):
        self._spectralSource  = spectralSource
        # self._cbpath          = cbpath
        self.__dct            = dctMatrix
        self.__mel            = melMatrix
        self.__cepLen         = cepLen
        self._fftLen2         = fftLen / 2
        self.__cms            = zeros(self.__cepLen, Float)

    def __iter__(self):
        """Return the next cepstra feature."""
        for spectralSample in self._spectralSource:
            # Calculate the frequency-domain output 'Vl' of the GSC
            Vl = spectralSample[0:(self._fftLen2+1)]

            # Calculate mel-warped frequency components 'Vtm' and
            # MFCC vector 'v'
            Vt = matrixmultiply(self.__mel, abs(Vl)**2)
            v  = matrixmultiply(self.__dct, log10(Vt))

#             print 'Spectrum       = ', Vl[0:20]
#             print 'Power spectrum = ', (abs(Vl)**2)[0:20]
#             print 'Mel            = ', Vt[0:20]
#             print 'Cepstra        = ', v

            yield v
            # self.__isamp += 1

    def nextSpkr(self, meanDir = '', spkId = ''):
        """Reset weight vectors for new speaker."""
        self.__isamp    = 0
        self.__logLhood = 0.0

        if meanDir != '' and spkId != '':
            cm = FVector(self.__cepLen)
            cm.bload(meanDir + '/' + spkId + '.mean')

            # load mean for CMS
            for n in range(self.__cepLen):
                self.__cms[n] = cm.getItem(n)

    def nextUtt(self, cbkNamesFile = ''):
        if self.__logLhood > 0.0:
            print 'Utterance %12s : LogLhood = %g' %(os.path.basename(self.__cbNames), self.__logLhood)
            self.__logLhood = 0.0

        # attach next viterbi path to variables
        if cbkNamesFile != '':
            self.__cbNames = cbkNamesFile

class HMMSubbandBeamformer(SubbandBeamformer):
    """
    Unconstrained Hidden Markov model beamformer. Provides functionality
    for performing global optimization.
    """
    def __init__(self, spectralSources, dspath, inDataDir, vitPathDir, meanDir,
                 plotting = False, cepLen = 13):
        """Initialize the HMM sub-band LMS beamformer."""
        SubbandBeamformer.__init__(self, spectralSources)
        self.__plotFlag         = plotting
        self.__dct              = dctMatrix
        self.__mel              = melMatrix
        self.__cepLen           = cepLen
        self.__dspath           = dspath
        self.__cms              = zeros(self.__cepLen, Float)
        self.__inDataDir        = inDataDir
        self.__vitPathDir       = vitPathDir
        self.__meanDir          = meanDir
        self.__cepLen           = cepLen
        self.__ndim             = 2 * (self._fftLen2 + 1) * (self._nChan)

        # Optional interference parameters
        self.__itfDir           = ''
        self.__itfName          = ''
        self.__dBItfLevel       = 0

        # Gradient parameters
        self.__grad_waHK        = []
        self._waHK              = []
        for m in range(self._fftLen2+1):
            self.__grad_waHK.append(zeros(self._nChan, Complex))
            self._waHK.append(zeros(self._nChan, Complex))


    def setInterference(self, itfDir, itfName, dBItfLevel):
        self.__itfDir     = itfDir
        self.__itfName    = itfName
        self.__dBItfLevel = dBItfLevel


    def logLhoodGrad(self, x, utt, vitpathName, cmsName, calcGrad = 0):
        analysisFBs = []
        for i in range(self._nChan):
            fb = FftFB(self._fftLen, self._nBlocks, self._subSampRate)
            nextFile = ("%s/%s.%s.adc" %(self.__inDataDir, utt, (i+1)))
            # print 'Next File = ', nextFile
            soundSource = safeIO(lambda x: OffsetCorrectedFileSoundSource(x, blkLen = self._fftLen,  lastBlk = "unmodified"), nextFile)

            if self.__itfName != '':
                itfFile = ("%s/%s.%s" %(self.__itfDir, self.__itfName, (i+1)))
                soundSource.addInterference(itfFile, dBLevel = self.__dBItfLevel)

            fb.nextUtt(soundSource)
            analysisFBs.append(fb.__iter__())

        # Unpack current weights
        for m in range(1,self._fftLen2+1):
            self._waHK[m] = x[(2*m*(self._nChan)):((2*m+1)*(self._nChan))] + 1j * x[((2*m+1)*(self._nChan)):(2*(m+1)*(self._nChan))]

        # Zero gradient vector
        if calcGrad:
            self.__grad_waHK = []
            for m in range(0,self._fftLen2+1):
                self.__grad_waHK.append(zeros(self._nChan, Complex))

        dsNamePath = ("%s/%s.path" %(self.__vitPathDir, vitpathName))
        # print 'cbkNamePath = ', dsNamePath
        self.__dspath.read(dsNamePath)
        dsiter = self.__dspath.iterator()

        cm = FVector(self.__cepLen)
        cmsPath = ("%s%s.mean" %(self.__meanDir, cmsName))
        # print 'cmsPath = ', cmsPath
        cm.bload(cmsPath)

        # load mean for CMS
        for n in range(self.__cepLen):
            self.__cms[n] = cm.getItem(n)

        logLhood  = 0.0
    
        firstTime = True

        while True:
            try:
                sigmaK = self.updateSnapShotArray(analysisFBs)

                # Calculate the frequency-domain output 'Vl' of the GSC
                Vl  = zeros(self._fftLen2+1, Complex)
                for m in range(1,self._fftLen2+1):
                    # Get next snapshot and form output of blocking matrix.
                    XK     = self.getSnapShot(m)

                    # Calculate the array output.
                    Vl[m] = innerproduct(self._waHK[m], XK)

                    if (m == 100 and firstTime):
                        norm_waK = abs(innerproduct(self._waHK[m], conjugate(self._waHK[m])))
                        print '||waHK||^2  = %8.4e' %(norm_waK)
                        print '  waHK      = ', abs(self._waHK[m])
                        if (self.__plotFlag == 1):
                            plotBeamPattern(self._waHK[m])

                        firstTime = False

                # Calculate mel-warped frequency components 'Vtm' and
                # MFCC vector 'v'
                Vt = matrixmultiply(melMatrix, abs(Vl)**2)
                v  = matrixmultiply(dctMatrix, log10(Vt))

                # get next codebook and find most likely Gaussian component
                ds = dsiter.next()
                cb = ds.cbk()
                diff = zeros(self.__cepLen, Float)
                for n in range(self.__cepLen):
                    diff[n] = (v[n] - self.__cms[n])

                lhoodIndex  = ds.logLhood(diff)
                logLhood   += lhoodIndex.lhood()

                if calcGrad == 0:
                    continue

                gaussIndex  = lhoodIndex.index()

                # Calculate 'rho_nm(t)'
                rho = matrixmultiply(matrixmultiply(dctMatrix, diag(1.0 / Vt)), melMatrix)

                # Calculate 'nu_m(t)'
                for n in range(self.__cepLen):
                    diff[n] = (diff[n] - cb.mean(gaussIndex,n)) * cb.invCov(gaussIndex,n)
                nu = matrixmultiply(diff, rho)

                # Calculate the gradient
                for m in range(1,self._fftLen2+1):
                    XK                   = self.getSnapShot(m)
                    epa                  = Vl[m]
                    self.__grad_waHK[m] += nu[m] * conjugate(XK) * epa

            except StopIteration:
                print 'logLhood = ', logLhood

                if calcGrad == 0:
                    return logLhood

                grad = zeros(self.__ndim, Float)
                for m in range(1,self._fftLen2+1):
                    grad[(2*m*(self._nChan)):((2*m+1)*(self._nChan))]     = self.__grad_waHK[m].real
                    grad[((2*m+1)*(self._nChan)):(2*(m+1)*(self._nChan))] = self.__grad_waHK[m].imag
                return (logLhood, grad)
