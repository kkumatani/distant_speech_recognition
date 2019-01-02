import sys
import string
import numpy
import os.path
import pickle
import re
from types import FloatType
import getopt, sys
import copy
import gzip

from btk.common import *
from btk.stream import *
from btk.feature import *
from btk.matrix import *
from btk.utils import *

#from pygsl import *
from pygsl import multiminimize
from pygsl import sf
import pygsl.errors as errors

from btk import dbase
from btk.modulated import *
from btk.subbandBeamforming import *
from btk.beamformer import *
import btk.GGDEst
from btk.GGDEst import *
import btk.GGDcEst
from btk.GGDcEst import *


class MNSubbandBeamformer:
    def __init__(self, spectralSources, nSource, NC, alpha, halfBandShift ):

        if NC > 2:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()
        if nSource > 2:
            print "not support more than 2 sources"
            sys.exit(1)
        if halfBandShift==True:
            print "not support halfBandShift==True yet"
            sys.exit(1)
            
        self._halfBandShift = halfBandShift
        self._NC = NC
        self._logfp = 0

        # ouptputs of analysis filter banks
        self._spectralSources = spectralSources
        # the number of channels
        self._nChan           = len(spectralSources)
        # fft length = the number of subbands
        self._fftLen          = spectralSources[0].fftLen()
        # regularization term
        self._alpha = alpha        
        # the number of sound sources
        self._nSource = nSource        
        # input vectors [frameN][chanN]
        self._observations    = []
        # covariance matrix of input vectors [fftLen/2+1][chanN][chanN]
        self._SigmaX = []
        # quiescent vectors : _wq[nSource][fftLen2+1]
        self._wq = []
        # blocking matricies : _B[nSource][fftLen2+1]
        self._B = []
        # the entire GSC 's weight, wq - B * wa : _wo[nSource][fftLen2+1]        
        self._wo = []
        for srcX in range(self._nSource):
            self._wo.append( numpy.zeros( (self._fftLen/2+1,self._nChan), numpy.complex) )

    def nextSpkr(self):
        del self._observations
        del self._SigmaX
        del self._wq
        del self._B
        del self._wo
        self._observations = []
        self._SigmaX = []
        self._wq = []
        self._B = []
        self._wo = []
        for srcX in range(self._nSource):
            self._wo.append( numpy.zeros( (self._fftLen/2+1,self._nChan), numpy.complex) )
        if self._logfp != 0:
            self._logfp.flush()

    def openLogFile(self, logfilename, fbinXD = {50:True,100:True} ):
        self._logfp = gzip.open(logfilename, 'w',1)
        #self._logfp = utils.gzipOpen(logfilename, 'w',1)
        self._fbinXD4log = fbinXD

    def closeLogFile(self):
        if self._logfp != 0:
            self._logfp.close()

    def writeLogFile(self,msg):
        if self._logfp != 0:
            self._logfp.write(msg)
        
    def accumObservations(self, sFrame, eFrame, R=1 ):
        """@brief accumulate observed subband components for adaptation """
        """@param sFrame: the start frame"""
        """@param eFrame: the end frame"""
        """@param R : R = 2**r, where r is a decimation factor"""
        """@return XL[frame][fftLen][chanN] : input subband snapshots"""

        fftLen = self._fftLen
        chanN  = self._nChan
        if R < 1:
            R = 1

        self._observations = []
        for analFB in self._spectralSources:
            analFB.reset()

        # zero mean at this time... , mean = numpy.zeros(chanN).astype(numpy.complex)
        snapShotArray = SnapShotArrayPtr( fftLen, chanN )
        print 'Getting from %d to %d' %(sFrame, eFrame)

        #for sX in range(sFrame,eFrame):
        for sX in range(eFrame):
            try:
                ichan = 0
                for analFB in self._spectralSources:
                    sbSample = numpy.array(analFB.next())
                    snapShotArray.newSample( sbSample, ichan )
                    ichan += 1

                snapShotArray.update()
                if sX >= sFrame and sX < eFrame :
                    X_t = [] # X_t[fftLen][chanN]
                    if sX % R == 0:
                        for fbinX in range(fftLen):
                            X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )
                        self._observations.append( X_t )
            except:
                print 'Accumulated from %d to %d' %(sFrame, eFrame)
                break

        for analFB in self._spectralSources:
            analFB.reset()
            
        del snapShotArray
        return self._observations

    def calcCov(self):
        """@brief calculate covariance matricies of inputs over all frequency bins"""
        """@return the covariance matricies of input vectors : SigmaX[fftLen][chanN][chanN]"""

        if len(self._observations) == 0:
            print "Zero observation! Call getObservations() first!"
            sys.exit()

        samples = self._observations
        frameN  = len( samples )
        fftLen  = self._fftLen
        fftLen2 = fftLen/2
        chanN   = self._nChan
        
        SigmaX = numpy.zeros( (fftLen2+1,chanN,chanN), numpy.complex )

        # zero mean at this time... , mean = numpy.zeros(chanN).astype(numpy.complex)
        for sX in range(frameN):
            for fbinX in range(fftLen2+1):
                # zero mean assumption
                SigmaX[fbinX] += numpy.outer( samples[sX][fbinX], conjugate(samples[sX][fbinX]) )

        for fbinX in range(fftLen2+1):
            SigmaX[fbinX] /= frameN

        self._SigmaX = SigmaX
        
        return self._SigmaX


    def calcEntireWeights_f(self, fbinX, wa_f ):
        """@breif calculate the entire weight vector of the beamformer for each bin"""
        """@param fbinX  : the index of the subband frequency bin"""
        """@param wa_f[nSource][nChan-NC]    """

        nOutput = self._nSource

        for oX in range(nOutput):
            self._wo[oX][fbinX] = self._wq[oX][fbinX] - numpy.dot( self._B[oX][fbinX], wa_f[oX] )

        return self._wo

    def calcGSCOutput_f(self, wo, Xft ):
        """@breif calculate outputs of the GSC at a subband frequency bin"""
        """@param wo[nChan]  : the entire beamformer's weight"""
        """@param Xft[nChan] : the input vector"""
        """@return an output value of a GSC beamformer at a subband frequency bin"""
        """@note this function supports half band shift only"""

        wH  = numpy.transpose( numpy.conjugate( wo ) )
        Yt  = numpy.inner( wH, Xft )

        return Yt

    def H_gaussian(self, detSigmaY ):
        """ @brief calculate the entropy of Gaussian r.v.s."""
        return numpy.log( detSigmaY ) + ( 1 + numpy.log( numpy.pi ) )

    def deltaDetSigmaY( self, srcX, fbinX ):
        """ @brief calculate the derivative of the variance"""
        BH = numpy.conjugate( numpy.transpose( self._B[srcX][fbinX] ) )
        return numpy.dot( numpy.dot( BH, self._SigmaX[fbinX] ), -self._wo[srcX][fbinX] )

    def deltaH_gaussian(self, detSigmaY, deltaSigmaY ):
        """ @brief calculate the derivative of the entropy of the Gaussian variables"""
        return ( deltaSigmaY / detSigmaY )

    def getSourceN(self):
        return  self._nSource

    def getChanN(self):
        return  self._nChan

    def getSampleN(self):
        return len( self._observations )

    def getFftLen(self):
        return self._fftLen

    def getWq(self, srcX, fbinX):
        return self._wq[srcX][fbinX]

    def getB(self, srcX, fbinX):
        return self._B[srcX][fbinX]

    def getAlpha(self):
        return self._alpha

    def setFixedWeights(self, wq, updateBlockingMatrix=False ):
        # @brief set the given quiescent vectors. 
        #        If the second argument is True, blocking matricies are re-calculated.
        # @param wq : wq[srcX][fbinX][chanX]
        # @param updateBlockingMatrix : True or False
        fftLen2 = self._fftLen / 2
        self._wq = []
        if updateBlockingMatrix==True:
            self._B = []

        if self._NC == 1:
            for srcX in range(self._nSource):
                wq_n = []
                if updateBlockingMatrix==True:
                    B_n  = []
                for fbinX in range(fftLen2+1):
                    wq_nf = numpy.zeros( self._nChan, numpy.complex )
                    for chanX in range(self._nChan):
                        wq_nf[chanX] = wq[srcX][fbinX][chanX]
                    wq_n.append(wq_nf)
                    if updateBlockingMatrix==True:
                        B_nf  = calcBlockingMatrix(wq_nf)
                        B_n.append(B_nf)
                self._wq.append(wq_n)
                if updateBlockingMatrix==True:
                    self._B.append(B_n)
        else:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()

    def calcFixedWeights(self, sampleRate, delays ):
        # @brief calculate the quiescent vectors and blocking matricies
        # @param sampleRate : sampling rate (Hz)
        # @param delays[nSource][nChan] : 
        fftLen2 = self._fftLen / 2
        self._wq = []
        self._B  = []

        if self._NC == 1:
            for srcX in range(self._nSource):
                wq_n = []
                B_n  = []      
                for fbinX in range(fftLen2+1):
                    wq_nf = calcArrayManifold_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[0], self._halfBandShift )
                    B_nf  = calcBlockingMatrix(wq_nf)
                    wq_n.append(wq_nf)
                    B_n.append(B_nf)
                self._wq.append(wq_n)
                self._B.append(B_n)
        elif self._nSource==2 and self._NC == 2:
            wq1 = []
            wq2 = []
            B1  = []
            B2  = []                
            for fbinX in range(fftLen2+1):
                wds1 = calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[0], self._halfBandShift)
                wds2 = calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays[1], self._halfBandShift)
                wq1_nf = calcNullBeamformer( wds1, wds2, self._NC )
                wq2_nf = calcNullBeamformer( wds2, wds1, self._NC )
                B1_nf  = calcBlockingMatrix( wq1_nf, self._NC )
                B2_nf  = calcBlockingMatrix( wq2_nf, self._NC )
                wq1.append(wq1_nf)
                wq2.append(wq2_nf)
                B1.append(B1_nf)                
                B2.append(B2_nf)                
            self._wq.append(wq1)
            self._wq.append(wq2)
            self._B.append(B1)
            self._B.append(B2)    
        else:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()

    def UnpackWeights( self, waAs ):
        """@brief Unpack the active weight vector at a frequency bin"""
        nSource = self._nSource
        chanN   = self._nChan
        NC      = self._NC

        weights = []
        idx = 0
        for srcX in range(nSource):
            waA = numpy.zeros(chanN-NC, numpy.complex)
            for chanX in range(chanN-NC):
                waA[chanX] = waAs[2 * chanX + idx ] + 1j * waAs[2 * chanX + 1 + idx]
            weights.append( waA )
            #print '|wa|', numpy.sqrt( inner(waA, conjugate(waA)) )
            idx += ( 2 * (chanN - NC) )

        return weights

# @memo fun_MN() and dfun_MN() are call back functions for pygsl.
#       You can easily implement a new MN beamformer by writing a new class derived from
#       a class 'MNSubbandBeamformer' which have methods, calcEntireWeights_f(),
#       calcCovarOutputs_f(), negentropy() and gradient().
def fun_MN(x, (fbinX, MNSubbandBeamformerPtr, NC) ):
    # @brief calculate the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MNSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants (not yet implemented)
    
    chanN   = MNSubbandBeamformerPtr.getChanN()
    frameN  = MNSubbandBeamformerPtr.getSampleN()
    fftLen  = MNSubbandBeamformerPtr.getFftLen()
    sourceN = MNSubbandBeamformerPtr.getSourceN()
        
    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( numpy.zeros( chanN-NC, numpy.complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )
    
    wa = MNSubbandBeamformerPtr.normalizeWa( fbinX, wa )
    # Calculate the objective function, the negative of the negentropy
    MNSubbandBeamformerPtr.calcEntireWeights_f( fbinX, wa )
    MNSubbandBeamformerPtr.calcCovarOutputs_f( fbinX )
    JY = 0
    for srcX in range(sourceN):
        JY += ( - MNSubbandBeamformerPtr.negentropy( srcX, fbinX ) )
        
    # a regularization term
    alpha = MNSubbandBeamformerPtr.getAlpha()
    rterm = 0
    for srcX in range(sourceN):        
        rterm += alpha * inner(wa[srcX], conjugate(wa[srcX]))
    JY += rterm.real

    if False : # fbinX == 100:
        print 'JY = %e : rterm = %g : alpha = %g' %(JY.real, rterm.real, alpha)

    return JY.real

def dfun_MN(x, (fbinX, MNSubbandBeamformerPtr, NC ) ):
    # @brief calculate the derivatives of the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MNSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants (not yet implemented)
    
    chanN  = MNSubbandBeamformerPtr.getChanN()
    frameN = MNSubbandBeamformerPtr.getSampleN()
    fftLen = MNSubbandBeamformerPtr.getFftLen()
    sourceN = MNSubbandBeamformerPtr.getSourceN()

    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( numpy.zeros( chanN-NC, numpy.complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )
    
    wa = MNSubbandBeamformerPtr.normalizeWa( fbinX, wa )
    # Calculate a gradient
    MNSubbandBeamformerPtr.calcEntireWeights_f( fbinX, wa )
    MNSubbandBeamformerPtr.calcCovarOutputs_f( fbinX )
    deltaWa = []
    for srcX in range(sourceN):
        deltaWa_n = - MNSubbandBeamformerPtr.gradient( srcX, fbinX )
        deltaWa.append( deltaWa_n )

    # add the derivative of the regularization term
    alpha = MNSubbandBeamformerPtr.getAlpha()
    for srcX in range(sourceN):
        deltaWa[srcX] += alpha * wa[srcX]
    
    # Pack the gradient
    grad = numpy.zeros(2 * sourceN * (chanN - NC), numpy.float)
    idx = 0
    for srcX in range(sourceN):
        for chanX in range(chanN - NC):
            grad[2*chanX+ idx]     = deltaWa[srcX][chanX].real
            grad[2*chanX + 1+ idx] = deltaWa[srcX][chanX].imag
        idx += ( 2 * (chanN - NC) )

    return grad

def fdfun_MN(x, (fbinX, MNSubbandBeamformerPtr, NC ) ):
    f  = fun_MN(x, (fbinX, MNSubbandBeamformerPtr, NC ) )
    df = dfun_MN(x, (fbinX, MNSubbandBeamformerPtr, NC ) )

    return f, df

# @class maximum negentropy beamformer with the GG pdf assumption
# usage:
# 1. construct an object, mnBf = MNSubbandBeamformerGGD( spectralSources, ggdL )
# 2. calculate the fixed weights, mnBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mnBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mnBf.calcCov()
# 5. estimate active weight vectors, mnBf.estimateActiveWeights( fbinX, startpoint )
class MNSubbandBeamformerGGDc_pr(MNSubbandBeamformer):
    def __init__(self, spectralSources, ggdL, nSource=1, NC=1, alpha = 1.0E-02, beta=0.5, gamma=-1.0, halfBandShift=False ):
        MNSubbandBeamformer.__init__( self, spectralSources, nSource, NC, alpha, halfBandShift )
        self._ggdL = ggdL
        # beamformer's outputs
        self._outputs = [] # numpy.zeros( (frameN, nSource), numpy.complex )
        # the covariance matrix of the outputs
        self._SigmaY  = [] # numpy.zeros( (nSource,nSource), numpy.complex )
        # SigmaYf[n] = \frac{1}{T} sum_{t=0}^{T-1} |Ynt|^2f
        self._SigmaYf = [] 
        # ScalingPar[n] = ( Gamma(2/f) * Gamma(1/f) ) ( \frac{f}{T} sum_{t=0}^{T-1} |Ynt|^2f )^(1/f)
        self._ScalingPar = numpy.zeros( nSource, numpy.complex )
        self._beta = beta  # weight for entropy of the super-Gaussian 
        self._gamma = gamma
        
    def calcCovarOutputs_f(self, fbinX ):
        # @brief calculate the covariance matrix of beamformer's outputs and
        #        the average values of the p-th power of outputs
        # @param fbinX : the index of the frequency bin

        nSource = self._nSource
        frameN  = len( self._observations )
        # beamformer's outputs
        self._outputs = numpy.zeros( (frameN, nSource), numpy.complex )
        # Sigma_Y = Y * Y^H
        self._SigmaY  = numpy.zeros( (nSource,nSource), numpy.complex )
        # Sigma_Yp[n] = (1/T) sum_{t=0}^{T-1} |Ynt|^2f
        self._SigmaYf = numpy.zeros( nSource, numpy.float )

        f  = self._ggdL[fbinX].getShapePar()
        for frX in range(frameN):
            for srcX in range(nSource):
                self._outputs[frX][srcX] = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            # zero mean assumption
            SigmaY = numpy.outer( self._outputs[frX], numpy.conjugate(self._outputs[frX]) )
            self._SigmaY += SigmaY
            for srcX in range(nSource):
                self._SigmaYf[srcX] += numpy.power( SigmaY[srcX][srcX].real, f )

        self._SigmaY  = self._SigmaY / frameN
        self._SigmaYf = self._SigmaYf / frameN
        self._ScalingPar = numpy.power( f * self._SigmaYf, (1/f) ) / self._ggdL[fbinX].getB()        

    def H_ggaussian(self, srcX, fbinX ):
        # @ brief calculate the entropy with the genealized Gaussian pdf
        entropy = self._ggdL[fbinX].entropy( self._ScalingPar[srcX] )
        return entropy

    def negentropy( self, srcX, fbinX ):
        # @brief calculate the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process
        JY = self.H_gaussian( self._SigmaY[srcX][srcX] ) - self._beta * self.H_ggaussian( srcX, fbinX )

        return JY

    def gradient( self, srcX, fbinX ):
        # @brief calculate the derivative of the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process

        nSource   = self._nSource
        frameN    = len( self._observations )
        f         = self._ggdL[fbinX].getShapePar()
        sigma2    = self._SigmaY[srcX][srcX]
        sigmaYf   = self._SigmaYf[srcX]
        BH        = numpy.conjugate( self._B[srcX][fbinX] ).transpose()
        dJ        = 0

        frameCounter = 0
        for frX in range(frameN):
            for srcX in range(nSource):
                Y       = self._outputs[frX][srcX]
                absY    = abs( Y )
                BH_X_Ya = numpy.dot( BH, self._observations[frX][fbinX] ) * numpy.conjugate( Y )
                if absY > 0.0:
                    val = numpy.power( absY, 2 * f - 2 ) / sigmaYf
                    dJ += ( - (1/sigma2) + self._beta * val ) * BH_X_Ya
                    frameCounter += 1
                else:
                    print 'Skip frame %d with zero power' %(frX)
                    
        return dJ  / frameCounter

    def normalizeWeight( self, srcX, fbinX, wa ):

        nrm_wa2  = numpy.inner(wa, numpy.conjugate(wa))
        #print nrm_wa2,tmp
        nrm_wa  = numpy.sqrt( nrm_wa2.real )
        if self._gamma < 0:
            gamma = numpy.abs( numpy.sqrt( numpy.inner(self._wq[srcX][fbinX],conjugate(self._wq[srcX][fbinX])) ) )
        else:
            gamma = self._gamma
        #if nrm_wa > abs(gamma) :
        #     wa  = gamma * wa / nrm_wa
        #print nrm_wa

        return wa

    def normalizeWa(self, fbinX, wa_f):
        wa = []
        for srcX in range(self._nSource):
            wa.append( self.normalizeWeight( srcX, fbinX, wa_f[srcX] ) )

        return wa

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.2 ):
        # @brief estimate active weight vectors at a frequency bin
        # @param fbinX: the frequency bin index you process
        # @param startpoint: the initial active weight vector
        # @param NC: the number of constrants (not yet implemented)
        # @param MAXITNS: the maximum interation for the gradient algorithm
        # @param TOLERANCE : tolerance for the linear search
        # @param STOPTOLERANCE : tolerance for the gradient algorithm
        
        if fbinX > self._fftLen/2 :
            print "fbinX %d > fftLen/2 %d?" %(fbinX,self._fftLen/2)

        ndim   = 2 * self._nSource * ( self._nChan - self._NC )
        # initialize gsl functions
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MN, dfun_MN, fdfun_MN, [fbinX, self, self._NC], ndim )
        solver = multiminimize.conjugate_pr( sys, ndim )
        #solver = multiminimize.conjugate_pr_eff( sys, ndim )
        #solver = multiminimize.vector_bfgs( sys, ndim )
        #solver = multiminimize.steepest_descent( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "No progress error %f" %mi
                print msg
                break
            except:
                print "Unexpected error:"
                raise
            gradient = solver.gradient()
            waAs = solver.getx()
            mi   = solver.getf()
            try: 
                status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )
            except:
                print "multiminimize.test_gradient:%d %d %f" %(fbinX, itera, mi)

            #print  '%d: %d %e' %(fbinX, itera, mi)

            if self._logfp != 0:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = '%d: %d %e\n' %(fbinX, itera, mi)
                    self._logfp.write( msg )
            if status2==0 :
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged1 %d %d %e\n' %(fbinX, itera,mi)
                    self._logfp.write( msg )
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged2 %d %d %e (%e)\n' %(fbinX, itera,mi, diff)
                    self._logfp.write( msg )
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        return waAs


# @class maximum negentropy beamformer with the GG pdf assumption
# usage:
# 1. construct an object, mnBf = MNSubbandBeamformerGGD( spectralSources, ggdL )
# 2. calculate the fixed weights, mnBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mnBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mnBf.calcCov()
# 5. estimate active weight vectors, mnBf.estimateActiveWeights( fbinX, startpoint )
class MNSubbandBeamformerGGDc_nrm(MNSubbandBeamformer):
    def __init__(self, spectralSources, ggdL, nSource=1, NC=1, alpha = 1.0E-02, beta=1.0, gamma=-1.0, halfBandShift=False ):
        MNSubbandBeamformer.__init__( self, spectralSources, nSource, NC, alpha, halfBandShift )
        self._ggdL = ggdL
        # beamformer's outputs
        self._outputs = [] # numpy.zeros( (frameN, nSource), numpy.complex )
        # the covariance matrix of the outputs
        self._SigmaY  = [] # numpy.zeros( (nSource,nSource), numpy.complex )
        # SigmaYf[n] = \frac{1}{T} sum_{t=0}^{T-1} |Ynt|^2f
        self._SigmaYf = [] 
        # ScalingPar[n] = ( Gamma(2/f) * Gamma(1/f) ) ( \frac{f}{T} sum_{t=0}^{T-1} |Ynt|^2f )^(1/f)
        self._ScalingPar = 0.0
        self._beta = beta  # weight for entropy of the super-Gaussian 
        self._gamma = gamma
        
    def calcCovarOutputs_f(self, fbinX ):
        # @brief calculate the covariance matrix of beamformer's outputs and
        #        the average values of the p-th power of outputs
        # @param fbinX : the index of the frequency bin

        nSource = self._nSource
        frameN  = len( self._observations )
        # beamformer's outputs
        self._outputs = numpy.zeros( (frameN, nSource), numpy.complex )
        # Sigma_Y = Y * Y^H
        self._SigmaY  = numpy.zeros( (nSource,nSource), numpy.complex )
        # Sigma_Yp[n] = (1/T) sum_{t=0}^{T-1} |Ynt|^2f
        self._SigmaYf = numpy.zeros( nSource, numpy.float )

        f  = self._ggdL[fbinX].getShapePar()
        for frX in range(frameN):
            for srcX in range(nSource):
                self._outputs[frX][srcX] = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            # zero mean assumption
            SigmaY = numpy.outer( self._outputs[frX], conjugate(self._outputs[frX]) )
            self._SigmaY += SigmaY
            for srcX in range(nSource):
                self._SigmaYf[srcX] += numpy.power( SigmaY[srcX][srcX].real, f )

        self._SigmaY  = self._SigmaY / frameN
        self._SigmaYf = self._SigmaYf / frameN
        self._ScalingPar = numpy.power( f * self._SigmaYf, (1/f) ) / self._ggdL[fbinX].getB()        

    def H_ggaussian(self, fbinX ):
        # @ brief calculate the entropy with the genealized Gaussian pdf
        entropy = self._ggdL[fbinX].entropy( self._ScalingPar )
        return entropy

    def negentropy( self, srcX, fbinX ):
        # @brief calculate the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process
        JY = self.H_gaussian( self._SigmaY[srcX][srcX] ) - self._beta * self.H_ggaussian( fbinX )

        return JY

    def gradient( self, srcX, fbinX ):
        # @brief calculate the derivative of the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process

        nSource   = self._nSource
        frameN    = len( self._observations )
        f         = self._ggdL[fbinX].getShapePar()
        sigma2    = self._SigmaY[srcX][srcX]
        sigmaYf   = self._SigmaYf[srcX]
        BH        = numpy.transpose( numpy.conjugate( self._B[srcX][fbinX] ) )
        dJ        = 0

        frameCounter = 0
        for frX in range(frameN):
            for srcX in range(nSource):
                Y       = self._outputs[frX][srcX]
                absY    = abs( Y )
                BH_X_Ya = numpy.dot( BH, self._observations[frX][fbinX] ) * numpy.conjugate( Y )
                if absY > 0.0:
                    val = numpy.power( absY, 2 * f - 2 ) / sigmaYf
                    dJ += ( - (1/sigma2) + self._beta * val ) * BH_X_Ya
                    frameCounter += 1
                else:
                    print 'Skip frame %d with zero power' %(frX)
                    
        return dJ  / frameCounter

    def normalizeWeight( self, srcX, fbinX, wa ):

        nrm_wa2  = inner(wa, conjugate(wa))
        #print nrm_wa2,tmp
        nrm_wa  = sqrt( nrm_wa2.real )
        if self._gamma < 0:
            gamma = abs( sqrt( inner(self._wq[srcX][fbinX],conjugate(self._wq[srcX][fbinX])) ) )
        else:
            gamma = self._gamma
        #if nrm_wa > abs(gamma) :
        #     wa  = gamma * wa / nrm_wa
        #print nrm_wa

        return wa

    def normalizeWa(self, fbinX, wa_f):
        wa = []
        for srcX in range(self._nSource):
            wa.append( self.normalizeWeight( srcX, fbinX, wa_f[srcX] ) )

        return wa

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.2 ):
        # @brief estimate active weight vectors at a frequency bin
        # @param fbinX: the frequency bin index you process
        # @param startpoint: the initial active weight vector
        # @param NC: the number of constrants (not yet implemented)
        # @param MAXITNS: the maximum interation for the gradient algorithm
        # @param TOLERANCE : tolerance for the linear search
        # @param STOPTOLERANCE : tolerance for the gradient algorithm
        
        if fbinX > self._fftLen/2 :
            print "fbinX %d > fftLen/2 %d?" %(fbinX,self._fftLen/2)

        ndim   = 2 * self._nSource * ( self._nChan - self._NC )
        # initialize gsl functions
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MN, dfun_MN, fdfun_MN, [fbinX, self, self._NC], ndim )
        solver = multiminimize.conjugate_pr( sys, ndim )
        #solver = multiminimize.conjugate_pr_eff( sys, ndim )
        #solver = multiminimize.vector_bfgs( sys, ndim )
        #solver = multiminimize.steepest_descent( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "No progress error %f" %mi
                print msg
                break
            except:
                print "Unexpected error:"
                raise
            gradient = solver.gradient()
            waAs = solver.getx()
            mi   = solver.getf()
            try: 
                status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )
            except:
                print "multiminimize.test_gradient:%d %d %f" %(fbinX, itera, mi)

            #print  '%d: %d %e' %(fbinX, itera, mi)

            if self._logfp != 0:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = '%d: %d %e\n' %(fbinX, itera, mi)
                    self._logfp.write( msg )
            if status2==0 :
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged1 %d %d %e\n' %(fbinX, itera,mi)
                    self._logfp.write( msg )
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged2 %d %d %e (%e)\n' %(fbinX, itera,mi, diff)
                    self._logfp.write( msg )
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        return waAs


# @class maximum negentropy beamformer with the Gamma pdf assumption
# @note not tested yet (see 000008)
# usage:
# 1. construct an object, mnBf = MNSubbandBeamformerGamma( spectralSources  )
# 2. calculate the fixed weights, mnBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mnBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mnBf.calcCov()
# 5. estimate active weight vectors, mnBf.estimateActiveWeights( fbinX, startpoint )
SRangeCheck = True
ARGMAX_MEIJERG = 60
ARGMIN_MEIJERG = 1.0E-08
ARGMAX_GAMMA   = 70
gamma2 = gammaPdfPtr(2)
gamma4 = gammaPdfPtr(4)
class MNSubbandBeamformerGamma(MNSubbandBeamformer):
    def __init__(self, spectralSources,nSource=1, NC=1, alpha = 1.0E-02, halfBandShift=False  ):
        MNSubbandBeamformer.__init__(self, spectralSources, nSource, NC, alpha, halfBandShift )
        # beamformer's outputs
        self._outputs = [] # numpy.zeros( (frameN, nSource), numpy.complex )
        # the covariance matrix of the outputs
        self._SigmaY  = [] # numpy.zeros( (nSource,nSource), numpy.complex )
        
    def calcCovarOutputs_f(self, fbinX ):
        # @brief calculate the covariance matrix of beamformer's outputs and
        #        the average values of the p-th power of outputs
        # @param fbinX : the index of the frequency bin

        nSource = self._nSource
        frameN = len( self._observations )
        # beamformer's outputs
        self._outputs = numpy.zeros( (frameN, nSource), numpy.complex )
        # Sigma_Y = Y * Y^H
        self._SigmaY  = numpy.zeros( (nSource,nSource), numpy.complex )
    
        for frX in range(frameN):
            for srcX in range(nSource):
                self._outputs[frX][srcX] = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            # zero mean assumption
            self._SigmaY += numpy.outer( self._outputs[frX], conjugate(self._outputs[frX]) )

        self._SigmaY /= frameN


    def calcLogGamma( self, sigmaY, absY ):
        # -1.5632291352502243 = log(2*0.2443) - 0.5 * log(2*pi) - 0.25 * log( 2* 3.0 / 8.0 )
        tmp = absY / sigmaY 
        x_g2 = ( 3.0 / 4.0 ) * ( tmp * tmp )
        logL = -1.5632291352502243 - 0.5 * numpy.log(sigmaY)  - 1.5 * numpy.log(absY) + gamma2.calcLog( x_g2.real, 13 )

        return logL

    def H_gamma(self, srcX ):
        # @ brief calculate the entropy with the gamma pdf 
        expec = 0.0
        frameN = len( self._outputs )
        rSigmaY = numpy.sqrt( self._SigmaY[srcX][srcX] )
        
        for frX in range(frameN):
            absY = numpy.absolute( self._outputs[frX][srcX] )
            expec += self.calcLogGamma( rSigmaY, absY )

        return ( -expec / frameN )

    def negentropy( self, srcX, fbinX ):
        # @brief calculate the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process

        JY = self.H_gaussian( self._SigmaY[srcX][srcX] ) -  self.H_gamma( srcX )
        return JY

    def deltaAbsY( self, absY, wo, B, X ):
        x2 = numpy.outer( X, numpy.conjugate( X ) )
        L1 = numpy.dot( numpy.conjugate( numpy.transpose( B ) ), x2 )
        L2 = numpy.dot( L1, wo )
        delta = -0.5 * L2 / absY

        return delta

    def calcLogDeltaGamma( self, sigmaY, absY, dsigmaY, wo, B, X ):
        dAbsY = self.deltaAbsY( absY, wo, B, X )
        tmp  = absY / sigmaY
        tmp2 = tmp * tmp
        x_g2 = ( 3.0 / 4.0 ) * tmp2.real
        tmpV = ( 1.5 * tmp2 ) * ( ( dAbsY / absY ) - ( dsigmaY / sigmaY ) )
        g2   = gamma2.calcLog( x_g2, 13 )
        delta_g2 = gamma2.calcDerivative1( x_g2, 13 ) * tmpV
        delta = - ( 1.5 * dAbsY / absY ) - ( 0.5 * dsigmaY / sigmaY ) + delta_g2 / g2

        return delta

    def delta_sigmaY( self, sigmaY, wo, B, SigmaX ):
        BH = numpy.conjugate( numpy.transpose( B ) )

        return numpy.dot( numpy.dot( BH, SigmaX ), wo ) / ( - 2 * sigmaY )

    def deltaH_Gamma( self, srcX, fbinX, Y, sigmaY2, wo, B, X, SigmaX ):
        expec = numpy.zeros( len(B[0]), numpy.complex )
        frameN = len( Y )
        sigmaY = numpy.sqrt(sigmaY2)

        for frX in range(frameN):
            absY = numpy.absolute( Y[frX][srcX] )
            dsigmaY = self.delta_sigmaY( sigmaY, wo, B, SigmaX )
            expec  += self.calcLogDeltaGamma( sigmaY, absY, dsigmaY, wo, B, X[frX][fbinX] )

        return ( - expec / frameN )

    def gradient( self, srcX, fbinX ):
        # @brief calculate the derivative of the negentropy
        # @param srcX: the source index you process
        # @param fbinX: the frequency bin index you process

        detSigmaY = self._SigmaY[srcX][srcX]

        # calculate the derivative of the entropy of the Gamma r.v.s
        deltaH_gammaY = self.deltaH_Gamma( srcX, fbinX, self._outputs, self._SigmaY[srcX][srcX], self._wo[srcX][fbinX], self._B[srcX][fbinX], self._observations, self._SigmaX[fbinX] )
        
        # calculate the derivative of the entropy of the Gaussian r.v.s
        deltaSigmaY = self.deltaDetSigmaY( srcX, fbinX )
        deltaH_gaussianY = self.deltaH_gaussian( detSigmaY, deltaSigmaY )
        
        return deltaH_gaussianY -  deltaH_gammaY
    

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.2 ):
        # @brief estimate active weight vectors at a frequency bin
        # @param fbinX: the frequency bin index you process
        # @param startpoint: the initial active weight vector
        # @param NC: the number of constrants (not yet implemented)
        # @param MAXITNS: the maximum interation for the gradient algorithm
        # @param TOLERANCE : tolerance for the linear search
        # @param STOPTOLERANCE : tolerance for the gradient algorithm
        
        if fbinX > self._fftLen/2 :
            print "fbinX %d > fftLen/2 %d?" %(fbinX,self._fftLen/2)

        ndim   = 2 * self._nSource * ( self._nChan - self._NC )
        # initialize gsl functions
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MN, dfun_MN, fdfun_MN, [fbinX, self, self._NC], ndim )
        solver = multiminimize.conjugate_pr( sys, ndim )
        #solver = multiminimize.conjugate_pr_eff( sys, ndim )
        #solver = multiminimize.vector_bfgs( sys, ndim )
        #solver = multiminimize.steepest_descent( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "No progress error (%d: %d %e)"  %(fbinX, itera, mi)
                print msg
                break
            except:
                print "Unexpected error:"
                raise
            gradient = solver.gradient()
            waAs = solver.getx()
            mi   = solver.getf()
            status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

            if self._logfp != 0:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = '%d: %d %e\n' %(fbinX, itera, mi)
                    self._logfp.write( msg )
            if status2==0 :
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged1 %d %d %e\n' %(fbinX, itera,mi)
                    self._logfp.write( msg )
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged2 %d %d %e (%e)\n' %(fbinX, itera,mi, diff)
                    self._logfp.write( msg )
                break
            preMi = mi
            
        wa = numpy.zeros( self._nChan - self._NC, numpy.complex )
        for chanX in range( self._nChan - self._NC ):
            wa[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
        wa = self.normalizeWeight( 0, fbinX, wa )
        for chanX in range( self._nChan - self._NC ):
            waAs[2*chanX]     = wa[chanX].real
            waAs[2*chanX + 1] = wa[chanX].imag

        #print '=== %d' %(fbinX)
        return waAs

