import sys
import string
import numpy
from numpy import *
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

APPZERO = 1.0E-20

class MKSubbandBeamformer:
    def __init__(self, spectralSources, NC, alpha, halfBandShift ):

        # the number of sound sources
        self._nSource = 1
        self._logfp = 0

        if NC > 2:
            print 'not yet implemented in the case of NC > 2'
            sys.exit()
        if halfBandShift==True:
            print "not support halfBandShift==True yet"
            sys.exit(1)
            
        self._halfBandShift = halfBandShift
        self._NC = NC
        
        # ouptputs of analysis filter banks
        self._spectralSources = spectralSources
        # the number of channels
        self._nChan           = len(spectralSources)
        # fft length = the number of subbands
        self._fftLen          = spectralSources[0].fftLen()
        # regularization term
        self._alpha = alpha        
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
        """@return self._observations[frame][fftLen][chanN] : input subband snapshots"""

        fftLen = self._fftLen
        chanN  = self._nChan
        if R < 1:
            R = 1

        self._observations = []
        
        # zero mean at this time... , mean = numpy.zeros(chanN).astype(numpy.complex)
        snapShotArray = SnapShotArrayPtr( fftLen, chanN )
        #print 'from %d to %d, fftLen %d' %( sFrame, eFrame, snapShotArray.fftLen() )

        #for sX in range(sFrame,eFrame):
        counter = 0
        try:            
            for sX in range(eFrame):
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
                            X_t.append( numpy.array( snapShotArray.getSnapShot(fbinX) ) )
#                            X_t.append( copy.deepcopy( snapShotArray.getSnapShot(fbinX) ) )
                        self._observations.append( X_t )
                        #print X_t
                counter = sX

            for analFB in self._spectralSources:
                analFB.reset()
        except :
            print 'reach the end %d' %counter
            return self._observations
            
        #del snapShotArray
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

    def calcGSCOutput_f(self, wo, Xft ):
        """@breif calculate outputs of the GSC at a subband frequency bin"""
        """@param wo[nChan]  : the entire beamformer's weight"""
        """@param Xft[nChan] : the input vector"""
        """@return an output value of a GSC beamformer at a subband frequency bin"""
        """@note this function supports half band shift only"""

        wH  = numpy.transpose( numpy.conjugate( wo ) )
        Yt  = numpy.dot( wH, Xft )

        return Yt

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

    def setFixedWeights(self, wq, updateBlockingMatrix=False, norm=1 ):
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
                        wq_nf[chanX] = wq[srcX][fbinX][chanX] / norm
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
        elif self._NC == 2:
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
            #print '|wa|', numpy.sqrt( dot(waA, conjugate(waA)) )
            idx += ( 2 * (chanN - NC) )

        return weights

# @memo fun_MK() and dfun_MK() are call back functions for pygsl.
#       You can easily implement a new MK beamformer by writing a new class derived from
#       a class 'MKSubbandBeamformer' which have methods, normalizeWa( wa ),
#       calcKurtosis( srcX, fbinX, wa ) and gradient( srcX, fbinX, wa ).
def fun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC) ):
    # @brief calculate the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MNSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants (not yet implemented)

    chanN   = MKSubbandBeamformerPtr.getChanN()
    frameN  = MKSubbandBeamformerPtr.getSampleN()
    fftLen  = MKSubbandBeamformerPtr.getFftLen()
    sourceN = MKSubbandBeamformerPtr.getSourceN()
        
    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( numpy.zeros( chanN-NC, numpy.complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )

    wa    = MKSubbandBeamformerPtr.normalizeWa( fbinX, wa )
    # Calculate the objective function, the negative of the kurtosis
    nkurt = 0.0
    for srcX in range(sourceN):
        nkurt -= MKSubbandBeamformerPtr.calcKurtosis( srcX, fbinX, wa )
    # a regularization term
    rterm = 0.0
    alpha = MKSubbandBeamformerPtr.getAlpha()
    for srcX in range(sourceN):    
        rterm  +=  alpha * numpy.inner(wa, conjugate(wa)) 
    nkurt  += rterm.real

    del wa
    return nkurt

def dfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) ):
    # @brief calculate the derivatives of the objective function for the gradient algorithm
    # @param x[2(chanN-NC)] : active weights (packed)
    # @param fbinX: the frequency bin index you process 
    # @param MKSubbandBeamformerPtr: the class object for calculating functions 
    # @param NC: the number of constrants 
    
    chanN  = MKSubbandBeamformerPtr.getChanN()
    frameN = MKSubbandBeamformerPtr.getSampleN()
    fftLen = MKSubbandBeamformerPtr.getFftLen()
    sourceN = MKSubbandBeamformerPtr.getSourceN()

    # Unpack current weights : x[2*nSource*(chanN - NC )] -> wa[nSource][chanN-NC]
    wa = []
    idx = 0
    for srcX in range(sourceN):
        wa.append( numpy.zeros( chanN-NC, numpy.complex) )
        for chanX in range(chanN-NC):
            wa[srcX][chanX] = x[2 * chanX+ idx] + 1j * x[2 * chanX + 1+ idx]
        idx += ( 2 * (chanN - NC) )

    wa    = MKSubbandBeamformerPtr.normalizeWa( fbinX, wa )    
    # Calculate a gradient
    deltaWa = []
    for srcX in range(sourceN):
        deltaWa_n = - MKSubbandBeamformerPtr.gradient( srcX, fbinX, wa )
        deltaWa.append( deltaWa_n )

    # add the derivative of the regularization term
    alpha = MKSubbandBeamformerPtr.getAlpha()
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

    #if fbinX == 10:
    #    print 'grad', grad
    del wa
    return grad

def fdfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) ):
    f  = fun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) )
    df = dfun_MK(x, (fbinX, MKSubbandBeamformerPtr, NC ) )

    return f, df

# @class maximum empirical kurtosis beamformer 
# usage:
# 1. construct an object, mkBf = MKSubbandBeamformerGGDr( spectralSources  )
# 2. calculate the fixed weights, mkBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mkBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mkBf.calcCov()
# 5. estimate active weight vectors, mkBf.estimateActiveWeights( fbinX, startpoint )
class MEKSubbandBeamformer_pr(MKSubbandBeamformer):
    def __init__(self, spectralSources, NC=1, alpha = 1.0E-02, beta = 3.0, halfBandShift=False  ):
        MKSubbandBeamformer.__init__(self, spectralSources, NC, alpha, halfBandShift )
        self._beta  = beta

        self.resetStatistics()

    def resetStatistics(self):
        self._prevAvgY4  = numpy.zeros( (self._nSource,self._fftLen/2+1), numpy.float )
        self._prevAvgY2  = numpy.zeros( (self._nSource,self._fftLen/2+1), numpy.float )
        self._prevFrameN = numpy.zeros( (self._nSource,self._fftLen/2+1), numpy.int )

    def storeStatistics(self, srcX, fbinX, wa_f):
        frameN = len( self._observations )
        self._prevFrameN[srcX][fbinX] += frameN

        for frX in range(frameN):
            self.calcEntireWeights_f( fbinX, wa_f )
            Y  = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            Y2 = Y * numpy.conjugate( Y )
            Y4 = Y2 * Y2
            self._prevAvgY2[srcX][fbinX] += ( Y2.real / self._prevFrameN[srcX][fbinX] )
            self._prevAvgY4[srcX][fbinX] += ( Y4.real / self._prevFrameN[srcX][fbinX] )
        #print 'Store %d : %e %e %d' %(fbinX,self._prevAvgY4[srcX][fbinX],self._prevAvgY2[srcX][fbinX],self._prevFrameN[srcX][fbinX])

    def normalizeWa(self, fbinX, wa):
        return wa
    
    def calcEntireWeights_f(self, fbinX, wa_f ):
        """@breif calculate the entire weight vector of the beamformer for each bin"""
        """@param fbinX  : the index of the subband frequency bin"""
        """@param wa_f[nSource][nChan-NC]    """

        for srcX in range(self._nSource):
            self._wo[srcX][fbinX] = self._wq[srcX][fbinX] - numpy.dot( self._B[srcX][fbinX], wa_f[srcX] )
            
        return self._wo

    def calcKurtosis( self, srcX, fbinX, wa_f ):
        # @brief calculate empirical kurtosis :
        #        \frac{1}{T} \sum_{t=0}^{T-1} Y^4 - 3 ( \frac{1}{T} \sum_{t=0}^{T-1} Y^2 )
        # @param srcX: the source index you process
        # @param fbinX  : the index of the subband frequency bin"""
        # @param wa_f[nSource][nChan-NC]
        frameN = len( self._observations )
        totalFrameN = self._prevFrameN[srcX][fbinX] + frameN

        exY4 = ( self._prevAvgY4[srcX][fbinX] / totalFrameN ) * self._prevFrameN[srcX][fbinX]
        exY2 = ( self._prevAvgY2[srcX][fbinX] / totalFrameN ) * self._prevFrameN[srcX][fbinX]
        for frX in range(frameN):
            self.calcEntireWeights_f( fbinX, wa_f )
            Y  = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            Y2 = Y * numpy.conjugate( Y )
            Y4 = Y2 * Y2
            exY2 += ( Y2.real / totalFrameN )
            exY4 += ( Y4.real / totalFrameN )

        kurt = exY4 - self._beta * exY2 * exY2
        return kurt

    def gradient( self, srcX, fbinX, wa_f ):
        # @brief calculate the derivative of empirical kurtosis w.r.t. wa_H
        # @param srcX: the source index you process
        # @param fbinX  : the index of the subband frequency bin"""
        # @param wa_f[nSource][nChan-NC]
        frameN = len( self._observations )
        totalFrameN = self._prevFrameN[srcX][fbinX] + frameN

        exY2  = ( self._prevAvgY2[srcX][fbinX] / totalFrameN ) * self._prevFrameN[srcX][fbinX] 
        dexY2 = numpy.zeros( ( self._nChan - self._NC ), numpy.complex )
        dexY4 = numpy.zeros( ( self._nChan - self._NC ), numpy.complex )

        BH    = numpy.transpose( numpy.conjugate( self._B[srcX][fbinX] ) )
        for frX in range(frameN):
            self.calcEntireWeights_f( fbinX, wa_f )
            Y   = self.calcGSCOutput_f( self._wo[srcX][fbinX], self._observations[frX][fbinX] )
            BHX = - numpy.dot( BH, self._observations[frX][fbinX] ) # BH * X
            Y2  = Y * numpy.conjugate( Y )
            dexY4 += ( 2 * Y2 * BHX * numpy.conjugate( Y ) / totalFrameN )
            dexY2 += ( BHX * numpy.conjugate( Y ) / totalFrameN )
            exY2  += ( Y2.real / totalFrameN )

        deltaKurt = dexY4 - 2 * self._beta * exY2 * dexY2
        del dexY2
        del dexY4

        return deltaKurt

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-05, STEPSIZE=0.01 ):
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
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MK, dfun_MK, fdfun_MK, [fbinX, self, self._NC], ndim )
        solver = multiminimize.conjugate_pr( sys, ndim )
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
            status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

            if fbinX % 10 == 0:
                print 'EK %d %d %e' %(fbinX, itera, mi)
            if status2==0 :
                print 'EK Converged %d %d %e' %(fbinX, itera,mi)
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE:
                print 'EK Converged %d %d %e (%e)' %(fbinX, itera,mi, diff)
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        return waAs

# @class maximum empirical kurtosis beamformer.
#        The entire weight is normalized at each step in the steepest gradient algorithm.
# usage:
# 1. construct an object, mkBf = MEKSubbandBeamformer_nrm( spectralSources  )
# 2. calculate the fixed weights, mkBf.calcFixedWeights( sampleRate, delay )
# 3. accumulate input vectors, mkBf.accumObservations( sFrame, eFrame, R )
# 4. calculate the covariance matricies of the inputs, mkBf.calcCov()
# 5. estimate active weight vectors, mkBf.estimateActiveWeights( fbinX, startpoint )
class MEKSubbandBeamformer_nrm(MEKSubbandBeamformer_pr):
    def __init__(self, spectralSources, NC=1, alpha = 0.1, beta=3.0, gamma=-1.0, halfBandShift=False  ):
        MEKSubbandBeamformer_pr.__init__(self, spectralSources, NC, alpha, beta, halfBandShift )
        self._gamma = gamma

    def normalizeWeight( self, srcX, fbinX, wa ):
        nrm_wa2 = numpy.inner(wa, conjugate(wa))
        nrm_wa  = sqrt( nrm_wa2.real )
        if self._gamma < 0:
            gamma = sqrt( numpy.inner(self._wq[srcX][fbinX],conjugate(self._wq[srcX][fbinX])) )
        else:
            gamma = self._gamma
        if nrm_wa > abs(gamma) : # >= 1.0:
            wa  =  abs(gamma) * wa / nrm_wa

        return wa

    def normalizeWa(self, fbinX, wa_f):
        wa = []
        for srcX in range(self._nSource):
            wa.append( self.normalizeWeight( srcX, fbinX, wa_f[srcX] ) )
            
        return wa
    
    def calcEntireWeights_f(self, fbinX, wa_f ):
        """@breif calculate and normalize the entire weight vector of the beamformer for each bin"""
        """@param fbinX  : the index of the subband frequency bin"""
        """@param wa_f[nSource][nChan-NC]    """

        for srcX in range(self._nSource):
            wa = self.normalizeWeight(  srcX, fbinX, wa_f[srcX] )
            self._wo[srcX][fbinX] = self._wq[srcX][fbinX] - numpy.dot( self._B[srcX][fbinX], wa )
            
        return self._wo

    def estimateActiveWeights( self, fbinX, startpoint, MAXITNS=40, TOLERANCE=1.0E-03, STOPTOLERANCE = 1.0E-02, DIFFSTOPTOLERANCE= 1.0E-10, STEPSIZE=0.01 ):
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
        sys    = multiminimize.gsl_multimin_function_fdf( fun_MK, dfun_MK, fdfun_MK, [fbinX, self, self._NC], ndim )
        solver = multiminimize.steepest_descent( sys, ndim )
        solver.set(startpoint, STEPSIZE, TOLERANCE )
        waAs = startpoint
        #print "Using solver ", solver.name()

        MINITERA = 2
        mi = 10000.0
        preMi = 10000.0
        for itera in range(MAXITNS):
            try: 
                status1 = solver.iterate()
            except errors.gsl_NoProgressError, msg:
                print "solver.iterate(): No progress error %d" %(fbinX)
                print msg,mi
                break
            except:
                print "solver.iterate(): Unexpected error:"
                break
            status2 = 0
            try:
                gradient = solver.gradient()
                status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )
            except errors.gsl_NoProgressError, msg:
                print "multiminimize.test_gradient: No progress error %d" %(fbinX)
                print msg,mi
                break
            except:
                print "multiminimize.test_gradient: Unexpected error:"                
                break
            waAs = solver.getx()
            mi   = solver.getf()

            if self._logfp != 0:
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = '%d: %d %e\n' %(fbinX, itera, mi)
                    self._logfp.write( msg )
            if status2==0 and itera > MINITERA :
                print 'Converged1 %d %d %e' %(fbinX, itera,mi)
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged1 %d %d %e\n' %(fbinX, itera,mi)
                    self._logfp.write( msg )
                break
            diff = abs( preMi - mi )
            if diff < DIFFSTOPTOLERANCE and itera > MINITERA:
                print 'Converged2 %d %d %e (%e)' %(fbinX, itera,mi, diff)
                if self._fbinXD4log.has_key(fbinX)==True:
                    msg = 'Converged2 %d %d %e (%e)\n' %(fbinX, itera,mi, diff)
                    self._logfp.write( msg )
                break
            preMi = mi

        #print '=== %d' %(fbinX)
        # Unpack current weights and normalize them
        wa = numpy.zeros( self._nChan - self._NC, numpy.complex)
        for chanX in range( self._nChan - self._NC ):
            wa[chanX] = waAs[2 * chanX] + 1j * waAs[2 * chanX + 1]
        wa = self.normalizeWeight( 0, fbinX, wa )
        self.storeStatistics( 0, fbinX, [wa] )
        for chanX in range( self._nChan - self._NC ):
            waAs[2*chanX]     = wa[chanX].real
            waAs[2*chanX + 1] = wa[chanX].imag

        del wa
        #print waAs
        return waAs

