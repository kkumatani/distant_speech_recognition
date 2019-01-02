import sys
import string
#import Numeric
#from Numeric import *
import numpy
import pickle
import os
import os.path
import shutil
import pickle
import re
from types import FloatType
import getopt
import copy
import pygsl
from  pygsl import sf

LZERO = (-1.0E10)

# @brief contains parameters of the generalized Gaussian 
#        distribution for the complex r.v.s, namely, 
#        a scale parameter, shape parameter and mean vector.
#        where the second-order circular condition is assumed
class CGGaussianD:
    def __init__(self, p=0.7, sa=0.0, zeroMean=True, mean=0.0 ):
        if zeroMean == True:
            self._zeroMean = True
            self._mean     = 0.0
        else:
            self._zeroMean = False
            self._mean     = mean
        self._sa      = sa    # a scaling factor
        self._p       = p     # a shape parameter
        self._B       = -1    # Gamma(1/p) / Gamma(2/p)
        self._lNF = 0.0       # log(pi) + log( Gamma(1/p ) + log( self._B )
        self._NgConst = 0.0   # pre-calculate the constant value for the negentropy
        self._LlConst = 0.0   # pre-calculate the constant value for the log-likelihood
                              # log(p) - log( pi Gamma(1/p) ka B )
        self._fixed   = False

    def fixed(self):
        return self._fixed 
        
    # @brief initializes parameters and fixes constant values
    def fixConst(self):
        #g1 = sf.gamma( 1.0 / self._p )
        #g2 = sf.gamma( 2.0 / self._p )
        lg1 = sf.lngamma( 1.0 / self._p )
        lg2 = sf.lngamma( 2.0 / self._p )
        #lg3 = sf.lngamma( 1.0 + 1.0 / self._p )
        lB  = lg1[0] - lg2[0]

        #print g1[1],g2[1]
        #print lg1[1],lg2[1]
        self._B       = numpy.exp(lB) #g1[0] / g2[0]
        #self._lNF     = log( Numeric.pi ) + log( g1[0] ) + log( self._B )
        self._lNF     = numpy.log( numpy.pi ) + lg1[0] + lB
        self._NgConst = self._lNF - numpy.log( self._p ) + ( 1.0 / self._p )
        #self._NgConst = log( Numeric.pi )+ lB + lg3[0] + ( 1.0 / self._p )
        self._LlConst = numpy.log( self._p ) - self._lNF
        self._fixed   = True

    # @brief set the scaling parameter of the GGD
    def setScalingPar( self, sa ):
        self._sa = sa
        self._fixed   = False

    # @brief get the scaling parameter of the GGD
    def getScalingPar( self):
        return self._sa

    # @brief set the shape parameter of the GGD
    def setShapePar( self, p ):
        self._p = p
        self._fixed   = False
        
    # @brief get the shape parameter of the GGD
    def getShapePar( self):
        return self._p

    # @brief get Gamma(1/p) / Gamma(2/p)
    # @note it is necessary to calculate sigma based on the ML criterion.
    def getB( self):
        return self._B

    def getLlConst( self ):
        # -log(2*Gamma(1+1/p)A(p))
        return self._LlConst

    def prob( self, X, sigma=0 ):
        loglikelihood = -1000
        if  self._fixed == False:
            print 'call fixConst(self)'
            return
        absX = abs(X-self._mean)
        X2   = absX * absX
        if sigma == 0:
            val = - numpy.power( X2 / ( self._sa * self._B ), self._p )
            loglikelihood = self._LlConst - numpy.log(self._sa) + val
        else:
            val = - numpy.power( X2 / ( sigma * self._B ), self._p )
            loglikelihood = self._LlConst - numpy.log(sigma) + val

        return loglikelihood

    # @brief get the differential entropy
    # @param sigma : a scaling parameter (could be the variance)
    def entropy( self, sigma ):
        if  self._fixed == False:
            print 'call fixConst(self)'
            return
        if sigma == 0:
            return LZERO
        #print "entropy", self._NgConst, numpy.log( sigma )
        val = self._NgConst + numpy.log( sigma )
        return val
        
    def writeParam( self, filename ):
        fp = open( filename, 'w' )
        fp.write( '%e %e %e\n' %(self._sa, self._p, self._mean) )
        fp.write( '%e %e\n' %( self._B, self._lNF ) )
        fp.close()

    def readParam( self, filename, zeroMean=True, fixValues=True ):

        if not os.path.exists(filename):
            print 'Could not find file %s' %filename
            return False

        fp = open( filename, 'r' )
        entries = string.split( string.strip( fp.readline() ) )
        self._sa   = float( entries[0] )
        self._p    = float( entries[1] )
        self._mean = float( entries[2] )
        entries = string.split( string.strip( fp.readline() ) )
        self._B   = float( entries[0] )
        self._lNF = float( entries[1] )
        fp.close()
        
        if zeroMean == True:
            self._zeroMean = True
            self._mean     = 0.0
        else:
            self._zeroMean = False

        print 'Load %s' %(filename)
        print 'scaling par %e, shape par %e, mean %e, B %e, lNF %e' %(self._sa, self._p, self._mean, self._B, self._lNF )
        if fixValues == True:
            self.fixConst( )
        self._fixed   = True
        
        return True

# @brief estimates paramters based on the maximum likelihood criterion.
class MLE4CGGaussianD(CGGaussianD):
    def __init__(self, p=0.7, sigma=0.0, zeroMean=True, mean=0.0, alpha = 0.05 ):
        CGGaussianD.__init__(self, p, sigma, zeroMean, mean )

        # variables for keeping statistics
        self._acc1S    = 0.0
        self._acc1P    = 0.0
        self._acc2P    = 0.0
        self._nSample  = 0
        self._totalL   = 0.0
        # for the gradient algorithm
        self._alpha    = alpha        
        self._nItr     = 0
        self._thresh   = 0.00001
        self._floorV   = 10e-8
        self._floorP   = 0.07
        self._converge = False

    def isConverged(self ):
        return self._converge

    def writeAccs(self, filename ):
        fp = open( filename, 'w' )
        fp.write( '%e\n' %(self._acc1S) )
        fp.write( '%e\n' %(self._acc1P) )
        fp.write( '%e\n' %(self._acc2P) )
        fp.write( '%d\n' %(self._nSample) )
        fp.write( '%e\n' %(self._totalL) )
        
        fp.write( '%e\n' %(self._alpha) )
        fp.write( '%d\n' %(self._nItr) )
        fp.write( '%e\n' %(self._thresh) )
        fp.write( '%e\n' %(self._floorV) )
        fp.write( '%e\n' %(self._floorP) )
        fp.close()
        
    def readAccs(self, filename ):
        fp = open( filename, 'r' )
        acc1S   = float( string.split( string.strip( fp.readline() ) ) )
        acc1P   = float( string.split( string.strip( fp.readline() ) ) )
        acc2P   = float( string.split( string.strip( fp.readline() ) ) )
        nSample = int( string.split( string.strip( fp.readline() ) ) )
        totalL  = float( string.split( string.strip( fp.readline() ) ) )

        alpha   = float( string.split( string.strip( fp.readline() ) ) )
        nItr    = int( string.split( string.strip( fp.readline() ) ) )
        thresh  = float( string.split( string.strip( fp.readline() ) ) )
        floorV  = float( string.split( string.strip( fp.readline() ) ) )
        floorP  = float( string.split( string.strip( fp.readline() ) ) )
        fp.close()

        if nItr != 1:
            print '%s contains Acc data which are updated %d times' %(filename,nItr)
            print 'Those Acc data are ignored'
            return False
        self._nItr = 1
        self._acc1S += acc1S
        self._acc1P += acc1P
        self._acc2P += acc2P
        self._nSample += nSample
        self._totalL  += totalL
        return True
    
    def accSigma(self, Xi ):
        val = abs( Xi - self._mean )
        self._acc1S   += numpy.power( val * val, self._p )
        
    def accP(self, Xi ):
        val  = abs( Xi - self._mean )
        argE = ( val * val ) / ( self._B * self._sa )
        tmp  = numpy.power( argE, self._p )
        if val > 10e-12:
            self._acc1P += tmp * numpy.log( argE )
        self._acc2P += tmp 
        
    def acc( self, Xi ):
        if self._fixed == False:
            print 'Error: Parameters are not initialized correctly'
            return
        self.accSigma( Xi )
        self.accP( Xi )
        self._nSample += 1
        self._totalL  += CGGaussianD.prob( self, Xi )
        
    def clearAcc(self):
        self._acc1S   = 0.0
        self._acc1P   = 0.0
        self._acc2P   = 0.0
        self._nSample = 0
        self._totalL   = 0.0
        
    def update( self, filename = '', sigmaOnly = False ):
        if self._fixed == False:
            print 'Parameters are not initialized correctly'
            print 'could not write %s' %(filename)
            return
        if  self._converge == True:
            print 'Nothing changes'
            return [self._sa, self._p, self._converge]
        # for sigma
        val = numpy.power( ( self._p * self._acc1S ) / self._nSample, 1.0 / self._p )
        newSigma = ( 1 / self._B ) * val

        # for p
        if sigmaOnly == False:
            dg1  = sf.psi( 1.0 /self._p )
            dg2  = sf.psi( 2.0 /self._p )
            dLp1 = ( self._nSample / ( self._p * self._p ) ) * ( self._p  + 2 * dg1[0] - 2 * dg2[0] )
            dLp2 = self._acc1P + self._acc2P * ( dg1[0] - 2 * dg2[0] ) / ( self._p )
            dLp  = ( dLp1 - dLp2 ) # / self._nSample
            newP = self._p + dLp * ( self._alpha  / ( 1 + self._nItr ) )

        print 'UPDATE with %d samples' %(self._nSample)
        print 'AVG Log-Likelihood %e'  %(self._totalL/self._nSample)
        print 'Scaling Factor = %e -> %e ' %(self._sa, newSigma)
        if sigmaOnly == False:
            print 'p     = %e -> %e ' %(self._p, newP)
            if abs( newP - self._p ) < self._thresh :
                self._converge = True
                print 'converged'
        if newSigma < self._floorV:
            print 'Flooring the variance %e to %e' %(newSigma,self._floorV)
            self._sa = self._floorV
        else:
            self._sa = newSigma
        if sigmaOnly == False:
            if newP < self._floorP:
                print 'Flooring p %e to %e' %(newP,self._floorP)
                self._p     = self._floorP
            else:
                self._p     = newP
        CGGaussianD.fixConst( self )

        if filename != '' :
            CGGaussianD.writeParam( self, filename )
        self.clearAcc()
        self._nItr += 1
        
        return [ self._sa, self._p, self._converge ]

