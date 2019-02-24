import sys
import string
from numpy import *
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
#        distribution, a scale parameter, shape parameter 
#        and mean vector.
class GGaussianD:
    def __init__(self, p=0.7, sigma=0.0, zeroMean=True, mean=0.0 ):
        if zeroMean == True:
            self._zeroMean = True
            self._mean     = 0.0
        else:
            self._zeroMean = False
            self._mean     = mean
        self._sigma   = sigma # a scaling factor
        self._p       = p     # a shape parameter
        self._A       = -1    # A(p, sigma)
        self._lNF     = -1    # - log( 2 * Gamma(1+1/p) * A(p,sigma) )
        self._C       = 0.0   # sqrt( Gamma(3/p) / Gamma(1/p) )
        self._NgConst = 0.0   # pre-calculate the constant value for the negentropy
        self._LlConst = 0.0   # pre-calculate the constant value for the log-likelihood
        self._fixed   = False

    def fixed(self):
        return self._fixed 
        
    # @brief initializes parameters and fixes constant values
    def fixConst(self):
        g1 = sf.gamma( 1.0 / self._p )
        g2 = sf.gamma( 3.0 / self._p )
        g3 = sf.gamma( 1 + 1.0 / self._p )
        # A(p, sigma)        
        self._A       = self._sigma * sqrt( g1[0] / g2[0] )
        val = log( 2 * g3[0] )
        # - log( 2 * Gamma(1+1/p) * A(p,sigma) )
        self._lNF     = -( val + log(self._A) )
        # sqrt( Gamma(3/p) / Gamma(1/p) )
        self._C = sqrt( g2[0] / g1[0] ) 
        # H(Y) - log( sigma )
        self._NgConst = val + ( 1.0 / self._p ) + 0.5 * log( g1[0] / g2[0] )
        self._LlConst =  - val - 0.5 * log(  g1[0] / g2[0] ) # -log(2*Gamma(1+1/p)A(p))
        self._fixed   = True

    # @brief set the shape parameter of the GGD
    def setP( self, p ):
        self._p = p
        self._fixed   = False

    # @brief set the scaling parameter of the GGD
    def setSigma( self, sigma ):
        self._sigma = sigma
        self._fixed   = False

    # @brief get the scaling parameter of the GGD
    def getSigma( self):
        return self._sigma
        
    # @brief get the shape parameter of the GGD
    def getP( self):
        return self._p

    # @brief get sqrt( Gamma(3/p) / Gamma(1/p) )
    # @note it is necessary to calculate sigma based on the ML criterion.
    def getC( self):
        return self._C
        
    def getA( self ):
        return self._A

    def getLlConst( self ):
        # -log(2*Gamma(1+1/p)A(p))
        return self._LlConst

    def prob( self, X, sigma=0 ):
        loglikelihood = -1000
        if  self._fixed == False:
            print 'call fixConst(self)'
            return loglikelihood
        if sigma == 0:
            val = - power( abs((X-self._mean)/self._A), self._p )
            loglikelihood = self._lNF + val
        else:
            val = - power( abs((X-self._mean)*self._C/sigma), self._p )
            loglikelihood = self._lNF + log(self._sigma) - log(sigma) + val
            
        return loglikelihood

    # @brief get the entropy
    # @param sigma : a scaling parameter (could be the standard deviation)
    def entropy( self, sigma ):
        if  self._fixed == False:
            print 'call fixConst(self)'
            return
        if sigma == 0:
            return LZERO
        val = self._NgConst + log( sigma )
        return val
        
    def writeParam( self, filename ):
        fp = open( filename, 'w' )
        fp.write( '%e %e %e\n' %(self._sigma, self._p, self._mean) )
        fp.write( '%e %e\n' %( self._A, self._lNF ) )
        fp.close()

    def readParam( self, filename, zeroMean=True, fixValues=True ):

        if not os.path.exists(filename):
            print 'Could not find file %s' %filename
            return False

        fp = open( filename, 'r' )
        entries = string.split( string.strip( fp.readline() ) )
        self._sigma = float( entries[0] )
        self._p    = float( entries[1] )
        self._mean = float( entries[2] )
        entries = string.split( string.strip( fp.readline() ) )
        self._A   = float( entries[0] )
        self._lNF = float( entries[1] )
        fp.close()
        
        if zeroMean == True:
            self._zeroMean = True
            self._mean     = 0.0
        else:
            self._zeroMean = False

        print 'Load %s' %(filename)
        print 'sigma %e, p %e, mean %e, A %e, lNF %e' %(self._sigma, self._p, self._mean, self._A, self._lNF )
        if fixValues == True:
            self.fixConst( )
        self._fixed   = True
        
        return True

class MME4GGaussianD(GGaussianD):
    def __init__(self, p, sigma=0.0, zeroMean=True, mean=0.0 ):
        GGaussianD.__init__(self, p, sigma, zeroMean, mean )

        # variables for keeping statistics
        self._NMp = 0.0
        self._newMean   = 0.0
        self._newSigma2 = 0.0
        self._nSample   = 0
        self._nItr     = 0

    def writeAccs(self, filename ):
        fp = open( filename, 'w' )
        fp.write( '%e\n' %(self._NMp) )
        fp.write( '%e\n' %(self._newMean) )
        fp.write( '%e\n' %(self._newSigma2) )
        fp.write( '%d\n' %(self._nSample) )
        fp.write( '%d\n' %(self._nItr) )
        fp.close()
        
    def readAccs(self, filename ):
        fp = open( filename, 'r' )
        NMp       = float( string.split( string.strip( fp.readline() ) ) )
        newMean   = float( string.split( string.strip( fp.readline() ) ) )
        newSigma2 = float( string.split( string.strip( fp.readline() ) ) )
        nSample = int( string.split( string.strip( fp.readline() ) ) )
        nItr    = int( string.split( string.strip( fp.readline() ) ) )
        fp.close()

        if nItr != 1:
            print '%s contains Acc data which are updated %d times' %(filename,nItr)
            print 'Those Acc data are ignored'
            return False
        self._nItr = 1
        self._NMp += NMp
        self._newMean   += newMean
        self._newSigma2 += newSigma2
        self._nSample   += nSample
        return True
        
    def accMean(self, Xi ):
        self._newMean  += Xi

    def accSigma(self, Xi ):
        val = Xi -  self._mean
        self._newSigma2 += val * val
        
    def accP(self, Xi ):
        self._NMp += abs( Xi - self._mean )

    def acc( self, Xi ):
        if self._zeroMean == True:
            self.accMean( Xi )
        self.accSigma( Xi )
        self.accP( Xi )
        self._nSample += 1

    def clearAcc(self):
        self._NMp       = 0.0
        self._newMean   = 0.0
        self._newSigma2 = 0.0
        self._nSample   = 0

    def update( self, filename = '' ):        
        if self._zeroMean == False:
            self._mean = self._newMean / self._nSample
    
        sigma2 = self._newSigma2 / self._nSample 
        self._sigma = sqrt( sigma2 )
        val =  self._NMp  / self._nSample
        Mp  = ( val * val )/ sigma2

        if Mp < 0.0131246:
            self._p = 2 * ( log( 27.0/16.0 ) ) / ( log(3/4) - 2 * log( Mp ) )
        elif Mp < 0.448994:
            a1 = -0.535707356
            a2 = 1.168939911
            a3 = -0.1516189217
            self._p = ( -a2 + sqrt( a2 * a2 - 4 * a1 * a3 + 4 * a1 * Mp ) ) / ( 2 * a1 )
        elif Mp < 0.671256:
            b1 = 0.9694429
            b2 = 0.8727534
            b3 = 0.07350824
            print 3
            val = b1 - b2 * Mp
            self._p = ( b1 - b2 * Mp - sqrt( val * val - 4 * b3 * Mp * Mp ) ) / ( 2 * b3 * Mp )
        elif Mp < 3/4:
            c1 = 0.3655157
            c2 = 0.6723532
            c3 = 0.033834
            self._p = ( c2 - sqrt( c2 * c2 + 4 * c3 * log( (3-4*Mp)/(4*c1) ) ) ) / ( 2 * c3 )
        else:
            print "MP %e > 3/4. There is no solution." %(Mp)
            self.clearAcc()
            return [ self._sigma, self._p, self._mean ]

        print 'UPDATE with %d samples' %(self._nSample)
        print 'sigma = %e, p = %e, mean = %e, MP = %e' %(self._sigma, self._p, self._mean, Mp)
        GGaussianD.fixConst( self )

        if filename != '' :
            GGaussianD.writeParam( self, filename )
        self.clearAcc()
        self._nItr += 1
        
        return [ self._sigma, self._p, self._mean ]

# @brief estimates paramters based on the maximum likelihood criterion.
class MLE4GGaussianD(GGaussianD):
    def __init__(self, p=0.7, sigma=0.0, zeroMean=True, mean=0.0, alpha = 0.05 ):
        GGaussianD.__init__(self, p, sigma, zeroMean, mean )

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
        self._acc1S   += power( val, self._p )
        
    def accP(self, Xi ):
        val = abs( Xi - self._mean )
        tmp = power( val / self._A, self._p )
        if val > 10e-12:
            self._acc1P += tmp * log( val / self._A )
        self._acc2P += tmp 
        
    def acc( self, Xi ):
        if self._fixed == False:
            print 'Parameters are not initialized correctly'
            return
        self.accSigma( Xi )
        self.accP( Xi )
        self._nSample += 1
        self._totalL  += GGaussianD.prob( self, Xi )
        
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
            return [self._sigma, self._p, self._converge]
        # for sigma
        val = power( ( self._p * self._acc1S ) / self._nSample, 1.0 / self._p )
        g1  = sf.gamma( 3.0 / self._p )
        g2  = sf.gamma( 1.0 / self._p )
        newSigma = sqrt( g1[0] / g2[0] ) * val

        # for p
        if sigmaOnly == False:
            dg1 = sf.psi( 1 + ( 1.0 /self._p ) )
            dg2 = sf.psi( 1.0 /self._p )
            dg3 = sf.psi( 3.0 /self._p )
            dLp = ( dg1[0] + 0.5 * dg2[0] - 1.5 * dg3[0] ) / ( self._p * self._p ) - ( self._acc1P + ( dg2[0] - 3 * dg3[0] ) * self._acc2P / ( 2 * self._p ) ) / self._nSample
            newP = self._p + dLp * ( self._alpha  / ( 1 + self._nItr ) )

        print 'UPDATE with %d samples' %(self._nSample)
        print 'AVG Log-Likelihood %e'  %(self._totalL/self._nSample)
        print 'sigma = %e -> %e ' %(self._sigma, newSigma)
        if sigmaOnly == False:
            print 'p     = %e -> %e ' %(self._p, newP)
            if abs( newP - self._p ) < self._thresh :
                self._converge = True
                print 'converged'
        if newSigma < self._floorV:
            print 'Flooring the variance %e to %e' %(newSigma,self._floorV)
            self._sigma = self._floorV
        else:
            self._sigma = newSigma
        if sigmaOnly == False:
            if newP < self._floorP:
                print 'Flooring p %e to %e' %(newP,self._floorP)
                self._p     = self._floorP
            else:
                self._p     = newP
        GGaussianD.fixConst( self )

        if filename != '' :
            GGaussianD.writeParam( self, filename )
        self.clearAcc()
        self._nItr += 1
        
        return [ self._sigma, self._p, self._converge ]

def func_ll( p, MLE4GGaussianD_rfPtr ):
    return 1.0
    
def dfunc_ll( p, MLE4GGaussianD_rfPtr ):
    return 1.0
    
def fdfunc_ll( p, objPtr ):
    y  = func_ll( p, objPtr )
    dy = dfunc_ll( p, objPtr )
    return y , dy

# @brief estimates paramters based on the maximum likelihood criterion.
class MLE4GGaussianD_secant(GGaussianD):
    def __init__(self, p=0.7, sigma=0.0, zeroMean=True, mean=0.0, alpha = 0.05 ):
        MLE4GGaussianD.__init__(self, p, sigma, zeroMean, mean, alpha )
        self._observations = []
        
    def getAccSigma(self ):        
        return self._acc1S

    def getSampleN(self):
        return self._nSample

    def getTotalLogLikelihood(self):
        return numpy.log( self._totalL )

    def acc( self, Xi ):
        if self._fixed == False:
            print 'Parameters are not initialized correctly'
            return        
        self.accSigma( Xi )
        self.accP( Xi )
        self._nSample += 1
        self._totalL  += GGaussianD.prob( self, Xi )
        self._observations.append( Xi )

    def calcTotalLogLikelihood(self,p):
        sumYp = 0
        for Xi in self._observations:
            sumYp += power( abs((Xi-self._mean)/self._A), p )
        lg1 = sf.lngamma( 1.0 / p )
        lg2 = sf.lngamma( 3.0 / p )
        lg3 = sf.lngamma( 1 + 1.0 / p )
        lA = numpy.log( self._sigma ) + 0.5 * ( lg1[0] - lg2[0] )
        
        
    def findP(self):
        mysys = roots.gsl_function_fdf( func_ll, dfunc_ll, fdfunc_ll, self )

        solver = roots.newton(mysys)
        #solver = roots.secant(mysys)
        #solver = roots.steffenson(mysys)
        p = self_p
        solver.set(p)

        print "# Using solver ", solver.name()
        ok = 1
        for iter in range(10):
            status = solver.iterate()
            p0 = p
            p = solver.root()
            status = roots.test_delta(p, p0, 0.0, 1e-5)
            r = solver.root()
            if status == errno.GSL_SUCCESS:
                print "#  Convereged :"
            print "%5d  %e  %e" %(iter, r, p - p0)
            if status == errno.GSL_SUCCESS:
                break
        else:
            raise ValueError, "Exeeded maximum number of iterations!"

        return

    def update( self, filename = '' ):    
        return

