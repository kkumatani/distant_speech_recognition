"""
 @file kalman.py
 @brief Implementations of conventional and extended Kalman filters.
 @author John McDonough and Kenichi Kumatani
"""

import scipy
import scipy.stats
import scipy.stats.distributions
from copy import *
from numpy import *
from numpy.linalg import *
from tdoa import *

# make changes
# Conventional Kalman filter
class KalmanFilter:

    def __init__(self, source, F, U, sigmaV2, sigmaK2, deltaT, initialXk = None, H = None, gateProbability = 0.0, boundaries = None):

        # initialize all parameters
        self.source			= source
        self.F				= deepcopy(F)
        self.H				= H
        self.U				= deepcopy(U)
        self.sigmaV2			= sigmaV2
        self.stateLength		= F.shape[0]
        self.I				= numpy.identity(self.stateLength, numpy.float)
        self.deltaT			= deltaT
        self.gateProbability		= gateProbability
        self.boundaries			= boundaries
        self.observed                   = False

        print 'Gate probability = %10.4f' %gateProbability

        if self.gateProbability == 0.0:
            self.innovationFilter	= False
        else:
            self.innovationFilter	= True
            self.chi			= scipy.stats.distributions.chi_gen(a=0.0,name='chi',shapes='df')

        self.K_filter			= sigmaK2 * numpy.identity(self.stateLength, numpy.float)
        self.K_predict			= sigmaK2 * numpy.identity(self.stateLength, numpy.float)

        self.lastUpdateT		= -1
        self.time			= -1

        if initialXk is None:
            self.xk_filter = zeros(self.stateLength, numpy.float)
        else:
            self.xk_filter = initialXk


    def isSourceObserved(self):
        return self.observed
    

    def withinRoom(self, x):
        if self.boundaries is None:
            return True

        for n in range(len(x)):
            if x[n] < self.boundaries[n][0] or x[n] > self.boundaries[n][1]:
                return False

        return True
                        

    # Determine if an innovation should be filtered
    def filterInnovation(self):
        df	= len(self.s)
        d2	= dot(self.s, dot(self.Sinv, self.s))
        chi2	= self.chi._cdf(d2, df)

        if chi2 > self.gateProbability:
            return True

        return False

    # Prediction step in KF update
    def predict(self):
        self.xk_predict = dot(self.F, self.xk_filter)

    def adjustBoundaries(self, xk_filter):
        theta		= xk_filter[0]
        if len(xk_filter) > 1:
            phi		= xk_filter[1]
        else:
            phi         = 0

        if theta < 0.0:
            theta	= -theta
            phi	       += numpy.pi
        elif theta > pi:
            theta      -= numpy.pi
            phi	       += numpy.pi

        while phi < -numpy.pi:
            phi	       += 2.0 * numpy.pi
        while phi > numpy.pi:
            phi	       -= 2.0 * numpy.pi

        xk_filter[0]	= theta
        if len(xk_filter) > 1:
            xk_filter[1]	= phi

        return xk_filter

    # This method performs the actual state update based on the given parameters
    def update(self, yk, elapsedTime):
        U 		= elapsedTime * elapsedTime * self.U
        self.K_predict	= dot(dot(self.F, self.K_filter), transpose(self.F)) + U
        self.S		= dot(dot(self.H, self.K_predict), transpose(self.H)) + self.sigmaV2 * numpy.identity(len(yk), numpy.float)

        # Calculate the innovation
        self.Sinv	= inv(self.S)
        self.yk_hat	= dot(self.H, self.xk_predict)
        self.s		= yk - self.yk_hat

        # Perform update or filter innovation?
        if self.filterInnovation():
            print 'Filtering innovation at time step %f' %self.time
            return

        self.G		= dot(dot(self.K_predict, transpose(self.H)), self.Sinv)
        xk_filter	= self.xk_predict + dot(self.G, self.s)

        # Store updated state estimate estimation error covariance matrix for next time step
        self.xk_filter	= self.adjustBoundaries(xk_filter)
        self.K_filter	= dot((self.I - dot(self.G, self.H)), self.K_predict)

        self.lastUpdateT = self.time


    def next(self):

        # Perform prediction and correction if new observation is available
        self.predict()
        observation = self.source.next()
        if not observation is None:
            elapsedTime = (self.time - self.lastUpdateT) * self.deltaT
            self.update(observation, elapsedTime)
            self.observed = True
        else:
            self.observed = False
        self.time += 1

        return self.xk_filter

    def __iter__(self):
        return self


class NotImplementedException(Exception):

    def __init__(self, errorMessage):
        print errorMessage


# Implement an extended Kalman filter in Cartesian coordinates
class ExtendedKalmanFilter(KalmanFilter):

    def __init__(self, source, F, U, sigmaV2, sigmaK2, deltaT, initialXk = None, gateProbability = 0.0, boundaries = None):
        # @param [obj]  tdoa.TDOAFeatureVector()
        KalmanFilter.__init__(self, source, F, U, sigmaV2, sigmaK2, deltaT, initialXk, gateProbability = gateProbability, boundaries = boundaries)

    # Perform prediction and correction if new observation is available
    def next(self, frameX):

        self.predict()
        observation = self.source.next(frameX)
        if not observation is None:

            self.H		= self.source.linearize(self.xk_predict, observation)
            observationLinear	= self.source.calcLinearizedObservation(self.xk_predict, self.H, observation)
            elapsedTime		= (self.time - self.lastUpdateT) * self.deltaT
            self.update(observationLinear, elapsedTime)
            self.observed = True
        else:
            self.observed = False

        self.time += 1

        return self.xk_filter
