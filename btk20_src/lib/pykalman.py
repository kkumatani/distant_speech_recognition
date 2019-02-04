#!/usr/bin/python
"""
Implementations of conventional and extended Kalman filters.

.. moduleauthor:: John McDonough and Kenichi Kumatani <k_kumatani@ieee.org>
"""
try:
    import scipy
    import scipy.stats

    SCIPY_IMPORTED = True
except ImportError:
    SCIPY_IMPORTED = False

from copy import *
import numpy
from numpy import *
from numpy.linalg import *
from pytdoa import *

# make changes
class KalmanFilter:
    """
    Conventional Kalman filter
    """
    def __init__(self, source, F, U, sigmaV2, sigmaK2, time_delta, initialXk = None, H = None, gate_prob = 0.0, boundaries = None):
        """

        :param source: feature pointer containing an observation vector
        :param F: state transition matrix
        :param U: process noise covariance matrix
        :param sigmaV2: diagonal loading factor for measurement noise covariance matrix
        :param sigmaK2: initial Kalman gain
        :param time_delta:
        :param initialXk: initial state vector
        :param H:
        :param gate_prob: gate probability
        :param boundaries:
        """
        # initialize all parameters
        self.source       = source
        self.F            = deepcopy(F)
        self.H            = H
        self.U            = deepcopy(U)
        self.sigmaV2      = sigmaV2
        self.stateLength  = F.shape[0]
        self.I            = numpy.identity(self.stateLength, numpy.float)
        self.time_delta   = time_delta
        self.gate_prob    = gate_prob
        self.boundaries   = boundaries
        self.observed     = False

        if self.gate_prob == 0.0:
            self.innovationFilter	= False
        else:
            if SCIPY_IMPORTED == False:
                raise ImportError("Failed to import scipy for KalmanFiltering with gate option")
            self.innovationFilter	= True

        self.K_filter			= sigmaK2 * numpy.identity(self.stateLength, numpy.float)
        self.K_predict			= sigmaK2 * numpy.identity(self.stateLength, numpy.float)

        self.lastUpdateT		= -1
        self.time			= -1

        if initialXk is None:
            self.xk_filter = zeros(self.stateLength, numpy.float)
        else:
            self.xk_filter = initialXk

    def is_observed(self):
        return self.observed

    def within_room(self, x):
        if self.boundaries is None:
            return True

        for n in range(len(x)):
            if x[n] < self.boundaries[n][0] or x[n] > self.boundaries[n][1]:
                return False

        return True

    def calc_innovation(self, yk):
        """
        Calculate the innovation
        """
        self.S         = dot(dot(self.H, self.K_predict), transpose(self.H)) + self.sigmaV2 * numpy.identity(len(yk), numpy.float)

        self.Sinv   = inv(self.S)
        self.yk_hat = dot(self.H, self.xk_predict).flatten()
        self.s      = yk - self.yk_hat

    def filter_innovation(self):
        """
        Determine if an innovation should be filtered
        """
        df = len(self.s)
        d2 = dot(self.s, dot(self.Sinv, self.s))
        chi2 = scipy.stats.chi.cdf(d2, df)

        if chi2 > self.gate_prob:
            return True

        return False

    def predict(self):
        """
        Prediction step in KF update
        """
        self.xk_predict = dot(self.F, self.xk_filter)

    def adjust_boundaries(self, xk_filter):
        """
        Put a position estimate back into a boundary if it is out of the range
        """
        theta  = xk_filter[0]
        if len(xk_filter) > 1:
            phi = xk_filter[1]
        else:
            phi = 0

        if theta < 0.0:
            theta  = -theta
            phi   += numpy.pi
        elif theta > pi:
            theta -= numpy.pi
            phi   += numpy.pi

        while phi < -numpy.pi:
            phi += 2.0 * numpy.pi
        while phi > numpy.pi:
            phi -= 2.0 * numpy.pi

        xk_filter[0] = theta
        if len(xk_filter) > 1:
            xk_filter[1] = phi

        return xk_filter

    def update(self, yk, elapsed_time):
        """
        Performs the actual state update based on the given parameters

        :returns: True if update is done
        """
        self.K_predict = dot(dot(self.F, self.K_filter), transpose(self.F)) + elapsed_time * elapsed_time * self.U
        self.calc_innovation(yk)
        # Perform update or filter innovation?
        if self.filter_innovation():
            print('Filtering innovation at time step %f' %self.time)
            return False

        self.G    = dot(dot(self.K_predict, transpose(self.H)), self.Sinv)
        xk_filter = self.xk_predict + dot(self.G, self.s)

        # Store updated state estimate estimation error covariance matrix for next time step
        self.xk_filter = self.adjust_boundaries(xk_filter)
        self.K_filter  = dot((self.I - dot(self.G, self.H)), self.K_predict)

        self.lastUpdateT = self.time
        return True

    def next(self, frame_no):
        """
        Perform prediction and correction if new observation is available
        """
        self.predict()
        observation = self.source.next(frame_no)
        if not observation is None:
            elapsed_time = (self.time - self.lastUpdateT) * self.time_delta
            self.update(observation, elapsed_time)
            self.observed = True
        else:
            self.observed = False
        self.time += 1

        return self.xk_filter

    def set_time(self, frame_no):
        self.time = frame_no

    def __iter__(self):
        while True:
            yield self.next(self.time)


class ExtendedKalmanFilter(KalmanFilter):
    """
    Implement an extended Kalman filter in Cartesian coordinates
    """
    def __init__(self, source, F, U, sigmaV2, sigmaK2, time_delta, initialXk = None, gate_prob = 0.0, boundaries = None):
        """
        :param source: TDOA feature pointer
        :type source: TDOAFeatureVector()
        """
        KalmanFilter.__init__(self, source, F, U, sigmaV2, sigmaK2, time_delta, initialXk, gate_prob = gate_prob, boundaries = boundaries)

    def next(self, frame_no):
        """
        Perform prediction and correction if new observation is available
        """
        self.predict()
        observation = self.source.next(frame_no)
        if not observation is None:
            self.H            = self.source.linearize(self.xk_predict, observation)
            observationLinear = self.source.calc_linearized_observation(self.xk_predict, self.H, observation)
            elapsed_time      = (self.time - self.lastUpdateT) * self.time_delta
            self.update(observationLinear, elapsed_time)
            self.observed = True
        else:
            self.observed = False

        self.time += 1

        return self.xk_filter


class IteratedExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Implement an iterated extended Kalman filter in Cartesian coordinates
    """
    def __init__(self, source, F, U, sigmaV2, sigmaK2, time_delta, initialXk = None, gate_prob = 0.0, boundaries = None, num_iterations = 3, iteration_threshold = 1e-4):
        """
        :param source: TDOA feature pointer
        :type source: TDOAFeatureVector()
        """
        ExtendedKalmanFilter.__init__(self, source, F, U, sigmaV2, sigmaK2, time_delta, initialXk, gate_prob = gate_prob, boundaries = boundaries)
        self.num_iterations = num_iterations
        self.iteration_threshold  = iteration_threshold

    def update(self, yk, elapsed_time):
        """
        Performs the actual state update based on the given parameters

        :returns: True if update is done
        """
        eta = self.xk_predict
        self.K_predict = dot(dot(self.F, self.K_filter), transpose(self.F)) + elapsed_time * elapsed_time * self.U
        # Peform the local iterations
        for i in range(self.num_iterations):
            self.calc_innovation(yk)
            # Perform update or filter innovation?
            if self.filter_innovation():
                print('Filtering innovation at time step %f' %self.time)
                return False

            self.G    = dot(dot(self.K_predict, transpose(self.H)), self.Sinv)
            zeta = self.s
            if i > 0:
                zeta -= dot(self.H, (self.xk_predict - eta))

            eta_prev = eta
            eta = self.xk_predict + dot(self.G, zeta)
            # Check the convergence
            diff = eta - eta_prev
            if inner(diff, diff) < self.iteration_threshold:
                break

        xk_filter = eta
        # Store updated state estimate estimation error covariance matrix for next time step
        self.xk_filter = self.adjust_boundaries(xk_filter)
        self.K_filter  = dot((self.I - dot(self.G, self.H)), self.K_predict)

        self.lastUpdateT = self.time
        return True
