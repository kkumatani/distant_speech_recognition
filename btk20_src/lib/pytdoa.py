"""
Calculating a time delay of arrival (TDOA) features for various microphone configurations.

 .. moduleauthor:: John McDonough and Kenichi Kumatani <k_kumatani@ieee.org>
"""
from btk20.feature import *
import numpy
from numpy.fft import *
from numpy import linalg

class PHATFeature:
    """
    Implement the phase transform for a given microphone pair
    """
    def __init__(self, src1, src2, fftlen, energy_threshold = 64):
        """
        Initialize a phase transformation feature

        :param src1: spectral src
        :type  src1: feature pointer object
        :param src2: spectral src
        :type  src2: feature pointer object
        :param fftlen: dimension of the spectral feature
        :type  fftlen: int
        """
        self._src1 = src1
        self._src2 = src2
        self._fftlen2 = fftlen // 2
        self._energy_threshold = energy_threshold
        self.reset()

    def next(self, frame_no):
        """
        compute the cross correlation of two spectral srcs

        :param frame_no: frame index
        :type  frame_no: int
        :returns: generalized cross-correlation (GCC)
        """
        block1 = self._src1.next(frame_no)
        block2 = self._src2.next(frame_no)
        if len(block1) > self._fftlen2+1:
            block1 = block1[0:self._fftlen2+1]
        if len(block2) > self._fftlen2+1:
            block2 = block2[0:self._fftlen2+1]

        energy1 = numpy.abs(numpy.inner(block1, numpy.conjugate(block1))) * 2
        energy2 = numpy.abs(numpy.inner(block2, numpy.conjugate(block2))) * 2
        if energy1 <= self._energy_threshold and energy2 <= self._energy_threshold:
            return numpy.zeros(1, numpy.float)

        cross_spectrum = block1 * numpy.conjugate(block2) / numpy.abs(block1 * numpy.conjugate(block2))

        return irfft(cross_spectrum)

    def __iter__(self):
        while True:
            yield self.next(self._isamp)
            self._isamp += 1

    def reset(self):
        self._isamp = 0
        self._src1.reset()
        self._src2.reset()

class TDOAFeature:
    """
    Perform peak picking to find the TDOA based on a cross-correlation criterion
    """
    def __init__(self, src, fftlen, samplerate):
        """
        Initialize the TDOA feature

        :param src: cross-correlation feature
        :type  src: feature pointer object
        :param fftlen: dimension of the spectral feature
        :type  fftlen: int
        :param samplerate: sampling rate
        :type  samplerate: int
        """
        self._src     = src
        self._fftlen  = fftlen
        self._fftlen2 = fftlen // 2
        self._Ts      = 1.0 / samplerate
        self.reset()

    def next(self, frame_no):
        """
        Find the cross-correlation peak

        :param frame_no: frame index
        :type  frame_no: int
        :returns: Tuple of time delay and GCC peak if there is a peak.
                  Otherwise, return (None, 0.0)
        """
        block = self._src.next(frame_no)

        highest_peak = 0.0
        for n, height in enumerate(block):
            height = abs(height)
            if height > highest_peak:
                if n < self._fftlen2:
                    delay = float(n) * self._Ts
                else:
                    delay = float(n - self._fftlen) * self._Ts

                # pack as [tau_d, peakHeight]
                best_peak    = [delay, height]
                highest_peak = height

        if highest_peak == 0.0:
            return [None, 0.0]
        else:
            return best_peak

    def __iter__(self):
        while True:
            yield self.next(self._isamp)
            self._isamp += 1

    def reset(self):
        self._isamp = 0
        self._src.reset()

class MicrophonePair:
    """
    Contain index integers to identify a microphone pair
    """
    def __init__(self, pairx, first_micx, second_micx):
        """
        :param pairx: index for a mic. pair (start from '0')
        :type  pairx: int
        :param first_micx: mic. index (start from '0')
        :type  first_micx: int
        :param second_micx: mic. index (start from '0')
        :type  second_micx: int
        """
        self.pairx       = pairx
        self.first_micx  = first_micx
        self.second_micx = second_micx


class MicrophonePairObservation(MicrophonePair):
    """
    Observation (TDOA vector) for each microphone pair
    """
    def __init__(self, pairx, first_micx, second_micx, observation):
        """
        :param pairx: index for a mic. pair (start from '0')
        :type  pairx: int
        :param first_micx: mic index (start from '0')
        :type  first_micx: int
        :param second_micx: mic index (start from '0')
        :type  second_micx: int
        :param observation: measurement such as time delay
        :type  observation: float
        """
        MicrophonePair.__init__(self, pairx, first_micx, second_micx)
        self.observation = observation


class MicrophonePairSource(MicrophonePair):
    """
    Time delay and GCC peak (TDOA Feature) for each microphone pair
    """
    def __init__(self, pairx, first_micx, second_micx, src):
        """
        :param pairx: index for a mic. pair (start from '0')
        :type  pairx: int
        :param first_micx: mic index (start from '0')
        :type  first_micx: int
        :param second_micx: mic index (start from '0')
        :type  second_micx: int
        :param src: time delay and GCC peak
        :type  src: feature pointer such as TDOAFeature()
        """
        MicrophonePair.__init__(self, pairx, first_micx, second_micx)
        self._src = src

    def next(self, frame_no):
        return self._src.next(frame_no)

    def reset(self):
        self._src.reset()


class TDOAFeatureVector:
    """
    Concatenate multiple TDOA estimates with a higher GCC peak than the threshold
    """
    def __init__(self, mic_pair_srcs, mpos, minimum_pairs = 2, threshold = 0.12, c = 343000.0):
        """
        Initialize a TDOA feature vecotr

        :param mic_pair_srcs: Feature pointers for microphone pairs
        :type  mic_pair_srcs: list of MicrophonePairSource
        :param mpos: microphone position matrix where each row vector represents a position vector
        :type  mpos: matrix[no. microphones][position vector dim.]
        :param minimum_pairs: minimum no. microphone pairs which have to have a higher GCC than the threshold
        :type  minimum_pairs: int
        :param threshold: CC threshold
        :type  threshold: float
        :param c: speed of sound (mm/seconds)
        :type  c: float
        """
        self._mic_pair_srcs = mic_pair_srcs
        self._mpos          = mpos
        self._minimum_pairs = minimum_pairs
        self._threshold     = threshold
        self._c             = c
        self.reset()

    def tdoa(self, mic_pair, x_cart):
        """
        Calculate a true TDOA based on predicted state estimate

        :param mic_pair: object that has two microphone indecies
        :type  mic_pair: MicrophonePairSource
        :param x_cart: three-dim vector in a Cartesian coordinate system
        :type  x_cart: float vector
        :returns: time delay of arrival
        """
        diff1 = x_cart - self._mpos[mic_pair.first_micx]
        diff2 = x_cart - self._mpos[mic_pair.second_micx]
        dist1 = numpy.sqrt(numpy.dot(diff1, diff1))
        dist2 = numpy.sqrt(numpy.dot(diff2, diff2))
        return (dist1 - dist2) / self._c

    def calc_linearized_observation(self, xk_predict, H, observations):
        """
        Calculating linearized observation

        :param xk_predict: predicted position
        :type  xk_predict: float vector
        :param H: linear prediction matrix
        :type  H: matrix[no. observations][position vector dim.]
        :param observations: list of observation objects for microphone pairs
        :type  observations: MicrophonePairObservation
        :returns: linearized observation
        """
        yk = numpy.zeros(len(observations), numpy.float)
        for n, mic_pair in enumerate(observations):
            yk[n] = observations[n].observation - (self.tdoa(mic_pair, xk_predict) - numpy.dot(H[n,:], xk_predict))

        return yk

    def linearize(self, xk_predict, observations):
        """
        Linearize the observation functional about xk_predict

        :param xk_predict: predicted position
        :type  xk_predict: float vector
        :param observations: list of observation objects for microphone pairs
        :type  observations: MicrophonePairObservation
        :returns: transform matrix that approximates a mapping function between the position and time delay
        """
        H = numpy.zeros([len(observations), len(xk_predict)], numpy.float)
        for rowx, obs in enumerate(observations):
            diff1 = xk_predict - self._mpos[obs.first_micx]
            diff2 = xk_predict - self._mpos[obs.second_micx]
            D1 = numpy.sqrt(numpy.dot(diff1, diff1))
            D2 = numpy.sqrt(numpy.dot(diff2, diff2))
            H[rowx,:] = ((diff1 / D1) - (diff2 / D2)) / self._c

        return H

    def next(self, frame_no):
        """
        Determine which microphone pair had a correlation peak above the threshold

        :returns: list of observation objects (MicrophonePairObservation)
        """
        observations = []
        self._tdoabuf = {}
        for mic_pair_src in self._mic_pair_srcs:
            [delay, cc_height] = mic_pair_src.next(frame_no)
            if not (mic_pair_src.first_micx in self._tdoabuf):
                self._tdoabuf[mic_pair_src.first_micx] = {}
            self._tdoabuf[mic_pair_src.first_micx][mic_pair_src.second_micx] = delay
            if cc_height > self._threshold:
                mic_pair_obs = MicrophonePairObservation(mic_pair_src.pairx, mic_pair_src.first_micx, mic_pair_src.second_micx, delay)
                observations.append(mic_pair_obs)

        if len(observations) < self._minimum_pairs:
            return None # None of hypos exceeds the threshold
        else:
            return observations

    def instantaneous_position(self, frame_no):
        """
        Should return a position estimate (without trajectory information)
        """
        pass

    def mic_pair_tdoa(self):
        return self._tdoabuf

    def __iter__(self):
        while True:
            yield self.next(self._isamp)
            self._isamp += 1

    def reset(self):
        self._isamp = 0
        for mic_pair_src in self._mic_pair_srcs:
            mic_pair_src.reset()

def are_collinear_and_consistent_direction(points):
    """
    Examine two conditions:
    1) All the points are on the same line (collinear), and
    2) The vector from the first point to each point points at the same direction

    :returns: True if points satisfy two conditions described above
    """
    shifted_points = numpy.array([points[i] - points[0] for i in range(len(points))])
    x0 = shifted_points[1]
    absx0 = numpy.sqrt(numpy.inner(x0, x0))
    for i in range(2, len(points)):
        xi = shifted_points[i]
        absxi = numpy.sqrt(numpy.inner(xi, xi))
        nip   = numpy.inner(x0, xi) / (absx0 * absxi)
        if abs(nip - 1) > 0.01:
            print('{}-th point is not collinear: norm inner prodct={}'.format(i, nip))
            return False
        elif (nip - 1) < 0:
            print('Inconsistent vector direction at {}-th point')
            return False

    return True

class FarfieldLinearArrayTDOAFeatureVector(TDOAFeatureVector):
    """
    Concatenate together those TDOAs whose CC peak exceeds a given threshold
    in the case of the linear microphone array under the far-field (FF) assumption
    """
    def __init__(self, mic_pair_srcs, mpos, minimum_pairs = 2, threshold = 0.12, c = 343000.0):
        """
        Initialize a TDOA feature for a linear array under the FF assumption

        :param mic_pair_srcs: Feature pointers for microphone pairs
        :type  mic_pair_srcs: list of MicrophonePairSource
        :param mpos: microphone position matrix where each row vector represents a direction
        :type  mpos: matrix[no. microphones][1]
        :param minimum_pairs: minimum no. microphone pairs which have to have a higher GCC than the threshold
        :type  minimum_pairs: int
        :param threshold: CC threshold
        :type  threshold: float
        :param c: speed of sound (mm/seconds)
        :type  c: float
        """
        TDOAFeatureVector.__init__(self, mic_pair_srcs, mpos, minimum_pairs, threshold, c)
        if False == are_collinear_and_consistent_direction(mpos):
            raise ValueError('Microphone positions have to be collinear and the 1st sensor position has to be the tail')

        # Project the xyz-coordinate position onto the same line and
        # translate each point w.r.t the first sensor position
        distances = numpy.zeros((len(mpos), 1), numpy.float)
        for i in range(1, len(mpos)):
            diff = mpos[i] - mpos[0]
            distances[i] = numpy.sqrt(numpy.dot(diff, diff))

        self._mpos = distances # update the microphone positions

    def tdoa(self, mic_pair, azimuth):
        """
        Calculate a true TDOA based on predicted state estimate
        """
        dist = self._mpos[mic_pair.second_micx] - self._mpos[mic_pair.first_micx]
        tau  = dist * numpy.cos(azimuth) / self._c
        return numpy.array([tau], numpy.float)

    def calc_linearized_observation(self, azimuthk_predict, H, observations):
        """
        Calculate the linearized time delay

        :returns: linearized observation (time delay vector)
        """
        yk = numpy.zeros(len(observations), numpy.float)
        for n, mic_pair in enumerate(observations):
            yk[n] = observations[n].observation - (self.tdoa(mic_pair, azimuthk_predict) - numpy.inner(H[n,:], azimuthk_predict))

        return yk

    def linearize(self, azimuthk_predict, observations):
        """
        Linearize the observation functional about azimuthk_predict
        """
        H = numpy.zeros([len(observations), len(azimuthk_predict)], numpy.float)
        for rowx, obs in enumerate(observations):
            dist      = self._mpos[obs.second_micx] - self._mpos[obs.first_micx]
            grad      = - dist * numpy.sin(azimuthk_predict) / self._c
            H[rowx,:] = numpy.array([grad], numpy.float)

        return H

    def next(self, frame_no):
        """
        Determine which microphone pairs had a correlation peak above the threshold
        """
        observation   = []
        self._tdoabuf = {}
        for mic_pair_src in self._mic_pair_srcs:
            [delay, cc_height] = mic_pair_src.next(frame_no)
            if not (mic_pair_src.first_micx in self._tdoabuf):
                self._tdoabuf[mic_pair_src.first_micx] = {}
            self._tdoabuf[mic_pair_src.first_micx][mic_pair_src.second_micx] = delay
            if cc_height > self._threshold:
                mic_pair_obs = MicrophonePairObservation(mic_pair_src.pairx, mic_pair_src.first_micx, mic_pair_src.second_micx, delay)
                observation.append(mic_pair_obs)

        if len(observation) < self._minimum_pairs:
            return None
        else:
            return observation

    def instantaneous_position(self, frame_no):
        """
        Obtain a position estimate (without trajectory information)

        :returns: position vector if there is a 'valid' TDOA estimate
        """
        sum_azimuth = 0.0
        num_hypos = 0
        for mic_pair_src in self._mic_pair_srcs:
            [delay, cc_height] = mic_pair_src.next(frame_no)
            if cc_height > self._threshold:
                dist = self._mpos[mic_pair_src.second_micx] - self._mpos[mic_pair_src.first_micx]
                val = delay * self._c / dist[0]
                if val < -1:
                    val = -1
                elif val > 1:
                    val = 1
                sum_azimuth += numpy.arccos(val)
                num_hypos += 1

        if num_hypos < self._minimum_pairs:
            return numpy.array([-1e10])

        return numpy.array([sum_azimuth / float(num_hypos)])


class FarfieldCircularArrayTDOAFeatureVector(TDOAFeatureVector):
    """
    Concatenate together those TDOAs whose cross-correlation peak is above a given threshold
    for the case of a circular microphone array under a far-field assumption
    """
    def __init__(self, mic_pair_srcs, mpos, minimum_pairs = 2, threshold = 0.12, c = 343000.0):
        """
        :param mpos: microphone positions
        :type  mpos: matrix[no. microphones][1]
        """
        if len(mpos) == 2:
            raise ValueError('No. microphones is only 2. Use FarfieldLinearArrayTDOAFeatureVector()')
        for i, mic_position in enumerate(mpos):
            assert len(mic_position) >= 2, '{}-th microphone: position vector dimension < 2'.format(i)

        TDOAFeatureVector.__init__(self, mic_pair_srcs, mpos, minimum_pairs, threshold, c)

    def tdoa(self, mic_pair, polarX):
        """
        Calculate true time delays of arrival based on predicted state estimate
        """
        theta  = polarX[0] # polar angle
        phi    = polarX[1] # azimuth
        offset = self._mpos[mic_pair.second_micx] - self._mpos[mic_pair.first_micx]
        u      = numpy.array([numpy.sin(theta) * numpy.cos(phi), numpy.sin(theta) * numpy.sin(phi), numpy.cos(theta)])
        tau    = numpy.dot(u, offset) / self._c
        return numpy.array([tau], numpy.float)

    def calc_linearized_observation(self, polarX, H, observations):
        """
        Calculate the linearized time delay
        """
        yk = numpy.zeros(len(observations), numpy.float)
        for n, mic_pair in enumerate(observations):
            yk[n] = observations[n].observation - (self.tdoa(mic_pair, polarX) - numpy.dot(H[n,:], polarX))

        return yk

    def linearize(self, polarX, observations):
        """
        Linearize the TDOA functional about polarX
        """
        theta = polarX[0] # polar angle
        phi   = polarX[1] # azimuth
        H = numpy.zeros([len(observations), len(polarX)], numpy.float)
        for rowx, obs in enumerate(observations):
            offset    = self._mpos[obs.second_micx] - self._mpos[obs.first_micx]
            du_dtheta = numpy.array([numpy.cos(theta) * numpy.cos(phi), numpy.cos(theta) * numpy.sin(phi), - numpy.sin(theta)])
            du_dphi   = numpy.array([- numpy.sin(theta) * numpy.sin(phi), numpy.sin(theta) * numpy.cos(phi), 0.0])
            dtau_dtheta = numpy.dot(du_dtheta, offset) / self._c
            dtau_dphi   = numpy.dot(du_dphi,   offset) / self._c
            H[rowx,:]   = numpy.array([dtau_dtheta, dtau_dphi])

        return H

    def next(self, frame_no):
        """
        Determine which microphone pairs had a correlation peak above the threshold
        """
        observation   = []
        self._tdoabuf = {}
        for mic_pair_src in self._mic_pair_srcs:
            [delay, cc_height] = mic_pair_src.next(frame_no)
            if not (mic_pair_src.first_micx in self._tdoabuf):
                self._tdoabuf[mic_pair_src.first_micx] = {}
            self._tdoabuf[mic_pair_src.first_micx][mic_pair_src.second_micx] = delay
            if cc_height > self._threshold:
                mic_pair_obs = MicrophonePairObservation(mic_pair_src.pairx, mic_pair_src.first_micx, mic_pair_src.second_micx, delay)
                observation.append(mic_pair_obs)

        if len(observation) < self._minimum_pairs:
            return None
        else:
            return observation

    def instantaneous_position(self, frame_no):
        """
        Obtain a position estimate (without trajectory information)

        :returns: position vector [polar angle, azimuth]
        """
        P = [] # relative position matrix: each row vector representing a sensor position
        D = [] # time delay vector: each row containing a time delay
        for mic_pair_src in self._mic_pair_srcs:
            [delay, cc_height] = mic_pair_src.next(frame_no)
            offset = self._mpos[mic_pair_src.second_micx] - self._mpos[mic_pair_src.first_micx]
            if cc_height > self._threshold:
                P.append(offset)
                D.append(delay)

        if len(D) < self._minimum_pairs:
            return numpy.array([-1e10, -1e10])

        P = numpy.array(P)
        D = numpy.array(D) * self._c
        A = numpy.dot(numpy.linalg.pinv(P), D) # must be [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
        for i in range(len(A)):
            if A[i] < -1:
                A[i] = -1
            elif A[i] > 1:
                A[i] = 1

        A2 = A * A

        # check if there is a mic pair on a plane not parallel to the xy-plane
        mic_pair_exists_on_non_parallel_xy_plane = numpy.count_nonzero(P[:,2])

        # compute polar angle (theta)
        cos_theta2 = 1 - A2[0] - A2[1]
        if mic_pair_exists_on_non_parallel_xy_plane == 0: # every mic pair is on a plane parallel to the xy-plane
            if cos_theta2 < 0: # There is no valid solution
                return numpy.array([-1e10, -1e10])
            theta = numpy.arccos(numpy.sqrt(cos_theta2))
        else:
            if (cos_theta2 + A[2]) >= 0:
                theta = numpy.arccos(numpy.sqrt(cos_theta2 + A[2]) / 2.0)
            else:
                theta = numpy.arccos(A[2])

        # compute azimuth (phi)
        if mic_pair_exists_on_non_parallel_xy_plane == 0:
            if (A2[0] + A2[1]) == 0: # There is no valid solution
                return numpy.array([-1e10, -1e10])
            cos_phi2 =  A2[0] / (A2[0] + A2[1])
            if cos_phi2 < 0: # There is no valid solution
                return numpy.array([-1e10, -1e10])
            phi = numpy.arccos(numpy.sqrt(cos_phi2))
        else:
            sum_cos_phi = 0.0
            num_valid_solutions = 0
            cos_phi2 =  A2[0] / (A2[0] + A2[1])
            if (A2[0] + A2[1]) != 0 and cos_phi2 >= 0:
                sum_cos_phi += numpy.sqrt(cos_phi2)
                num_valid_solutions += 1
            if A2[2] != 1:
                cos_phi2 = - A2[0] / (A2[2] - 1)
                if cos_phi2 >= 0:
                    sum_cos_phi += numpy.sqrt(cos_phi2)
                    num_valid_solutions += 1
                cos_phi2 = (A2[1] + A2[2] - 1) / (A2[2] - 1)
                if cos_phi2 >= 0:
                    sum_cos_phi += numpy.sqrt(cos_phi2)
                    num_valid_solutions += 1
            if num_valid_solutions == 0: # There is no valid solution
                return numpy.array([-1e10, -1e10])
            phi = numpy.arccos(sum_cos_phi / num_valid_solutions)

        return numpy.array([theta, phi])


def make_tdoa_front_end(array_type, pair_ids, spec_sources, fftlen, samplerate, mpos, energy_threshold, minimum_pairs, threshold, sspeed = 343000.0):
    """
    Instantiate a TDOA feature vector front-end for a specific array type

    :param array_type: 'linear', 'circular' or 'nf'
    :type array_type: string
    :param pair_ids: Pairs of channel IDs starting from '0'
    :type pair_ids: list of tuples of two integers
    :param spec_sources: Spectral sources such as FFT features
    :type spec_sources: feature object pointer
    :param fftlen:
    :type fftlen: int
    :param samplerate:
    :type samplerate: int
    :param mpos:
    :type mpos: matrix[no. microphones][]
    :param minimum_pairs:
    :type minimum_pairs: int
    :param threshold:
    :type threshold: float
    :param sspeed:
    :type sspeed: float
    :returns:
    """
    mic_pair_srcs = []
    for pairx, (pair0, pair1) in enumerate(pair_ids):
        assert pair0 >= 0 and pair1 >= 0, 'mic. pair ID has to be equal or greater than 0'
        phat         = PHATFeature(spec_sources[pair0], spec_sources[pair1], fftlen, energy_threshold)
        tdoa         = TDOAFeature(phat, fftlen, samplerate)
        mic_pair_src = MicrophonePairSource(pairx, pair0, pair1, tdoa)
        mic_pair_srcs.append(mic_pair_src)

    if array_type == 'linear':
        return FarfieldLinearArrayTDOAFeatureVector(mic_pair_srcs, mpos, minimum_pairs, threshold, sspeed)
    elif array_type == 'circular':
        return FarfieldCircularArrayTDOAFeatureVector(mic_pair_srcs, mpos, minimum_pairs, threshold, sspeed)
    elif array_type == 'planar':
        raise NotImplementedError('Array type %s is not yet supported' %array_type)

    return TDOAFeatureVector(mic_pair_srcs, mpos, minimum_pairs, threshold, sspeed)
