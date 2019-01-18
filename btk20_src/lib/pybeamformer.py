#!/usr/bin/python
"""
A collection of Python functions for subband beamforming

.. moduleauthor:: Kenichi Kumatani <k_kumatani@ieee.org>
"""
import pickle, copy
import numpy

from btk20.common import *
from btk20.stream import *
from btk20.feature import *
from btk20.modulated import *
from btk20.beamformer import *


def calc_la_delays(mpos, azimuth, sspeed = 343740.0, ref_micx = None):
    """
    Calculate the delays of the linear array to focus the beam on (azimuth) under the far-field assumption

    :param mpos: distance between each microphone and the reference microphone
    :type  mpos: M x 1 float matrix where M is no. microphones
    :param azimuth: direction of interest
    :type  azimuth: float
    :param sspeed: speed of sound
    :type  sspeed: float
    :param ref_micx: the index of the reference microphone
    :type  ref_micx: integer
    """

    chanN  = len(mpos)
    # Normalize by delay of the reference microphone
    if ref_micx is None:
        ref_micx = chanN//2

    delays = numpy.zeros(chanN, numpy.float)
    for i in range(chanN):
        delays[i] = - mpos[i][0] * numpy.cos(azimuth) / sspeed

    return (delays - delays[ref_micx])


def calc_pa_delays(mpos, azimuth, polar_angle, sspeed = 343740.0, ref_micx = None):
    """
    Calculate the delays of the planar array to focus the beam on (azimuth, polar_angle) under the far-field assumption

    :param mpos: contains xy coordinate position from the origin (reference microphone position)
    :type  mpos: M x 2 float matrix where M is no. microphones
    :param azimuth: azimuth of interest
    :type  azimuth: float
    :param polar_angle: polar angle of interest
    :type  polar_angle: float
    :param sspeed: speed of sound
    :type  sspeed: float
    :param ref_micx: the index of the reference microphone
    :type  ref_micx: integer
    """

    chanN  = len(mpos)
    # Normalize by delay of the reference microphone
    if ref_micx is None:
        ref_micx = chanN//2

    delays = numpy.zeros(chanN, numpy.float)
    for i in range(chanN):
        dx = (mpos[i][0] - mpos[ref_micx][0])
        dy = (mpos[i][1] - mpos[ref_micx][1])
        delays[i] = - (dx * numpy.cos(azimuth) * numpy.sin(polar_angle) + dy * numpy.sin(azimuth) * numpy.sin(polar_angle))/ sspeed

    return delays


def calc_ca_delays(mpos, azimuth, polar_angle, sspeed = 343740.0):
    """
    Calculate the delays to focus the beam on (azimuth, polar_angle) with a circular array under the far-field assumption

    :param mpos: contains xyz coordinate position from the origin (the center of an array)
    :type  mpos: M x 3 float matrix where M is no. microphones
    :param azimuth: azimuth of interest
    :type  azimuth: float
    :param polar_angle: polar angle of interest
    :type  polar_angle: float
    :param sspeed: speed of sound
    :type  sspeed: float
    :param ref_micx: the index of the reference microphone
    :type  ref_micx: integer
    """
    chanN  = len(mpos)
    delays = numpy.zeros(chanN, numpy.float)
    c_x = - numpy.sin(polar_angle) * numpy.cos(azimuth)
    c_y = - numpy.sin(polar_angle) * numpy.sin(azimuth)
    c_z = - numpy.cos(polar_angle)
    for i in range(chanN):
        delays[i] = (c_x * mpos[i][0] + c_y * mpos[i][1] + c_z * mpos[i][2]) /sspeed

    return delays


def calc_nf_delays(mpos, x, y, z, sspeed = 343740.0, ref_micx = None):
    """
    Calculate the delays to focus the beam on (x,y,z) under the near-field assumption

    :param mpos: contains xyz coordinate position from the origin (the center of an array)
    :type  mpos: M x 3 float matrix where M is no. microphones
    """
    chanN  = len(mpos)
    # Normalize by delay of the reference microphone
    if ref_micx is None:
        ref_micx = chanN//2

    delays = numpy.zeros(chanN, numpy.float)
    for i in range(chanN):
        delays[i] = numpy.sqrt((x - mpos[i][0])**2+(y - mpos[i][1])**2+(z - mpos[i][2])**2) / sspeed

    return (delays - delays[ref_micx])


def calc_delays(array_type, mpos, position, sspeed = 343740.0, ref_micx = None):
    """
    wrapper function for time delay calculation
    """
    if array_type == 'linear':
        return calc_la_delays(mpos, position[0], sspeed = sspeed, ref_micx = ref_micx)
    elif array_type == 'planar':
        return calc_pa_delays(mpos, position[0], position[1], sspeed = sspeed, ref_micx = ref_micx)
    elif array_type == 'circular':
        return calc_ca_delays(mpos, position[0], position[1], sspeed = sspeed, ref_micx = ref_micx)

    return calc_nf_delays(mpos, position[0], position[1], position[2], sspeed = sspeed, ref_micx = ref_micx)


class SpectralSource:
    def __init__(self, sample_feat, fftlen, shiftlen):
        self._sample_feat = sample_feat
        self._fftlen   = fftlen
        self._fftlen2  = fftlen / 2
        self._shiftlen = shiftlen

    def __iter__(self):
        pass

    def shiftlen(self):
        return self.shiftlen

    def next_utt(self, input_audio_path, samplerate):
        self._sample_feat.read(input_audio_path, samplerate)

    def reset(self):
        self._sample_feat.reset()

    def size(self):
        return self._fftlen


class AnalysisFB(SpectralSource):
    """
    Perform subband analysis on a sound source.
    """
    def __init__(self, sample_feat, filter_coeffs, M,  m, r):
        """
        Initialize the analysis filter bank.

        :param sample_feat: time-domain signal source
        :type sample_feat: stream pointer object
        :param filter_coeffs: analysis filter coefficients
        :type filter_coeffs: float vector
        :param M: Number of subbands
        :type M: integer
        :param m: Filter length factor
        :type m: integer
        :param r: Exponential decimation factor
        :type r: integer
        :returns:
        """
        SpectralSource.__init__(self, sample_feat, M, M / 2**r)
        self._afb = OverSampledDFTAnalysisBankPtr(sample_feat, prototype = filter_coeffs, M = M, m = m, r = r, delay_compensation_type = 2)
        self._h = filter_coeffs

    def __iter__(self):
        """Return the next spectral sample."""
        while True:
            try:
                yield numpy.array(self._afb.next())
            except StopIteration:
                raise StopIteration

    def reset(self):
        """Set up to process the next utterance."""
        self._afb.reset()

    def size(self):
        return self._afb.size()


class FileSpectralSource(SpectralSource):
    """
    Read spectral samples from a file.
    """
    def __init__(self, fftlen, shiftlen):
        """Initialize the analysis filter bank."""
        SpectralSource.__init__(self, None, fftlen, shiftlen)

    def __iter__(self):
        for spec in self._spec_samples:
            yield spec

    def next_utt(self, filename):
        """Set up to process the next utterance."""
        with open(filename, 'r') as fp:
            self._spec_samples = load(fp)

    def reset(self):
        """Set up to process the next utterance."""
        self._spec_samples = None


class MultiChannelSource:
    """
    Handle multi-channel data: output of multiple analysis filter banks.

    :note:
    D&S beamforming == multiplying frequency-domain snapshot by conjugate of array manifold vector
    """
    def __init__(self, spec_sources):
        self._spec_sources = spec_sources # such as multiple analysis filter bank instances
        self._chan_num     = len(spec_sources)
        self._fftlen       = spec_sources[0].size()
        self._snapshot_array = SnapShotArrayPtr(self._fftlen, self._chan_num)

    def get_snapshot(self, fbin_no):
        """
        Obtain the array snapshot at a certain frequency

        :returns: snapshot of the current frame at the frequency bin 'fbin_no'
        """
        return self._snapshot_array.snapshot(fbin_no)

    def update_snapshot_array(self, chan_no = None):
        """
        Update frequency domain snapshots with new sensor outputs.

        :returns: the signal energy for channel 'chan_no'.
        """
        for chanX, afb in enumerate(self._spec_sources):
            subbands = numpy.array(afb.next())
            self._snapshot_array.set_samples(subbands, chanX)
            if chan_no == chanX:
                sigmaK = abs(numpy.dot(numpy.conjugate(subbands), subbands))

        self._snapshot_array.update()
        return sigmaK

    def reset(self):
        for spec_source in self._spec_sources:
            spec_source.reset()


def calc_array_manifold_f(fbinX, fftlen, samplerate, delays, half_band_shift):
    """
    Calculate one (conjugate) array manifold vector for each frequancy bin or subband.
    """
    chan_num = len(delays)
    Delta_f = samplerate / float(fftlen)
    J = (0+1j)
    fftlen2 = fftlen / 2

    if half_band_shift:
        if fbinX < fftlen2:
            phases = - J * 2.0 * numpy.pi * (0.5 + fbinX) * Delta_f * delays
            vs = numpy.exp( phases )
        else:
            phases = - J * 2.0 * numpy.pi * (0.5 - fftlen + fbinX) * Delta_f * delays
            vs = numpy.exp( phases )
    else:
        if fbinX <= fftlen2:
            vs = numpy.exp(- J * 2.0 * numpy.pi * fbinX * Delta_f * delays)
        else:
            vs = numpy.conjugate(numpy.exp(- J * 2.0 * numpy.pi * fbinX * Delta_f * delays))

    return vs / chan_num


def calc_blocking_matrix(vs , Nc = 1):
    """
    Calculate the blocking matrix for a distortionless beamformer,

    :param vs: array manifold vector
    :type vs: vector
    :param Nc: number of constraints
    :type Nc: integer
    :returns: Hermitian transpose of the blocking matrix (blocMat^H * vs ==0).
    """
    vsize    = len(vs)
    bsize    = vsize - Nc
    blockMat = numpy.zeros((vsize,bsize), numpy.complex)

    # Calculate the perpendicular projection operator 'PcPerp' for 'vs'.
    norm_vs  = numpy.inner( vs, numpy.conjugate(vs) )
    if abs(norm_vs) > 0.0:
        PcPerp   = numpy.eye(len(vs)) - numpy.outer( numpy.conjugate(vs), vs ) / norm_vs

        # Do Gram-Schmidt orthogonalization on the columns of 'PcPerp'.
        for idim in range(bsize):
            vec      = PcPerp[:,idim]
            for jdim in range(idim):
                rvec = blockMat[:,jdim]
                ip   = numpy.inner(numpy.conjugate(rvec), vec)
                vec -= rvec * ip
            norm_vec = numpy.sqrt( abs(numpy.inner(numpy.conjugate(vec),vec)) )
            blockMat[:,idim] = vec / norm_vec

    # Debugging:
    # print (numpy.dot(vs, blockMat))

    return blockMat


def calc_lcmv_weight(W_t, W_j):
    """
    calculate the weights of the LCMV beamformer subsuming null-steering beamformers

    :param W_t[numTargets][nChan]: Array manifold vector(s) for target source(s)
    :type W_t: complex matrix
    :param W_j[numJammers][nChan]: Array manifold vector(s) for jammer source(s)
    :type W_j: complex matrix
    :returns: beamformer weights that place null toward jammer sources while maintaining distortionless constraints for directions of interest
    """

    num_targets = len(W_t)
    num_jammers = len(W_j)
    chan_num = len(W_t[0])
    assert chan_num == len(W_j[0]), 'Inconsisten no. channels: %d != %d' %(chan_num, len(W_j[0]))

    Nc = num_targets + num_jammers # no. constraints
    Ct = numpy.zeros((Nc, chan_num), numpy.complex) # Combined array manifold vectors
    g  = numpy.zeros(Nc, numpy.float) # Geometric constraint: 1 for target source(s), 0 for interfering position(s)
    n = 0
    for targX in range(num_targets):
        Ct[n] = w_t[targX]
        g[n]  = 1
        n +=1
    for jammerX in range(num_jammers):
        Ct[n] = w_j[jammerX]
        n +=1

    CH  = numpy.conjugate(Ct)
    inverse = numpy.linalg.inv(numpy.dot(CH, C))
    V  = numpy.dot(inverse, g)
    wq = numpy.dot(C, V)

    return wq


class SubbandBeamformer:
    """
    Basic class for subband beamformer
    """
    def __init__(self, spec_sources):
        """
        Initialize the subband beamformer.
        """
        self._spec_sources = spec_sources # such as multiple analysis filter bank instances
        self._chan_num     = len(spec_sources)
        self._fftlen       = spec_sources[0].size()
        self._fftlen2      = self._fftlen // 2
        for c in range(1, self._chan_num):
            assert self._fftlen == spec_sources[c].size(), "%d-th channel: inconsistent FFT length" %c

        # beamformer instance
        self._beamformer = None

    def beamformer(self):
        return self._beamformer

    def __iter__(self):
        """
        :returns: the beamformer output.
        """
        if self._beamformer is None:
            raise NotImplementedError("Undefined beamformer object")

        while True:
            yield numpy.array(self._beamformer.next())

    def reset(self):
        if self._beamformer is None:
            raise NotImplementedError("Undefined beamformer object")

        self._beamformer.reset()

    def next_speaker(self):
        pass

    def size(self):
        return self._fftlen

    def save_active_weights(self, filename):
        with open(filename, 'w') as fp:
            pickle.dump(self._waH, fp, 1)

    def load_active_weights(self, filename):
        with open(filename, 'r') as fp:
            self._waH = pickle.load(fp)

        assert self._fftlen == len(self._waH), "Invalid FFT length of the active weight vector: %d != %d" %(self._fftlen,len(self._waH))
        assert self._Nc == (len(self._waH[0]) - self._chan_num), "Invalid no. constraints: %d != %d" %(self._Nc, len(self._waH[0]) - self._chan_num)
        self.set_active_weights()

    def set_active_weights(self):
        if self._beamformer is None:
            raise NotImplementedError("Undefined beamformer object")

        for fbinX in range(self._fftlen2+1):
            packed_wa = numpy.zeros(2 * (self._chan_num - self._Nc), numpy.float)
            for i in range(self._chan_num - self._Nc):
                packed_wa[2*i]   = numpy.real(self._waH[fbinX][i])
                packed_wa[2*i+1] = numpy.imag(self._waH[fbinX][i])
            self._beamformer.set_active_weights_f(fbinX, packed_wa)


class SubbandGSCBeamformer(SubbandBeamformer):
    """
    Beamformer including delay-and-sum and linear constrained minimum variance (LCMV) beamforming
    """
    def __init__(self, spec_sources, Nc = 1):
        """
        Initialize the subband beamformer.

        :param spec_sources: spectral sources such as OverSampledDFTAnalysisBankPtr()
        :type spec_sources: multiple complex stream objects
        :param Nc: no. linear constraints, total no. of look and null-steering directions
        :type Nc: integer
        """
        SubbandBeamformer.__init__(self, spec_sources)

        # beamformer instance
        self._beamformer = SubbandGSCPtr(fftlen=self._fftlen, half_band_shift=False)
        # set spectral sources to the beamformer object
        for source in self._spec_sources:
            self._beamformer.set_channel(source)

        # initialize the active weight vector
        self._Nc  = Nc # no. constraints
        self._waH = numpy.zeros((self._fftlen, self._chan_num - self._Nc), numpy.complex)

    def calc_beamformer_weights(self, samplerate, delays, update_active_weights = True):
        """
        Compute the GSC beamformer weight given the target direction only

        :param samplerate: sampling rate
        :type samplerate: integer
        :param delays: time delays for the target source
        :type delays: float vector
        """
        self._beamformer.calc_gsc_weights(samplerate, delays)
        if update_active_weights  == True:
            self.set_active_weights()

    def calc_beamformer_weights_n(self, samplerate, delays_t, delays_js, update_active_weights = True):
        """
        Compute the GSC beamformer weight given the target direction only

        :param samplerate: sampling rate
        :type samplerate: integer
        :param delays_t: time delays for the target source
        :type delays_t: float vector
        :param delays_js: time delays for the jammer(s)
        :type delays_js: float matrix
        """
        assert (self._Nc-1) == len(delays_js), 'Mismatch between no. constraints and no. jammers'

        self._beamformer.calc_gsc_weights_n(samplerate, delays_t, delays_js, self._Nc)
        if update_active_weights  == True:
            self.set_active_weights()


class SubbandMVDRBeamformer(SubbandBeamformer):
    """
    Beamformer for processing multi-channel spectral samples.
    """
    def __init__(self, spec_sources, Nc = 1):
        """
        Initialize the subband beamformer.

        :param Nc: no. linear constraints, total no. of look and null-steering directions
        :type Nc: integer
        """
        SubbandBeamformer.__init__(self, spec_sources)

        # beamformer instance
        self._beamformer = SubbandMVDRGSCPtr(fftlen=self._fftlen, half_band_shift=False)
        # set spectral sources to the beamformer object
        for source in self._spec_sources:
            self._beamformer.set_channel(source)

        # initialize the active weight vector
        self._Nc  = Nc # no. constraints
        self._waH = numpy.zeros((self._fftlen, self._chan_num - self._Nc), numpy.complex)

    def calc_sd_beamformer_weights(self, samplerate, delays, mpos, sspeed = 343740.0, mu = 0.01, update_active_weights = True):
        """
        Compute super-directive beamformer weight

        :param samplerate: sampling rate
        :type samplerate: integer
        :param delays: time delays for the target source
        :type delays: float vector
        :param mpos: matrix that specifies the array geometry
        :type mpos: M x 3 float matrix where M is no. microphones
        :param sspeed: speed of sound
        :type sspeed: float
        :param mu: diagonal loading
        :type mu : float
        """

        self._beamformer.calc_array_manifold_vectors(samplerate, delays)
        # Set the diffuse noise covariance matrix
        self._beamformer.set_diffuse_noise_model(mpos, samplerate, sspeed)
        self._beamformer.set_all_diagonal_loading(mu)
        self._beamformer.calc_mvdr_weights(samplerate, dthreshold = 1.0E-8, calc_inverse_matrix = True)
        if update_active_weights  == True:
            self.set_active_weights()


class SubbandGSCLMSBeamformer(SubbandBeamformer):
    """
    Leaky version of the least mean square (LMS) beamformer implemented in
    generalized sidelobe canceller (GSC) configuration with
    fixed diagonal loading. Also known as the leaky version of the Griffiths-Jim
    beamformer.

    :note: pure python implementation
    """
    def __init__(self, spec_sources,
                 beta = 0.97,
                 gamma = 0.01,
                 init_diagonal_load = 1.0E+6,
                 energy_floor = 90,
                 sil_thresh = 1.0E+8,
                 max_wa_l2norm = 100.0,
                 min_frames = 128,
                 slowdown_after = 1024,
                 Nc = 1):
        """
        Initialize the subband Griffiths-Jim beamformer.
        """
        SubbandBeamformer.__init__(self, spec_sources)
        # obtain the multi-channel subband input to process
        self._array_source = MultiChannelSource(spec_sources)
        self._Nc = Nc # No. linear constraints

        # weight vectors for the beamformer
        # conjugate array manifold vectors
        self._wqH = numpy.ones((self._fftlen2+1, self._chan_num), numpy.complex)
        # Hermitian transpose blocking matrices
        self._BmH =[numpy.zeros((self._chan_num - self._Nc, self._chan_num), numpy.complex) for m in range(self._fftlen2+1)]

        # parameters for active weight estimation
        self._beta               = beta # forgetting factor for the signal power
        self._init_gamma         = gamma # step size factor
        self._init_diagonal_load = init_diagonal_load # diagonal loading for active weight optimization
        self._energy_floor       = energy_floor # the floor value for the energy
        self._sil_thresh         = sil_thresh
        self._max_wa_l2norm      = max_wa_l2norm # the upper boundary of the active weight vector norm |waK|
        self._min_frames         = min_frames # minimum no. frames for adaptation
        self._slowdown_after     = slowdown_after # after this number of frames, decrease the step size
        self._isamp = 0
        self.reset_stats()

        # for debugging
        self._subband_no_printed = set([]) # print stats for these subband indecies set([100])

    def __iter__(self):
        """
        Return the next spectral sample.
        """

        while True:
            energy = self._array_source.update_snapshot_array(chan_no = 0) / self._fftlen
            output = numpy.zeros(self._fftlen, numpy.complex)

            if self._isamp > 0 and (self._isamp % self._slowdown_after) == 0:
                self._gamma /= 2.0
                print 'Setting gamma = %g after sample %d' %(self._gamma, self._slowdown_after)

            if energy > (self._energy/self._sil_thresh):
                self._ttl_updates += 1
            for m in range(self._fftlen2+1):
                XK = self._array_source.get_snapshot(m)
                # Get next snapshot and form output of blocking matrix.
                ZK = numpy.dot(self._BmH[m], XK)
                # Calculate output of upper branch.
                YcK = numpy.dot(self._wqH[m], XK)

                # Measure the power of each subband
                if self._isamp > 0 :
                    subband_energy = self._subband_energy[m] * self._beta + (1.0 - self._beta) * abs(numpy.dot(numpy.conjugate(XK), XK))
                else:
                    subband_energy = abs(numpy.dot(numpy.conjugate(XK), XK))

                if subband_energy < self._energy_floor:
                    subband_energy = self._energy_floor

                if energy > (self._energy/self._sil_thresh):
                    epa = YcK - numpy.dot(self._waH[m], ZK)
                    # Divide the step size by the subband energy
                    alphaK = self._gamma / subband_energy
                    # Update active weight vector.
                    watHK  = self._waH[m] + epa * numpy.conjugate(ZK) * alphaK
                    norm_watK = abs(numpy.dot(watHK, numpy.conjugate(watHK)))
                    if norm_watK > self._max_wa_l2norm:
                        cK = numpy.sqrt(self._max_wa_l2norm / norm_watK)
                        waHK = cK * watHK
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if m in self._subband_no_printed:
                        print ''
                        print 'Sample %d: Band %d' %(self._isamp, m)
                        print 'Subband power = %8.4e' %(subband_energy)
                        print 'Total power   = %8.4e' %(energy)
                        print 'Avg. power    = %8.4e' %(self._energy)
                        norm_waK = abs(numpy.dot(waHK, numpy.conjugate(waHK)))
                        print '||waK||^2     = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)

                    # Store values for next iteration
                    self._waH[m]            = copy.deepcopy(waHK)
                    self._subband_energy[m] = subband_energy

                # Calculate complete array output.
                if self._isamp >= self._min_frames:
                    output[m] = YcK - numpy.dot(self._waH[m], ZK)
                else:
                    output[m] = YcK
                if m > 0 and m < self._fftlen2:
                    output[self._fftlen - m] = numpy.conjugate(output[m])

            # Update the average power for all subbands
            self._energy = self._energy * self._beta + (1.0 - self._beta) * energy

            yield output
            self._isamp += 1

    def calc_beamformer_weights(self, samplerate, delays):
        """
        compute the quiescent weight and blocking matrix
        """
        for m in range(self._fftlen2+1):
            vs = calc_array_manifold_f(m, self._fftlen, samplerate, delays, half_band_shift=False)
            self._BmH[m] = numpy.transpose(calc_blocking_matrix(vs , self._Nc))
            self._wqH[m] = numpy.conjugate(vs)

    def reset_stats(self):
        """
        Reset weight vectors for a new environment.
        """

        if self._isamp > 0:
            print 'Updated weight vectors on %d of %d total frames.' %(self._ttl_updates, self._isamp)

        self._isamp       = 0 # no. frames processed already
        self._ttl_updates = 0 # no. frames used for active weight estimation
        self._gamma       = self._init_gamma
        self._energy      = self._init_diagonal_load
        self._subband_energy = numpy.array([self._init_diagonal_load for k in range(self._fftlen2+1)])
        self._waH         = numpy.zeros((self._fftlen2+1, self._chan_num - self._Nc), numpy.complex)

    def reset(self):
        self._array_source.reset()
        self.reset_stats()


class SubbandGSCRLSBeamformer(SubbandBeamformer):
    """
    Recursive least squares (RLS) beamformer implemented in
    generalized sidelobe canceller (GSC) configuration with
    fixed diagonal loading.

    :note: pure python implementation
    """
    def __init__(self, spec_sources,
                 beta = 0.97,
                 gamma = 0.1,
                 mu = 0.97,
                 init_diagonal_load = 1.0E+6,
                 fixed_diagonal_load = 0.01,
                 sil_thresh = 1.0E+8,
                 constraint_option = 3,
                 alpha2 = 10.0,
                 max_wa_l2norm = 100.0,
                 min_frames = 128,
                 slowdown_after = 1024,
                 Nc = 1):
        """
        Initialize the subband Griffiths-Jim beamformer.
        """
        SubbandBeamformer.__init__(self, spec_sources)
        # obtain the multi-channel subband input to process
        self._array_source = MultiChannelSource(spec_sources)
        self._Nc = Nc

        # weight vectors for the beamformer
        # conjugate array manifold vectors
        self._wqH = numpy.ones((self._fftlen2+1, self._chan_num), numpy.complex)
        # Hermitian transpose blocking matrices
        self._BmH =[numpy.zeros((self._chan_num - self._Nc, self._chan_num), numpy.complex) for m in range(self._fftlen2+1)]

        # parameters for active weight estimation
        self._beta               = beta  # forgetting factor for the signal power
        self._gamma              = gamma # step size factor
        self._mu                 = mu # recursive weight for covariance matrix estimation
        self._init_diagonal_load = init_diagonal_load # initial divisor for Pz
        self._fixed_diagonal_load = fixed_diagonal_load # ad-hoc diagonal loading for active weight vector update
        self._sil_thresh         = sil_thresh
        self._constraint_option  = constraint_option # 0:no constraint, 1: quadratic constraint, 2: norm normalization, 3: both
        self._alpha2             = alpha2         # threshold for the quadratic constraint
        self._max_wa_l2norm      = max_wa_l2norm  # threshold to the norm normalization
        self._min_frames         = min_frames     # minimum no. frames for adaptation
        self._slowdown_after     = slowdown_after # after this number of frames, decrease the step size
        self._isamp = 0
        self.reset_stats()

        # for debugging
        self._subband_no_printed = set([]) # print stats for these subband indecies set([100])

    def __iter__(self):
        """
        Return the next spectral sample.
        """

        while True:
            energy = self._array_source.update_snapshot_array(chan_no = 0) / self._fftlen
            output = numpy.zeros(self._fftlen, numpy.complex)

            if energy > (self._energy/self._sil_thresh):
                self._ttl_updates += 1
            for m in range(self._fftlen2+1):
                XK = self._array_source.get_snapshot(m)
                # Get next snapshot and form output of blocking matrix.
                ZK = numpy.dot(self._BmH[m], XK)
                # Calculate output of upper branch.
                YcK = numpy.dot(self._wqH[m], XK)

                if energy > (self._energy/self._sil_thresh):
                    # Calculate gain vector and update precision matrix.
                    PzZ   = numpy.dot(self._Pz[m], ZK)
                    ip    = numpy.dot(numpy.conjugate(ZK), PzZ)
                    gzK   = PzZ / (self._mu + ip)
                    temp  = numpy.dot(numpy.conjugate(ZK), self._Pz[m])
                    PzK   = (self._Pz[m] - numpy.outer(gzK, temp)) / self._mu

                    # Update active weight vector.
                    epK  = YcK - numpy.dot(self._waH[m], ZK)
                    waHK = self._waH[m] + self._gamma * numpy.conjugate(gzK) * epK
                    # Apply fixed diagonal loading
                    if self._fixed_diagonal_load > 0:
                        waHK = waHK - self._fixed_diagonal_load * numpy.dot(numpy.conjugate(PzK), self._waH[m])
                    waK = numpy.conjugate(waHK)

                    if self._constraint_option > 0:
                        waK2 = abs(numpy.dot(waHK, waK))
                        if (self._constraint_option == 1 or self._constraint_option == 3) and waK2 > self._alpha2:
                            # Still under control? Apply quadratic constraint.
                            va   = numpy.dot(PzK, waK)
                            a    = abs(numpy.dot(va, numpy.conjugate(va)))
                            b    = - 2.0 * (numpy.dot(numpy.conjugate(va), waK)).real
                            c    = waK2 - self._alpha2
                            arg  = b * b - 4.0 * a *c
                            if arg > 0:
                                betaK = - (b + numpy.sqrt(arg)) / (2.0 * a)
                            else:
                                betaK = - b / (2.0 * a)
                            waHK = waHK - betaK * numpy.conjugate(va)
                        if self._constraint_option >= 2 and waK2 > self._max_wa_l2norm:
                            # Normalize the norm of the active weight vector
                            waHK = waHK * numpy.sqrt( self._max_wa_l2norm / waK2 )
                            PzK  = numpy.identity(self._chan_num - self._Nc) / self._init_diagonal_load

                    # Dump debugging info.
                    if m in self._subband_no_printed:
                        print ''
                        print 'Sample %d: Band %d' %(self._isamp, m)
                        print 'SigmaK          = %8.4e' %(energy)
                        print 'Avg. SigmaK     = %8.4e' %(self._energy)
                        norm_gzK = abs(numpy.dot(numpy.conjugate(gzK), gzK))
                        print '||gzK||^2       = %8.4e' %(norm_gzK)
                        print '||Z^H P_z Z||^2 = %8.4e' %(abs(ip))
                        if self._constraint_option == 1:
                            print 'betaK           = %8.4e' %(betaK)
                        norm_waK = abs(numpy.dot(numpy.conjugate(waHK), waHK))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)

                    # Store values for next iteration
                    self._Pz[m]  = copy.deepcopy(PzK)
                    self._waH[m] = copy.deepcopy(waHK)

                # Calculate complete array output.
                if self._isamp >= self._min_frames:
                    output[m] = YcK - numpy.dot(self._waH[m], ZK)
                else:
                    output[m] = YcK
                if m > 0 and m < self._fftlen2:
                    output[self._fftlen - m] = numpy.conjugate(output[m])

            # Update the average power.
            self._energy = self._energy * self._beta + (1.0 - self._beta) * energy
            yield output
            self._isamp += 1

    def calc_beamformer_weights(self, samplerate, delays):
        """
        compute the quiescent weight and blocking matrix
        """
        for m in range(self._fftlen2+1):
            vs = calc_array_manifold_f(m, self._fftlen, samplerate, delays, half_band_shift=False)
            self._BmH[m] = numpy.transpose(calc_blocking_matrix(vs , self._Nc))
            self._wqH[m] = numpy.conjugate(vs)
            #print(numpy.dot(self._BmH[m], vs)) this has to be zero

    def reset_stats(self):
        """
        Reset weight vectors for a new environment.
        """

        if self._isamp > 0:
            print 'Updated weight vectors on %d of %d total frames.' %(self._ttl_updates, self._isamp)

        self._isamp       = 0 # no. frames processed already
        self._ttl_updates = 0 # no. frames used for active weight estimation
        self._energy      = self._init_diagonal_load
        self._Pz          = [numpy.identity(self._chan_num - self._Nc) / self._init_diagonal_load for m in range(self._fftlen2+1)]
        self._waH         = numpy.zeros((self._fftlen2+1, self._chan_num - self._Nc), numpy.complex)

    def reset(self):
        self._array_source.reset()
        self.reset_stats()
