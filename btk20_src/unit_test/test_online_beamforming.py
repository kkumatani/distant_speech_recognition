#!/usr/bin/python
"""
Test online subband beamforming algorithms that can update weights online (as opposed to batch processing)

.. reference::

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""
import argparse, json
import os.path
import pickle
import wave
import sys
import numpy

from btk20.common import *
from btk20.stream import *
from btk20.feature import *
from btk20.modulated import *
from btk20.beamformer import *
from btk20.pybeamformer import *
from btk20.postfilter import *

SSPEED = 343740.0

def check_position_data_format(ap_conf):
    """
    Check the following items in position information:
    - if the dimension of the postion vector is sufficient, and
    - if time stamp matches between target and noise sources,
    """

    def check_position_dim(t, pos):
        if ap_conf['array_type'] == 'linear':
            assert len(pos) >= 1, 'Insufficient position info. at time %0.3f' %t
        elif ap_conf['array_type'] == 'planar' or ap_conf['array_type'] == 'circular':
            assert len(pos) >= 2, 'Insufficient position info. at time %0.3f' %t
        else:
            assert len(pos) >= 3, 'Insufficient position info. at time %0.3f' %t

    assert 'positions' in ap_conf['target'], 'No target position'
    for posx, (targ_t, pos) in enumerate(ap_conf['target']['positions']):
        check_position_dim(targ_t, pos)
        if 'noises' in ap_conf:
            for noisex in range(len(ap_conf['noises'])):
                noise_t = ap_conf['noises'][noisex]['positions'][posx][0]
                assert targ_t == noise_t, "%d-th noise: Misaligned time stamp %0.4f != %0.4f" %(noisex, targ_t, noise_t)
                check_position_dim(noise_t, ap_conf['noises'][noisex]['positions'][posx][1])


def online_beamforming(h_fb, g_fb, D, M, m, r, input_audio_paths, out_path, ap_conf, samplerate):
    """
    Run a class of online beamforming algorithms

    :param h_fb: Analysis filter coefficeint; This must be generated with /tools/filterbank/{design_nyquist_filter.py,design_de_haan_filter.py}
    :type h_fb: numpy float vector
    :param g_fb: Synthesis filter coefficients paird with h_fb
    :type g_fbM: numpy float vector
    :param M: Number of subbands
    :type M: integer
    :param m: Filter length factor
    :type m: integer
    :param D: Decimation factor
    :type D: integer
    :param input_audio_path: List of input audio file that you would like to dereverb
    :type input_audio_path: List of string
    :param out_path: Output audio file
    :type out_path: string
    :param ap_conf: Dictionary to specify beamforming parameters
    :type ap_conf: Python dictionary
    :param samplerate: Sampling rate of the input audio
    :type samplerate: integer
    :returns: Tuple of total beamformer's output power and the number of processed frames
    """

    channels_num = len(input_audio_paths)

    sample_feats = []
    afbs = []
    for c, input_audio_path in enumerate(input_audio_paths):
        # Instantiation of an audio file reader
        sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
        sample_feat.read(input_audio_path, samplerate)
        # Instantiation of over-sampled DFT analysis filter bank
        afb = OverSampledDFTAnalysisBankPtr(sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)
        # Keep the instances
        sample_feats.append(sample_feat)
        afbs.append(afb)

    # Setting a beamformer
    bf_conf = ap_conf['beamformer']
    if bf_conf['type'] == 'delay_and_sum' :
        beamformer = SubbandGSCBeamformer(afbs, Nc = 1)
    elif bf_conf['type'] == 'lcmv':
        beamformer = SubbandGSCBeamformer(afbs, Nc = 2)
    elif bf_conf['type'] == 'super_directive':
        beamformer = SubbandMVDRBeamformer(afbs)
    elif bf_conf['type'] == 'gsclms':
        beamformer = SubbandGSCLMSBeamformer(afbs,
                                             beta  = bf_conf.get('beta', 0.97),  # forgetting factor for recursive signal power est.
                                             gamma = bf_conf.get('gamma', 0.01), # step size factor
                                             init_diagonal_load   = bf_conf.get('init_diagonal_load', 1.0E+6), # represent each subband energy
                                             regularization_param = bf_conf.get('regularization_param', 1.0E-4),
                                             energy_floor         = bf_conf.get('energy_floor', 90),     # flooring small energy
                                             sil_thresh           = bf_conf.get('sil_thresh', 1.0E+8),   # silence threshold
                                             max_wa_l2norm        = bf_conf.get('max_wa_l2norm', 100.0), # Threshold so |wa|^2 <= max_wa_l2nor
                                             min_frames           = bf_conf.get('min_frames', 128),
                                             slowdown_after       = bf_conf.get('slowdown_after', 4096))
    elif bf_conf['type'] == 'gscrls':
        beamformer = SubbandGSCRLSBeamformer(afbs,
                                             beta  = bf_conf.get('beta', 0.97), # forgetting factor for recursive signal power est.
                                             gamma = bf_conf.get('gamma', 0.04), # step size factor
                                             mu    = bf_conf.get('mu', 0.97),   # recursive weight for covariance matrix est.
                                             init_diagonal_load   = bf_conf.get('init_diagonal_load', 1.0E+6),
                                             regularization_param = bf_conf.get('regularization_param', 1.0E-2),
                                             sil_thresh           = bf_conf.get('sil_thresh', 1.0E+8),
                                             constraint_option    = bf_conf.get('constraint_option', 3), # Constrait method for active weight vector est.
                                             alpha2               = bf_conf.get('alpha2', 10.0),         # 1st threshold so |wa|^2 <= alpha2
                                             max_wa_l2norm        = bf_conf.get('max_wa_l2norm', 100.0), # 2nd threshold so |wa|^2 <= max_wa_l2norm
                                             min_frames           = bf_conf.get('min_frames', 128),
                                             slowdown_after       = bf_conf.get('slowdown_after', 4096))

    else:
        raise KeyError('Invalid beamformer type: {}'.format(bf_conf['type']))

    # Setting a post-filter
    use_postfilter = False
    pybeamformer = PyVectorComplexFeatureStreamPtr(beamformer) # convert a pure python class into BTK stream object
    if not ('postfilter' in ap_conf):
        spatial_filter = pybeamformer
    elif bf_conf['type'] == 'delay_and_sum' or bf_conf['type'] == 'lcmv' or  bf_conf['type'] == 'super_directive':
        pf_conf = ap_conf['postfilter']
        if pf_conf['type'] == 'zelinski':
            spatial_filter = ZelinskiPostFilterPtr(pybeamformer, M,
                                                   pf_conf.get('alpha', 0.6),
                                                   pf_conf.get('subtype', 2))
        elif pf_conf['type'] == 'mccowan':
            spatial_filter = McCowanPostFilterPtr(pybeamformer, M,
                                               pf_conf.get('alpha', 0.6),
                                               pf_conf.get('subtype', 2))
            spatial_filter.set_diffuse_noise_model(ap_conf['microphone_positions'], samplerate, SSPEED)
            spatial_filter.set_all_diagonal_loading(bf_conf.get('diagonal_load', 0.01))
        elif pf_conf['type'] == 'lefkimmiatis':
            spatial_filter = LefkimmiatisPostFilterPtr(pybeamformer, M,
                                                       pf_conf.get('min_sv', 1e-8),
                                                       pf_conf.get('fbin_no1', 128),
                                                       pf_conf.get('alpha', 0.8),
                                                       pf_conf.get('subtype', 2))
            spatial_filter.set_diffuse_noise_model(ap_conf['microphone_positions'], samplerate, SSPEED)
            spatial_filter.set_all_diagonal_loading(bf_conf.get('diagonal_load', 0.1))
            spatial_filter.calc_inverse_noise_spatial_spectral_matrix()
        else:
            raise KeyError('Invalid post-filter type: {}'.format(pf_conf['type']))
        use_postfilter = True
    else:
        raise NotImplementedError("Post-filter unsupported: {}".format(bf_conf['type']))

    # Setting the beamformer / post-filter instance to the synthesis filter bank
    sfb = OverSampledDFTSynthesisBankPtr(spatial_filter, prototype = g_fb, M = M, m = m, r = r, delay_compensation_type = 2)
    # Open an output file pointer
    wavefile = wave.open(out_path, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(samplerate))

    def wrapper_weights_calculator(delays_t, delays_js=None):
        """
        wrapping the functions for beamformer weight computation
        """
        if delays_js is not None and bf_conf['type'] != 'lcmv':
            print('Noise information will be ignored')

        if bf_conf['type'] == 'delay_and_sum':
            beamformer.calc_beamformer_weights(samplerate, delays_t)
        elif bf_conf['type'] == 'super_directive':
            beamformer.calc_sd_beamformer_weights(samplerate, delays_t,
                                                  ap_conf['microphone_positions'], sspeed = SSPEED,
                                                  mu = bf_conf.get('diagonal_load', 0.01))
        elif bf_conf['type'] == 'gsclms' or bf_conf['type'] == 'gscrls':
            beamformer.calc_beamformer_weights(samplerate, delays_t)
        elif bf_conf['type'] == 'lcmv': # put multiple linear constraints (distortionless and null)
            assert delays_js is not None, 'LCMV beamforming: missing noise source positions'
            beamformer.calc_beamformer_weights_n(samplerate, delays_t, delays_js)

    # Perform beamforming
    total_energy = 0
    elapsed_time = 0.0
    time_delta   = D / float(samplerate)
    # Set the initial look direction(s)
    posx = 0
    # Direction of the target souce
    target_position_t = ap_conf['target']['positions'][posx][1]
    delays_t = calc_delays(ap_conf['array_type'], ap_conf['microphone_positions'], target_position_t, sspeed = SSPEED)
    delays_js = None
    if 'noises' in ap_conf: # Direction of jammer(s)
        delays_js = numpy.zeros((len(ap_conf['noises']), len(delays_t)), numpy.float) # (no. noises) x (no. mics) matrix
        for noisex in range(len(delays_js)):
            noise_position_t = ap_conf['noises'][noisex]['positions'][posx][1]
            delays_js[noisex] = calc_delays(ap_conf['array_type'], ap_conf['microphone_positions'], noise_position_t, sspeed = SSPEED)

    # Compute the initial beamformer weight
    wrapper_weights_calculator(delays_t, delays_js)
    if use_postfilter == True:
        spatial_filter.set_beamformer(beamformer.beamformer())
    for frame_no, buf in enumerate(sfb):
        if frame_no % 128 == 0:
            print('%0.2f sec. processed' %(frame_no * time_delta))
        total_energy += numpy.inner(buf, buf)
        wavefile.writeframes(numpy.array(buf, numpy.int16).tostring())
        elapsed_time += time_delta
        if elapsed_time > ap_conf['target']['positions'][posx][0] and (posx + 1) < len(ap_conf['target']['positions']):
            # Update the look direction(s)
            posx += 1
            target_position_t = ap_conf['target']['positions'][posx][1]
            delays_t = calc_delays(ap_conf['array_type'], ap_conf['microphone_positions'], target_position_t, sspeed = SSPEED)
            if 'noises' in ap_conf: # Direction of jammer(s)
                delays_js = numpy.zeros((len(ap_conf['noises']), len(delays_t)), numpy.float) # (no. noises) x (no. mics) matrix
                for noisex in range(len(delays_js)):
                    noise_position_t = ap_conf['noises'][noisex]['positions'][posx][1]
                    delays_js[noisex] = calc_delays(ap_conf['array_type'], ap_conf['microphone_positions'], noise_position_t, sspeed = SSPEED)

            # Recompute the beamformer weight
            wrapper_weights_calculator(delays_t, delays_js)

    # Close all the output file pointers
    wavefile.close()

    return (total_energy, frame_no)


def test_online_beamforming(analysis_filter_path,
                            synthesis_filter_path,
                            M, m, r,
                            input_audio_paths,
                            out_path,
                            ap_conf,
                            samplerate=16000):

    D = M / 2**r # frame shift

    # Read analysis prototype 'h'
    with open(analysis_filter_path, 'r') as fp:
        h_fb = pickle.load(fp)

    # Read synthesis prototype 'g'
    with open(synthesis_filter_path, 'r') as fp:
        g_fb = pickle.load(fp)

    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
        except:
            pass

    return online_beamforming(h_fb, g_fb, D, M, m, r, input_audio_paths, out_path, ap_conf, samplerate)


def build_parser():

    M = 256
    m = 4
    r = 1

    protoPath    = 'prototype.ny'
    analysis_filter_path  = '%s/h-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)
    synthesis_filter_path = '%s/g-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)

    default_input_audio_paths = ['data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c1.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c2.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c3.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c4.wav']
    default_out_path =  'out/U1001_1M_beamformed.wav'

    parser = argparse.ArgumentParser(description='test subband WPE dereverberation.')
    parser.add_argument('-a', dest='analysis_filter_path',
                        default=analysis_filter_path,
                        help='analysis filter prototype file')
    parser.add_argument('-s', dest='synthesis_filter_path',
                        default=synthesis_filter_path,
                        help='synthesis filter prototype file')
    parser.add_argument('-M', dest='M',
                        default=M, type=int,
                        help='no. of subbands')
    parser.add_argument('-m', dest='m',
                        default=m, type=int,
                        help='Prototype filter length factor')
    parser.add_argument('-r', dest='r',
                        default=r, type=int,
                        help='Decimation factor')
    parser.add_argument('-i', dest='input_audio_paths', nargs='+',
                        default=default_input_audio_paths,
                        help='observation audio file(s)')
    parser.add_argument('-o', dest='out_path',
                        default=default_out_path,
                        help='output audio file')
    parser.add_argument('-c', dest='ap_conf_path',
                        default=None,
                        help='JSON path for array processing configuration')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    if args.ap_conf_path is None:
        # Default arrayp processing configuration
        ap_conf={'array_type':'linear', # 'linear', 'planar', 'circular' or 'nearfield'
                 'microphone_positions':[[-113.0, 0.0, 2.0],
                                         [  36.0, 0.0, 2.0],
                                         [  76.0, 0.0, 2.0],
                                         [ 113.0, 0.0, 2.0]],
                 'target':{'positions':[[0.0, [-1.306379, None, None]]]}, # [time, [position vector]], if 'linear', the position value is the direction of arrival in radian.
                 'beamformer':{'type':'super_directive'}, # 'delay_and_sum' 'gsclms' or 'gscrls'
                 'postfilter':{'type':'zelinski',
                               'subtype':2,
                               'alpha':0.7}
        }
    else:
        with open(args.ap_conf_path, 'r') as jsonfp:
            ap_conf = json.load(jsonfp)

    print('BF config.')
    check_position_data_format(ap_conf)
    print(json.dumps(ap_conf, indent=4))
    print('')
    (total_energy, frame_no) = test_online_beamforming(args.analysis_filter_path,
                                                       args.synthesis_filter_path,
                                                       args.M, args.m, args.r,
                                                       args.input_audio_paths,
                                                       args.out_path,
                                                       ap_conf,
                                                       samplerate=16000)
    print('Avg. output power: %f' %(total_energy / frame_no))
    print('No. frames processed: %d' %frame_no)
