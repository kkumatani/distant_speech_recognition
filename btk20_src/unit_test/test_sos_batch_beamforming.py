#!/usr/bin/python
"""
Test batch-processing subband beamformers with SOS, MVDR and GEV beamforming.

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

def check_label_data_format(ap_conf):
    """
    Check the JSON file format for the following items:
    - Position information,
    - Voice activity detection label, and
    - Time-frequency mask file
    """

    if ap_conf['beamformer'] == 'smimvdr':
        from test_online_beamforming import check_position_data_format
        check_position_data_format(ap_conf)
    else:
        if 'vad_label' in ap_conf['target']:
            for i, (b, e) in enumerate(ap_conf['target']['vad_label']):
                assert b < e, '%d-th segment: start time %f >= end time %f ?' %(i, b,e)
        elif 'tfmask_path' in ap_conf['target']:
            assert os.path.exists(ap_conf['target']['tfmask_path']), 'Could not find %s' %(ap_conf['target']['tfmask_path'])
            is_noise_tfmask = False
            for noise_conf in ap_conf['noises']:
                if 'tfmask_path' in noise_conf:
                    assert os.path.exists(ap_conf['noises'][0]['tfmask_path']), 'Could not find %s' %(ap_conf['noises']['tfmask_path'])
                    is_noise_tfmask = True
            assert is_noise_tfmask == True, 'Missing noise TF mask'
        else:
            raise KeyError('No segmentation information given. Specify \"vad_label\" or \"tfmask_path\"')


def load_tfmasks(ap_conf):
    """
    Load TF mask files for the target and noise sources.
    The TF mask file should contain a sequence of numpy vectors whose element
    indicates activity of each sound source at each frequnecy bin.
    It is saved as the Python pickle format.

    :param ap_conf: contains the TF mask file paths
    :type : Python dictionary
    :return: Numpy matrices for the target and noise sources,
             where no. rows and no. columns correspond to no. frames and no. bands, respectively.
    """
    def load_tfmask(tfmask_path):
        tfmask = []
        with open(tfmask_path, 'rb') as rfp:
            while True:
                try:
                    tfmask.append(pickle.load(rfp))
                except EOFError:
                    break

        return numpy.array(tfmask)

    print('Loading TF mask for the target source: %s' %ap_conf['target']['tfmask_path'])
    mask_t = load_tfmask(ap_conf['target']['tfmask_path'])
    print('Loaded: %d x %d TF mask' %(mask_t.shape[0], mask_t.shape[1]))
    mask_j = None
    for noise_conf in ap_conf['noises']:
        if 'tfmask_path' in noise_conf:
            print('Loading TF mask for noise source: %s' %noise_conf['tfmask_path'])
            mask_nj = load_tfmask(noise_conf['tfmask_path'])
            if mask_j is None:
                mask_j = mask_nj
            else:
                mask_j += mask_nj
            print('Loaded: %d x %d TF mask' %(mask_nj.shape[0], mask_nj.shape[1]))

    mask_j /= len(ap_conf['noises'])

    return (mask_t, mask_j)


def sos_batch_beamforming(h_fb, g_fb, D, M, m, r, input_audio_paths, out_path, ap_conf, samplerate):
    """
    Run a batch-processing beamformer with a SOS criterion

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
    if bf_conf['type'] == 'smimvdr' :
        beamformer = SubbandSMIMVDRBeamformer(afbs, Nc = 1)
    elif bf_conf['type'] == 'bmvdr':
        beamformer = SubbandBlindMVDRBeamformer(afbs)
    elif bf_conf['type'] == 'gev':
        beamformer = SubbandGEVBeamformer(afbs)
    else:
        raise KeyError('Invalid batch-processing beamformer type: {}'.format(bf_conf['type']))

    # Setting a post-filter
    use_postfilter = False
    pybeamformer = PyVectorComplexFeatureStreamPtr(beamformer) # convert a pure python class into BTK stream object
    if not ('postfilter' in ap_conf):
        spatial_filter = pybeamformer
    else:
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

    # Setting the beamformer / post-filter instance to the synthesis filter bank
    sfb = OverSampledDFTSynthesisBankPtr(spatial_filter, prototype = g_fb, M = M, m = m, r = r, delay_compensation_type = 2)
    # Open an output file pointer
    wavefile = wave.open(out_path, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(samplerate))

    def wrapper_weights_calculator():
        """
        wrapping the functions for beamformer weight computation
        """
        energy_threshold = bf_conf.get('energy_threshold', 10)
        if bf_conf['type'] == 'smimvdr': # MVDR beamforming with sample matrix inversion
            # Direction of the target souce
            posx = 0
            target_position_t = ap_conf['target']['positions'][posx][1]
            print ap_conf['microphone_positions']
            print target_position_t
            delays_t = calc_delays(ap_conf['array_type'], ap_conf['microphone_positions'], target_position_t, sspeed = SSPEED)
            # Compute a (spatial) covariance matrix
            beamformer.accu_stats_from_label(samplerate, target_labs = ap_conf['target']['vad_label'], energy_threshold = energy_threshold)
            beamformer.finalize_stats()
            beamformer.calc_beamformer_weights(samplerate, delays_t, mu = bf_conf.get('mu', 1e-4))
        elif bf_conf['type'] == 'bmvdr': # MVDR beamforming without the look direction a.k.a MMSE beamforming
            if 'tfmask_path' in ap_conf['target']: # Use a time-frequency mask for spatial spectral matrix estimation
                (mask_t, mask_j) = load_tfmasks(ap_conf)
                beamformer.accu_stats_from_tfmask(samplerate, mask_t, mask_j, energy_threshold = energy_threshold)
            else: # Use a VAD label for spatial spectral matrix estimation
                beamformer.accu_stats_from_label(samplerate, target_labs = ap_conf['target']['vad_label'], energy_threshold = energy_threshold)
            beamformer.finalize_stats(gamma = bf_conf.get('gamma', 1e-6))
            beamformer.calc_beamformer_weights(ref_micx = bf_conf.get('ref_micx', 0), offset = bf_conf.get('offset', 0.0))
        elif bf_conf['type'] == 'gev': # Generalized eigenvector beamforming
            if 'tfmask_path' in ap_conf['target']: # Use a TF mask for spatial spectral matrix estimation
                (mask_t, mask_j) = load_tfmasks(ap_conf)
                beamformer.accu_stats_from_tfmask(samplerate, mask_t, mask_j, energy_threshold = energy_threshold)
            else: # Use a VAD label for spatial spectral matrix estimation
                beamformer.accu_stats_from_label(samplerate, target_labs = ap_conf['target']['vad_label'], energy_threshold = energy_threshold)
            beamformer.finalize_stats(gamma = bf_conf.get('gamma', 1e-6))
            beamformer.calc_beamformer_weights()

    # Perform beamforming
    total_energy = 0
    elapsed_time = 0.0
    time_delta   = D / float(samplerate)
    # Compute the beamformer weight with a batch of data (one utterance)
    wrapper_weights_calculator()
    if use_postfilter == True:
        spatial_filter.set_beamformer(beamformer.beamformer())
    # Reloading the test data (reset the feature pointer)
    for c, input_audio_path in enumerate(input_audio_paths):
        sample_feats[c].read(input_audio_path, samplerate)
    for frame_no, buf in enumerate(sfb):
        if frame_no % 128 == 0:
            print('%0.2f sec. processed' %(frame_no * time_delta))
        total_energy += numpy.inner(buf, buf)
        wavefile.writeframes(numpy.array(buf, numpy.int16).tostring())
        elapsed_time += time_delta

    # Close all the output file pointers
    wavefile.close()

    return (total_energy, frame_no)


def test_batch_beamforming(analysis_filter_path,
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

    return sos_batch_beamforming(h_fb, g_fb, D, M, m, r, input_audio_paths, out_path, ap_conf, samplerate)


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
                 'target':{'positions':[[0.0, [-1.306379, None, None]]], # [time, [position vector]], if 'linear', the position value is the direction of arrival in radian.
                           'vad_label':[[1.5, 4.0]] # Sequential segements of voice activity, tuples of the start and end time
                 },
                 'beamformer':{'type':'smimvdr', # 'bmvdr' or 'gev'
                               'mu':1e-4,
                               'energy_threshold':10,
                 },
                 'postfilter':{'type':'zelinski',
                               'subtype':2,
                               'alpha':0.7}
        }
    else:
        with open(args.ap_conf_path, 'r') as jsonfp:
            ap_conf = json.load(jsonfp)

    print('BF config.')
    check_label_data_format(ap_conf)
    print(json.dumps(ap_conf, indent=4))
    print('')
    (total_energy, frame_no) = test_batch_beamforming(args.analysis_filter_path,
                                                      args.synthesis_filter_path,
                                                      args.M, args.m, args.r,
                                                      args.input_audio_paths,
                                                      args.out_path,
                                                      ap_conf,
                                                      samplerate=16000)
    print('Avg. output power: %f' %(total_energy / frame_no))
    print('No. frames processed: %d' %frame_no)
