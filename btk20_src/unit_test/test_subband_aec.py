#!/usr/bin/python
"""
Test subband acoustic echo cancellation on the single channel data.

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
from btk20.aec import *

def test_subband_aec(analysis_filter_path,
                     synthesis_filter_path,
                     M, m, r,
                     input_audio_path,
                     reference_audio_path,
                     out_path,
                     aec_conf,
                     samplerate=16000):

    D = M / 2**r # frame shift

    # Read analysis prototype 'h'
    with open(analysis_filter_path, 'r') as fp:
        h_fb = pickle.load(fp)

    # Read synthesis prototype 'g'
    with open(synthesis_filter_path, 'r') as fp:
        g_fb = pickle.load(fp)

    # Instantiation of an audio file reader
    input_sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
    reference_sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
    # Instantiation of over-sampled DFT analysis filter bank
    input_afb = OverSampledDFTAnalysisBankPtr(input_sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)
    reference_afb = OverSampledDFTAnalysisBankPtr(reference_sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)

    # Instantiation of subband AEC
    if aec_conf['type'].lower() == 'information_filter':
        # Information Kalman filter AEC
        aec = InformationFilterEchoCancellationFeaturePtr(reference_afb, input_afb,
                                                          sample_num = aec_conf.get('filter_length', 2),
                                                          beta = aec_conf.get('beta', 0.95),
                                                          sigmau2 = aec_conf.get('sigmau2', 10E-4),
                                                          sigmak2 = aec_conf.get('sigmak2', 5.0),
                                                          snr_threshold = aec_conf.get('snr_threshold', 0.01),
                                                          energy_threshold = aec_conf.get('energy_threshold', 100),
                                                          smooth = aec_conf.get('smooth', 0.9),
                                                          loading = aec_conf.get('loading', 1.0E-02),
                                                          amp4play = aec_conf.get('amp4play', 1.0))
    elif aec_conf['type'].lower() == 'square_root_information_filter':
        # Square root information filter
        aec = SquareRootInformationFilterEchoCancellationFeaturePtr(reference_afb, input_afb,
                                                                    sample_num = aec_conf.get('filter_length', 2),
                                                                    beta = aec_conf.get('beta', 0.95),
                                                                    sigmau2 = aec_conf.get('sigmau2', 10E-4),
                                                                    sigmak2 = aec_conf.get('sigmak2', 5.0),
                                                                    snr_threshold = aec_conf.get('snr_threshold', 0.01),
                                                                    energy_threshold = aec_conf.get('energy_threshold', 100),
                                                                    smooth = aec_conf.get('smooth', 0.9),
                                                                    loading = aec_conf.get('loading', 1.0E-02),
                                                                    amp4play = aec_conf.get('amp4play', 1.0))
    elif aec_conf['type'].lower() == 'dtd_block_kalman_filter':
        # Kalman filtering AEC with double-talk detector (DTD)
        aec = DTDBlockKalmanFilterEchoCancellationFeaturePtr(reference_afb, input_afb,
                                                             sample_num = aec_conf.get('filter_length', 2),
                                                             beta = aec_conf.get('beta', 0.95),
                                                             sigmau2 = aec_conf.get('sigmau2', 10E-4),
                                                             sigmak2 = aec_conf.get('sigmak2', 5.0),
                                                             snr_threshold = aec_conf.get('snr_threshold', 0.01),
                                                             energy_threshold = aec_conf.get('energy_threshold', 100),
                                                             smooth = aec_conf.get('smooth', 0.9),
                                                             amp4play = aec_conf.get('amp4play', 1.0))
    elif aec_conf['type'].lower() == 'nlms':
        # Normalized least mean square AEC
        aec =  NLMSAcousticEchoCancellationFeaturePtr(reference_afb, input_afb,
                                                      delta = aec_conf.get('delta', 100.0),
                                                      epsilon = aec_conf.get('epsilon', 1.0E-04),
                                                      threshold = aec_conf.get('energy_threshold', 100.0))
    else:
        raise KeyError('Invalid AEC type {}'.format(aec_conf['type']))

    # Instantiation of over-sampled DFT synthesis filter bank
    sfb = OverSampledDFTSynthesisBankPtr(aec, prototype = g_fb, M = M, m = m, r = r, delay_compensation_type=2)
    # Read the observed audio file
    input_sample_feat.read(input_audio_path, samplerate)
    # Read the reference audio file
    reference_sample_feat.read(reference_audio_path, samplerate)

    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
        except:
            pass
    wavefile = wave.open(out_path, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(samplerate))

    # Perform subband AEC through the oversampled DFT-modulated filer bank
    for frame_no, b in enumerate(sfb):
        if frame_no % 128 == 0:
            print('%0.2f sec. processed' %(frame_no * D / samplerate))
        storewave = numpy.array(b, numpy.int16)
        wavefile.writeframes(storewave.tostring())

    wavefile.close()


def build_parser():

    M = 256
    m = 4
    r = 1

    protoPath    = 'prototype.ny'
    analysis_filter_path  = '%s/h-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)
    synthesis_filter_path = '%s/g-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)

    parser = argparse.ArgumentParser(description='test subband AEC.')
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
    parser.add_argument('-i', dest='input_audio_path',
                        default='data/speech_and_reverb_lt.wav',
                        help='observation audio file')
    parser.add_argument('-o', dest='out_path',
                        default='out/aec_output.wav',
                        help='output audio file')
    parser.add_argument('-p', dest='reference_audio_path',
                        default='data/lt.wav',
                        help='reference audio file')
    parser.add_argument('-c', dest='aec_conf_path',
                        default=None,
                        help='JSON path for AEC configuration')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    if args.aec_conf_path is None:
        # Default AEC configuration
        aec_conf={'type':'dtd_block_kalman_filter', # 'information_filter' or 'square_root_information_filter'
                  'filter_length':36, # length of the subband Kalman filter
                  'loading':10e-4, # diagonal loading added to the information matrix
                  'sigmau2':10e-6, # initial variance
                  'sigmak2':5.0, # initial Kalman gain
                  'beta':0.95, # forgetting factor recursive observation noise variance estimation
                  'snr_threshold':0.01,
                  'energy_threshold':1.0E+01,
                  'smooth':0.95,
                  'amp4play':1.0,
                }
    else:
        with open(args.aec_conf_path, 'r') as jsonfp:
            aec_conf = json.load(jsonfp)

    print('AEC config.')
    print(json.dumps(aec_conf, indent=4))
    print('')
    test_subband_aec(args.analysis_filter_path,
                     args.synthesis_filter_path,
                     args.M, args.m, args.r,
                     args.input_audio_path,
                     args.reference_audio_path,
                     args.out_path,
                     aec_conf,
                     samplerate=16000)
