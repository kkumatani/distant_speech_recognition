#!/usr/bin/python
"""
Test a single-channel or multi-channel weighted prediction error (WPE) dereverberator on the subband domain.

It will switch to the multi-channel WPE If the multiple input audio files are specified.

.. reference::
[1] Kumatani, J. W. McDonough, S. Schachl, D. Klakow, P. N. Garner and W. Li, "Filter bank design based on minimization of individual aliasing terms for minimum mutual information subband adaptive beamforming," in ICASSP, Las Vegas, USA, 2008.

[2] T. Yoshioka and T. Nakatani, "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening," IEEE Trans. Audio, Speech, Language Process, pp. 2707-2720, 2012.

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
from btk20.dereverberation import *

def single_channel_wpe(h_fb, g_fb, D, M, m, r, input_audio_path, out_path, wpe_conf, samplerate, start_frame_no, end_frame_no):
    """
    Run weighted prediction error (WPE) dereverberation on single channel data

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
    :param input_audio_path: Input audio file that you would like to dereverb
    :type input_audio_path: string
    :param out_path: Output audio file
    :type out_path: string
    :param wpe_conf: Dictionary to specify WPE parameters
    :type wpe_conf: Python dictionary
    :param samplerate: Sampling rate of the input audio
    :type samplerate: integer
    :param start_frame_no: Start point used for filter estimation
    :type start_frame_no: integer
    :param end_frame_no: End point used for filter estimation
    :type end_frame_no: integer
    """

    # Instantiation of an audio file reader
    sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
    # Instantiation of over-sampled DFT analysis filter bank
    afb = OverSampledDFTAnalysisBankPtr(sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)
    # Instantiation of single channel WPE dereverberator
    dereverb = SingleChannelWPEDereverberationFeaturePtr(afb,
                                                         lower_num = wpe_conf.get('lower_num', 0),
                                                         upper_num = wpe_conf.get('upper_num', 64),
                                                         iterations_num = wpe_conf.get('iterations_num', 2),
                                                         load_db = wpe_conf.get('load_db', -20.0),
                                                         band_width = wpe_conf.get('band_width', 0.0),
                                                         samplerate = samplerate)
    # Instantiation of synthesis filter bank
    sfb = OverSampledDFTSynthesisBankPtr(dereverb, prototype=g_fb, M=M, m=m, r=r, delay_compensation_type=2)

    # Estimate the dereverberation filter
    sample_feat.read(input_audio_path, samplerate)
    dereverb.print_objective_func(50)
    frame_num = dereverb.estimate_filter()
    print('%d frames are used for filter estimation' %frame_num)

    # Opening the output audio file
    wavefile = wave.open(out_path, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(samplerate))

    # Run WPE dereverberation
    sample_feat.read(input_audio_path, samplerate)
    for frame_no, b in enumerate(sfb):
        if frame_no % 128 == 0:
            print('%0.2f sec. processed' %(frame_no * D / samplerate))
        storewave = numpy.array(b, numpy.int16)
        wavefile.writeframes(storewave.tostring())

    wavefile.close()


def multi_channel_wpe(h_fb, g_fb, D, M, m, r, input_audio_paths, out_paths, wpe_conf, samplerate, start_frame_no, end_frame_no):
    """
    Run weighted prediction error (WPE) on multi-channel data

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
    :param wpe_conf: Dictionary to specify WPE parameters
    :type wpe_conf: Python dictionary
    :param samplerate: Sampling rate of the input audio
    :type samplerate: integer
    :param start_frame_no: Start point used for filter estimation
    :type start_frame_no: integer
    :param end_frame_no: End point used for filter estimation
    :type end_frame_no: integer
    """

    channels_num = len(input_audio_paths)
    # Instantiation of multi-channel dereverberation filter estimation based on WPE
    pre_dereverb = MultiChannelWPEDereverberationPtr(subbands_num=M, channels_num=channels_num,
                                                     lower_num = wpe_conf.get('lower_num', 0),
                                                     upper_num = wpe_conf.get('upper_num', 32),
                                                     iterations_num = wpe_conf.get('iterations_num', 2),
                                                     load_db = wpe_conf.get('load_db', -20.0),
                                                     band_width = wpe_conf.get('band_width', 0.0),
                                                     diagonal_bias = wpe_conf.get('diagonal_bias', 0.001),
                                                     samplerate = samplerate)
    pre_dereverb.print_objective_func(50)

    sample_feats = []
    afbs = []
    for c, input_audio_path in enumerate(input_audio_paths):
        # Instantiation of an audio file reader
        sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
        sample_feat.read(input_audio_path, samplerate)
        # Instantiation of over-sampled DFT analysis filter bank
        afb = OverSampledDFTAnalysisBankPtr(sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)
        pre_dereverb.set_input(afb)
        # Keep the instances
        sample_feats.append(sample_feat)
        afbs.append(afb)

    # build the dereverberation filter
    frame_num = pre_dereverb.estimate_filter()
    print('%d frames are used for filter estimation' %frame_num)

    sfbs = []
    wavefiles = []
    for c in range(channels_num):
        # Reread the test audio
        sample_feats[c].read(input_audio_paths[c], samplerate)
        # Instantiate the multi-channel WPE feature object
        dereverb = MultiChannelWPEDereverberationFeaturePtr(pre_dereverb, channel_no=c)
        sfb = OverSampledDFTSynthesisBankPtr(dereverb, prototype = g_fb, M = M, m = m, r = r, delay_compensation_type = 2)
        sfbs.append(sfb)
        # Open an output file pointer
        wavefile = wave.open(out_paths[c], 'w')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2) #
        wavefile.setframerate(int(samplerate))
        wavefiles.append(wavefile)

    # Perform dereverberation on each channel data
    frame_no = 0
    while True:
        if frame_no % 128 == 0:
            print('%0.2f sec. processed' %(frame_no * D / samplerate))
        try:
            for c in range(channels_num):
                wavefiles[c].writeframes(numpy.array(sfbs[c].next(), numpy.int16).tostring())
        except StopIteration:
            break
        frame_no += 1

    # Close all the output file pointers
    for wavefile in wavefiles:
        wavefile.close()


def test_subband_dereverberator(analysis_filter_path,
                                synthesis_filter_path,
                                M, m, r,
                                input_audio_paths,
                                out_paths,
                                wpe_conf,
                                samplerate=16000,
                                start_frame_no = 0,
                                end_frame_no =  -1):

    assert len(input_audio_paths) == len(out_paths), 'No. input files have to be equal to no. output files'
    D = M / 2**r # frame shift

    # Read analysis prototype 'h'
    with open(analysis_filter_path, 'r') as fp:
        h_fb = pickle.load(fp)

    # Read synthesis prototype 'g'
    with open(synthesis_filter_path, 'r') as fp:
        g_fb = pickle.load(fp)

    for out_path in out_paths:
        if not os.path.exists(os.path.dirname(out_path)):
            try:
                os.makedirs(os.path.dirname(out_path))
            except:
                pass

    if len(input_audio_paths) == 1:
        single_channel_wpe(h_fb, g_fb, D, M, m, r, input_audio_paths[0], out_paths[0], wpe_conf, samplerate, start_frame_no, end_frame_no)
    else:
        multi_channel_wpe(h_fb, g_fb, D, M, m, r, input_audio_paths, out_paths, wpe_conf, samplerate, start_frame_no, end_frame_no)


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
    default_out_paths =         ['out/U1001_1M_c1.wav',
                                 'out/U1001_1M_c2.wav',
                                 'out/U1001_1M_c3.wav',
                                 'out/U1001_1M_c4.wav']

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
    parser.add_argument('-o', dest='out_paths', nargs='+',
                        default=default_out_paths,
                        help='output audio file(s)')
    parser.add_argument('-c', dest='wpe_conf_path',
                        default=None,
                        help='JSON path for WPE dereverberator configuration')
    parser.add_argument('-b', dest='start_frame_no',
                        default=26000,
                        help='Start frame point for filter estimation')
    parser.add_argument('-e', dest='end_frame_no',
                        default=-1, # 62000
                        help='end frame point for filter estimation. Will be the end of the file if it is -1')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    if args.wpe_conf_path is None:
        # Default WPE configuration
        wpe_conf={'lower_num':0,
                  'upper_num':32, # upper_num - lower_num == filter length,
                  'iterations_num':2,
                  'load_db': -18.0,
                  'band_width':0.0,
                  'diagonal_bias':0.0001, # Diagonal loading for Cholesky decomposition stabilization (Multi-channel WPE only)
        }
    else:
        with open(args.wpe_conf_path, 'r') as jsonfp:
            wpe_conf = json.load(jsonfp)

    print('WPE config.')
    print(json.dumps(wpe_conf, indent=4))
    print('')
    test_subband_dereverberator(args.analysis_filter_path,
                                args.synthesis_filter_path,
                                args.M, args.m, args.r,
                                args.input_audio_paths,
                                args.out_paths,
                                wpe_conf,
                                samplerate=16000,
                                start_frame_no=args.start_frame_no,
                                end_frame_no=args.end_frame_no)
