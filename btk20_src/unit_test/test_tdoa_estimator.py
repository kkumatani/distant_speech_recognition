#!/usr/bin/python
"""
Perform time delay of arrival (TODA) estimation

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""

import argparse, json
import sys, os

import numpy
from btk20.feature import *
from btk20.pytdoa import *

SSPEED = 343740.0

def check_tdoae_conf_format(ap_conf):
    """
    Check the JSON data structure for TDOA estimation
    """

    tdoae_conf = ap_conf['tdoae']
    assert 'pair_ids' in tdoae_conf, 'Missing \"pair_ids\" key in JSON'

    if 'D' in tdoae_conf and 'fftlen' in tdoae_conf:
        assert tdoae_conf['D'] <= tdoae_conf['fftlen'], 'Invalid FFT length: %d < %d' %(tdoae_conf['fftlen'], tdoae_conf['D'])


def tdoa_estimation(input_audio_paths, ap_conf, samplerate, tdoa_path, trj_pos_path, ave_pos_path):
    """
    Test a TDOA estimator

    :param input_audio_paths: multiple audio files
    :param ap_conf: JSON containing array and TDOA estimator parameters
    :param samplerate: audio sampling rate
    :param tdoa_path: output file that will contain the TDOA
    :param trj_pos_path: output file that will contain a position estimate at each frame
    :param ave_pos_path: output file that will contain a position averaged over frames
    """
    tdoae_conf = ap_conf['tdoae']
    D      = tdoae_conf.get('shiftlen', 8192)
    fftlen = tdoae_conf.get('fftlen', D * 2)
    # Build the processing chain for a sound source tracker
    sample_feats = []
    spectra      = []
    for c, input_audio_path in enumerate(input_audio_paths):
        print('%d-th channel: %s' %(c, input_audio_path))
        sample_feat = SampleFeaturePtr(block_len = D, shift_len = D, pad_zeros = True)
        sample_feat.read(input_audio_path, samplerate)
        sample_feats.append(sample_feat)
        hamming_feat = HammingFeaturePtr(sample_feat)
        fft          = FFTFeaturePtr(hamming_feat, fftlen)
        spectra.append(fft)

    # Instantiate a time delay estimator object
    frontend = make_tdoa_front_end(array_type = ap_conf['array_type'],
                                    pair_ids =  tdoae_conf['pair_ids'],
                                    spec_sources = spectra,
                                    fftlen = fftlen,
                                    samplerate = samplerate,
                                    mpos = numpy.array(ap_conf['microphone_positions']),
                                    energy_threshold = tdoae_conf.get('energy_threshold', 64),
                                    minimum_pairs = tdoae_conf.get('minimum_pairs', 2),
                                    threshold     = tdoae_conf.get('cc_threshold ', 0.1),
                                    sspeed = SSPEED)

    time_delta = float(D) / samplerate
    # Estimate the TDOA and sound source location
    with open(tdoa_path, 'w') as tdoa_fp, open(trj_pos_path, 'w') as trj_pos_fp:
        def dump_position(t, Xk):
            """
            Create a list object for output JSON
            """
            pos = [None for i in range(3)]
            for i, val in enumerate(Xk):
                pos[i] = val

            return [t, pos]

        tdoa_fp.write('[\n')
        trj_pos_fp.write('{\"positions\":[\n')

        num_tdoa = 0
        sum_pos = None
        num_pos = 0
        for frame_no, obs in enumerate(frontend):
            Xk = frontend.instantaneous_position(frame_no)
            if Xk[0] > -1e10: # Coherent sound source detected
                elapsed_time = frame_no * time_delta
                if num_tdoa > 0:
                    tdoa_fp.write(',\n')
                tdoa_fp.write(json.dumps([elapsed_time, frontend.mic_pair_tdoa()]))
                num_tdoa += 1
                if sum_pos is None:
                    sum_pos = Xk
                else:
                    sum_pos += Xk
                    trj_pos_fp.write(',\n')

                print('%0.3f: %s' %(elapsed_time, numpy.array_str(Xk)))
                trj_pos_fp.write(json.dumps(dump_position(elapsed_time, Xk)))
                num_pos +=1

        tdoa_fp.write('\n]')
        trj_pos_fp.write('\n]}')

        # write the position estimates averaged over a stream
        if num_pos > 0:
            with open(ave_pos_path, 'w') as ave_pos_fp:
                sum_pos /= float(num_pos)
                ave_pos_fp.write(json.dumps({'positions':[dump_position(0.0, sum_pos)]}))


def test_tdoa_estimation(input_audio_paths, out_prefix, ap_conf, samplerate):
    """
    Calling a TDOA estimation function
    """
    tdoa_path    = out_prefix + '.tdoa.json'
    trj_pos_path = out_prefix + '.trj.pos.json'
    ave_pos_path = out_prefix + '.ave.pos.json'
    if not os.path.exists(os.path.dirname(ave_pos_path)):
        try:
            os.makedirs(os.path.dirname(ave_pos_path))
        except:
            pass

    tdoa_estimation(input_audio_paths, ap_conf, samplerate, tdoa_path, trj_pos_path, ave_pos_path)


def build_parser():

    default_input_audio_paths = ['data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c1.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c2.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c3.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c4.wav']
    default_out_prefix =  'out/U1001_1M_sl'

    parser = argparse.ArgumentParser(description='estimate a TDOA')
    parser.add_argument('-i', dest='input_audio_paths', nargs='+',
                        default=default_input_audio_paths,
                        help='observation audio file(s)')
    parser.add_argument('-o', dest='out_prefix',
                        default=default_out_prefix,
                        help='output file prefix for TDOA speaker localization results')
    parser.add_argument('-c', dest='ap_conf_path',
                        default=None,
                        help='JSON path for array processing configuration')
    parser.add_argument('-r', dest='samplerate',
                        default=16000, type=int,
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
                 'tdoae':{'type':'gcc_phat',
                          'shiftlen':8192,
                          'fftlen':8192*2,
                          'energy_threshold':512, # Energy threshold for sound detection
                          'cc_threshold':0.16, # GCC threshold for sound detection
                          'minimum_pairs':5,  # Minimum number of mic. pairs with a GCC above the threshold
                          'pair_ids':[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2,3)],
                          }
        }
    else:
        with open(args.ap_conf_path, 'r') as jsonfp:
            ap_conf = json.load(jsonfp)

    print('TDOA estimator config.')
    check_tdoae_conf_format(ap_conf)
    print(json.dumps(ap_conf, indent=4))
    print('')
    test_tdoa_estimation(args.input_audio_paths,
                         args.out_prefix,
                         ap_conf,
                         args.samplerate)
