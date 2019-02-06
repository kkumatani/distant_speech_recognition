#!/usr/bin/python
"""
Test a speaker tracker

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""

import argparse, json
import sys, os

import numpy
from btk20.feature import *
from btk20.pytdoa import *
from btk20.pykalman import *

SSPEED = 343740.0

def check_tracker_conf_format(ap_conf):
    """
    Check the JSON data structure for speaker tracking
    """

    tracker_conf = ap_conf['tracker']
    assert 'pair_ids' in tracker_conf, 'Missing \"pair_ids\" key in JSON'

    if 'D' in tracker_conf and 'fftlen' in tracker_conf:
        assert tracker_conf['D'] <= tracker_conf['fftlen'], 'Invalid FFT length: %d < %d' %(tracker_conf['fftlen'], tracker_conf['D'])

    # Cechk the dimension of the initial position estimate vector
    if 'initial_estimate' in tracker_conf:
        if ap_conf['array_type'] == 'linear':
            assert len(tracker_conf['initial_estimate']) == 1, 'Invalid dimension: len(conf[\"tracker\"][\"initial_estimate\"]) != 1 for a linear array'
        elif ap_conf['array_type'] == 'circular':
            assert len(tracker_conf['initial_estimate']) == 2, 'Invalid dimension: len(conf[\"tracker\"][\"initial_estimate\"]) != 2 for a circular array'
        elif ap_conf['array_type']== 'planar':
            assert len(tracker_conf['initial_estimate']) == 2, 'Invalid dimension: len(conf[\"tracker\"][\"initial_estimate\"]) != 2 for a planar array'
        else:
            assert len(tracker_conf['initial_estimate']) == 3, 'Invalid dimension: len(conf[\"tracker\"][\"initial_estimate\"]) != 2 for a near-field assumption'


def ekf_source_tracking(input_audio_paths, ap_conf, samplerate, tdoa_path, trj_pos_path, ave_pos_path):
    """
    Test an extended Kalman filtering speaker tracker

    :param input_audio_paths: multiple audio files
    :param ap_conf: JSON containing array and tracker parameters
    :param samplerate: audio sampling rate
    :param tdoa_path: output file that will contain time delays of arrival
    :param trj_pos_path: output file that will contain a position estimate at each frame
    :param ave_pos_path: output file that will contain a position averaged over frames
    """
    tracker_conf = ap_conf['tracker']
    D      = tracker_conf.get('shiftlen', 4096)
    fftlen = tracker_conf.get('fftlen', D * 2)
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
                                    pair_ids =  tracker_conf['pair_ids'],
                                    spec_sources = spectra,
                                    fftlen = fftlen,
                                    samplerate = samplerate,
                                    mpos = numpy.array(ap_conf['microphone_positions']),
                                    energy_threshold = tracker_conf.get('energy_threshold', 100),
                                    minimum_pairs = tracker_conf.get('minimum_pairs', 3),
                                    threshold     = tracker_conf.get('cc_threshold ', 0.11),
                                    sspeed = SSPEED)

    if 'initial_estimate' in tracker_conf:
        initialXk  = numpy.array(tracker_conf['initial_estimate'])
    else:
        if ap_conf['array_type'] == 'linear':
            initialXk  = numpy.zeros(1, numpy.float)
        elif ap_conf['array_type'] == 'circular':
            initialXk  = numpy.zeros(2, numpy.float)
        else:
            initialXk  = numpy.zeros(3, numpy.float)

    # Detect the beginning of speech
    time_delta = float(D) / samplerate
    frame_no   = 0
    while True:
        Xk = frontend.instantaneous_position(frame_no)
        frame_no += 1
        if Xk[0] > -1e10: # Coherent sound source detected
            initialXk = Xk
            break

    # Instantiate the Kalman filter tracker object
    sizeXk      = len(initialXk)
    sigmaU2     = tracker_conf.get('sigmaU2', 10.0) # diagonal loading for the process noise covariance matrix
    if tracker_conf['type'] == 'iekf':
        tracker = IteratedExtendedKalmanFilter(frontend,
                                        F = numpy.identity(sizeXk, numpy.float), # state transition matrix
                                        U = sigmaU2 * numpy.identity(sizeXk, numpy.float), # process noise covariance matrix
                                        sigmaV2    = tracker_conf.get('sigmaV2', 4.0E-04), # diagonal loading for measurement noise covariance matrix
                                        sigmaK2    = tracker_conf.get('sigmaK2', 1.0E+10), # initial Kalman gain
                                        time_delta = time_delta,
                                        initialXk  = initialXk,
                                        gate_prob  = tracker_conf.get('gate_prob', 0.95),
                                        boundaries = numpy.array(tracker_conf.get('boundaries', None)), # Search space
                                        num_iterations = tracker_conf.get('num_iterations', 3), 
                                        iteration_threshold = tracker_conf.get('iteration_threshold', 1e-4)
                                        )
    elif tracker_conf['type'] == 'ekf':
        tracker = ExtendedKalmanFilter(frontend,
                                        F = numpy.identity(sizeXk, numpy.float), # state transition matrix
                                        U = sigmaU2 * numpy.identity(sizeXk, numpy.float), # process noise covariance matrix
                                        sigmaV2    = tracker_conf.get('sigmaV2', 4.0E-04), # diagonal loading for measurement noise covariance matrix
                                        sigmaK2    = tracker_conf.get('sigmaK2', 1.0E+10), # initial Kalman gain
                                        time_delta = time_delta,
                                        initialXk  = initialXk,
                                        gate_prob  = tracker_conf.get('gate_prob', 0.95),
                                        boundaries = numpy.array(tracker_conf.get('boundaries', None)) # Search space
                                        )

    print('Initial: %s' %(numpy.array_str(initialXk)))
    # Peform sound stracking
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

        elapsed_time = frame_no * time_delta
        num_tdoa = 0
        sum_pos = None
        num_pos = 0
        tracker.set_time(frame_no)
        for Xk in tracker:
            if num_tdoa > 0:
                tdoa_fp.write(',\n')
            tdoa_fp.write(json.dumps([elapsed_time, frontend.mic_pair_tdoa()]))
            if tracker.is_observed():
                if sum_pos is None:
                    sum_pos = Xk
                else:
                    sum_pos += Xk
                    trj_pos_fp.write(',\n')

                print('%0.3f: %s' %(elapsed_time, numpy.array_str(Xk)))
                trj_pos_fp.write(json.dumps(dump_position(elapsed_time, Xk)))
                num_pos +=1

            num_tdoa += 1
            elapsed_time += time_delta

        tdoa_fp.write('\n]')
        trj_pos_fp.write('\n]}')

        # write the position estimates averaged over a stream
        if num_pos > 0:
            with open(ave_pos_path, 'w') as ave_pos_fp:
                sum_pos /= float(num_pos)
                ave_pos_fp.write(json.dumps({'positions':[dump_position(0.0, sum_pos)]}))


def test_source_tracking(input_audio_paths, out_prefix, ap_conf, samplerate):
    """
    Calling a speaker tracking function
    """
    tdoa_path    = out_prefix + '.tdoa.json'
    trj_pos_path = out_prefix + '.trj.pos.json'
    ave_pos_path = out_prefix + '.ave.pos.json'
    if not os.path.exists(os.path.dirname(ave_pos_path)):
        try:
            os.makedirs(os.path.dirname(ave_pos_path))
        except:
            pass

    ekf_source_tracking(input_audio_paths, ap_conf, samplerate, tdoa_path, trj_pos_path, ave_pos_path)


def build_parser():

    default_input_audio_paths = ['data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c1.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c2.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c3.wav',
                                 'data/CMU/R1/M1005/KINECT/RAW/segmented/U1001_1M_16k_b16_c4.wav']
    default_out_prefix =  'out/U1001_1M_track'

    parser = argparse.ArgumentParser(description='test source tracking')
    parser.add_argument('-i', dest='input_audio_paths', nargs='+',
                        default=default_input_audio_paths,
                        help='observation audio file(s)')
    parser.add_argument('-o', dest='out_prefix',
                        default=default_out_prefix,
                        help='output file prefix for tracking results')
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
                 'tracker':{'type':'iekf', # 'ekf' or 'iekf' 
                            'shiftlen':4096,
                            'fftlen':8192,
                            'energy_threshold':100, # Energy threshold for sound detection
                            'cc_threshold':0.11, # GCC threshold for sound detection
                            'minimum_pairs':3,  # Minimum number of mic. pairs with a GCC above the threshold
                            'initial_estimate':[0], # Initial position estimate
                            'pair_ids':[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2,3)],
                            'sigmaV2':4.0E-04, # Diagonal loading factor for measurement noise covariance matrix
                            'sigmaU2':10.0,    # Diagonal loading factor for process noise covariance matrix
                            'sigmaK2':1.0E+10, # Initial Kalman gain
                            'gate_prob':0.95,  # Gate probablity
                            'boundaries':[[-numpy.pi, numpy.pi],
                                          [-numpy.pi, numpy.pi],
                                          [-numpy.pi, numpy.pi]]}
        }
    else:
        with open(args.ap_conf_path, 'r') as jsonfp:
            ap_conf = json.load(jsonfp)

    print('Tracker config.')
    check_tracker_conf_format(ap_conf)
    print(json.dumps(ap_conf, indent=4))
    print('')
    test_source_tracking(args.input_audio_paths,
                        args.out_prefix,
                        ap_conf,
                        args.samplerate)
