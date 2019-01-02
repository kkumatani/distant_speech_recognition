#!/usr/bin/python
"""
Test the oversampled DFT-modulated filter bank such as de Haan and Nqyust(M) filter.

The filter prototypes have to be generated with design_de_haan_filter.py or design_nyquist_filter.py.

References:
[1] Jan Mark De Haan, Nedelko Grbic, Ingvar Claesson, Sven E Nordholm, "Filter bank design for subband adaptive microphone arrays", IEEE TSAP 2003.

[2] Kenichi Kumatani, John McDonough, Stefan Schacht, Dietrich Klakow, Philip N Garner, Weifeng Li, "Filter bank design based on minimization of individual aliasing terms for minimum mutual information subband adaptive beamforming", ICASSP 2018.

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""
import os.path
import pickle
import wave
import sys
import numpy

from btk20.common import *
from btk20.stream import *
from btk20.feature import *
from btk20.modulated import *

def test_oversampled_dft_filter(analysis_filter_path,
                                synthesis_filter_path,
                                M, m, r,
                                audio_path,
                                out_path,
                                samplerate=16000):

    D = M / 2**r # frame shift

    # Read analysis prototype 'h'
    with open(analysis_filter_path, 'r') as fp:
        print('Loading analysis prototype from \'%s\'' %analysis_filter_path)
        h_fb = pickle.load(fp)

    # Read synthesis prototype 'g'
    with open(synthesis_filter_path, 'r') as fp:
        print('Loading synthesis prototype from \'%s\'' %synthesis_filter_path)
        g_fb = pickle.load(fp)

    # Instantiation of an audio file reader
    sample_feat = SampleFeaturePtr(blockLen = D, shiftLen = D, padZeros = True)
    # Instantiation of over-sampled DFT analysis filter bank
    afb = OverSampledDFTAnalysisBankPtr(sample_feat, prototype = h_fb, M = M, m = m, r = r, delay_compensation_type=2)
    # Instantiation of over-sampled DFT synthesis filter bank
    sfb = OverSampledDFTSynthesisBankPtr(afb, prototype = g_fb, M = M, m = m, r = r, delay_compensation_type=2)
    # Read the audio file
    sample_feat.read(audio_path, samplerate)

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    wavefile = wave.open(out_path, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(int(samplerate))

    # Reconstruct the signal through the analysis and synthesis filter bank
    synthesized_signal = []
    for b in sfb:
        storewave = numpy.array(b, numpy.int16)
        wavefile.writeframes(storewave.tostring())
        synthesized_signal.extend(b)

    wavefile.close()

    # Obtain the original input signal
    sample_feat.read(audio_path, samplerate)
    original_signal = []
    for i in sample_feat:
        original_signal.extend(i)

    # Measure the root mean square error (RMSE)
    synthesized_signal = numpy.array(synthesized_signal)
    original_signal = numpy.array(original_signal)
    diff = synthesized_signal - original_signal
    rmse = numpy.sqrt(numpy.inner(diff, diff) / len(diff))
    print('RMSE: {}'.format(rmse))

    # Compute the ratio of synthesized signal strength to the original signal
    ratios = []
    for i in range(len(synthesized_signal)):
        if synthesized_signal[i] > 0:
            ratios.append( abs(original_signal[i] / synthesized_signal[i]) )
    print('Amplification ratio: {}'.format(numpy.average(numpy.array(ratios))))


def build_parser():
    import argparse

    M = 64 #128 256
    m = 2 #4
    r = 1 #3

    # protoPath    = 'prototype.ny'
    # analysis_filter_path  = '%s/h-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)
    # synthesis_filter_path = '%s/g-M%d-m%d-r%d.pickle' %(protoPath, M, m, r)

    v = 100.0
    wpW = 1
    protoPath    = 'prototype.dh'
    analysis_filter_path  = '%s/h-M=%d-m=%d-r=%d-v=%0.4f-w=%0.2f.pickle' %(protoPath, M, m, r, v, wpW)
    synthesis_filter_path = '%s/g-M=%d-m=%d-r=%d-v=%0.4f-w=%0.2f.pickle' %(protoPath, M, m, r, v, wpW)
    out_path = './wav/M=%d-m=%d-r=%d_oversampled.wav' %(M, m, r)

    parser = argparse.ArgumentParser(description='run subband processing with oversampled DFT-modulated filter bank.')
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
    parser.add_argument('-i', dest='audio_path',
                        default='Headset1.wav',
                        help='input audio file')
    parser.add_argument('-o', dest='out_path',
                        default=out_path,
                        help='output audio file')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    test_oversampled_dft_filter(args.analysis_filter_path,
                                args.synthesis_filter_path,
                                args.M, args.m, args.r,
                                args.audio_path,
                                args.out_path,
                                samplerate=16000)
