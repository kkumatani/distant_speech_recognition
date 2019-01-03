#!/usr/bin/python
"""
Compute log power feature from an audio file
"""
import pickle, numpy
from btk20.common import *
from btk20.stream import *
from btk20.feature import *

D = 160 # 10 msec for 16 kHz audio
fft_len = 256
pow_num = fft_len//2 + 1
input_filename = "../tools/filterbank/Headset1.wav"
output_filename = "log_power.pickle"

# Audio file reader
samplefe  = SampleFeaturePtr(block_len=D, shift_len=D, pad_zeros=False)
# Hamming window calculator
hammingfe = HammingFeaturePtr(samplefe)
# FFT feature extractor
fftfe     = FFTFeaturePtr(hammingfe, fft_len=fft_len)
# Power (complex square) feature extractor
powerfe   = SpectralPowerFeaturePtr(fftfe, pow_num=pow_num)
# Log feature extractor
logfe     = LogFeaturePtr(powerfe)

# Reading the audio file
samplefe.read(input_filename)

with open(output_filename, 'w') as ofp:
    frame_no = 0
    # compute the log power feature at each frame
    for log_vector in logfe:
        # print the first 10-dimension vector
        print('fr. {}: {}..'.format(frame_no, numpy.array2string(log_vector[0:10], formatter={'float_kind':lambda x: "%.2f" % x})))
        pickle.dump(log_vector, ofp, True)
        frame_no += 1
