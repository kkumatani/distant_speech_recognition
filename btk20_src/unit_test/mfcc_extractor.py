#!/usr/bin/python
"""
Compute mel frequency cepstrum coefficient (MFCC) from an audio file
"""
import pickle, numpy
from btk20.common import *
from btk20.stream import *
from btk20.feature import *

samplerate  = 16000.0
D = 160 # 10 msec for 16 kHz audio
fft_len = 256
pow_num = fft_len//2 + 1
mel_num     = 30      # no. mel-filter bank output
lower       = 100.0   # lower frequency for the mel-filter bank
upper       = 6800.0  # upper frequency for the mel-filter bank
ncep        = 13      # no. cepstral coefficients

input_filename = "../tools/filterbank/Headset1.wav"
output_filename = "mfcc.pickle"

# Audio file reader
samplefe  = SampleFeaturePtr(block_len=D, shift_len=D, pad_zeros=False)
sample_storage = StorageFeaturePtr(samplefe)
# Hamming window calculator
hammingfe = HammingFeaturePtr(sample_storage)
# FFT feature extractor
fftfe     = FFTFeaturePtr(hammingfe, fft_len=fft_len)
# Power (complex square) feature extractor
powerfe   = SpectralPowerFeaturePtr(fftfe, pow_num=pow_num)
# Vocal tract length normalizer
vtlnfe    = VTLNFeaturePtr(powerfe, coeff_num=pow_num, edge=0.8, version=2)
# Mel-filter bank feature extractor
melfe     = MelFeaturePtr(vtlnfe, pow_num=pow_num, filter_num=mel_num, rate=samplerate, low=lower, up=upper, version=2)
# Log feature extractor
logfe     = LogFeaturePtr(powerfe)
# Cepstrum computation
cepfe     = CepstralFeaturePtr(logfe, ncep=ncep)
# Storage the MFCC feature
cep_storage = StorageFeaturePtr(cepfe)

# Reading the audio file
samplefe.read(input_filename)

with open(output_filename, 'w') as ofp:
    frame_no = 0
    # compute the MFCC at each frame
    for cep_vector in cep_storage:
        print('fr. {}: {}'.format(frame_no, numpy.array2string(cep_vector, formatter={'float_kind':lambda x: "%.2f" % x})))
        pickle.dump(cep_vector, ofp, True)
        frame_no += 1
