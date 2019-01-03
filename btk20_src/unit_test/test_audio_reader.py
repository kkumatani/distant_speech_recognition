from btk20.common import *
from btk20.stream import *
from btk20.feature import *

D = 160
input_filename = "../tools/filterbank/Headset1.wav"
output_filename = "test.wav"
SF_FORMAT_WAV  = 0x010000
SF_FORMAT_PCM_16 = 0x0002

itrFeature = IterativeSingleChannelSampleFeaturePtr(block_len=D)
sampleFeature = SampleFeaturePtr(block_len=D, shift_len=D, pad_zeros=False)
sampleFeature.read(input_filename)
sampleFeature.write(fn=output_filename, format=SF_FORMAT_WAV+SF_FORMAT_PCM_16 )
