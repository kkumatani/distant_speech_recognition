# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.sound
#   Purpose: Handling sound files.
#   Author:  Oliver Schrempf


"""
IO routines for ADC data

This module can read, write and play ADC sound data. All sound formats from
Janus5 are supported.
"""

from __future__ import generators
import Numeric
from btk import feature
from MLab import *

"""
Low level functions
"""

def read(fname, headMode="auto", byteMode="auto",
         chX=1, chN=1, cfrom=0, cto=-1, force=0):
    """
    returns data

    Read fname and save the ADC data to the NumPy array data
    """
    if fname[-4:] == ".shn":
	byteMode = "shorten"
    return _feature.read(fname, headMode, byteMode,
                            chX, chN, cfrom, cto, force)

def write(fname, data, headMode="auto", byteMode="auto"):
    """
    write NumPy array data to filename
    """
    return _feature.write(fname, data, headMode, byteMode)


"""
high level API
"""

class SoundSource:
    """
    SoundSource is the abstract base class of sound sources. This can
    be single files, a collection of files (microphone arrays) or
    sound cards.
    The iterator returns SoundBuffers with the constant length of a
    buffersize until the end of the source is reached.
    The number of channels in the SoundBuffer depends on the source.
    """
    def __init__(self, blkLen):
        self.__blkLen = blkLen

    def __iter__(self):
        pass

class FileSoundSource(SoundSource):
    """
    Class for sound data read from files.
    """
    def __init__(self, fname = "", samplingRate=0, headMode="auto", byteMode="auto",
                 chX=1, chN=1, cfrom=0, cto = -1, blkLen = 512, force=0,
                 delay = 0, lastBlk = "stuffed"):
        self._byteMode     = _feature.getADCbyteMode()
        self._fileName     = fname
        self._headMode     = headMode
        self._byteMode     = byteMode
        self._chX          = chX
        self._chN          = chN
        self._cfrom        = 0
        self._blkLen       = blkLen
        self._force        = force
        self._lastBlk      = lastBlk

        if fname != "":
            self._data         = read(self._fileName, self._headMode, self._byteMode,
                                      self._chX, self._chN, cfrom, cto, self._force)
            self._dataLen      = len(self._data)

        if delay > 0:
            delayedData         = Numeric.zeros(delay + len(self._data))
            delayedData[delay:] = self._data[:]
            self._data          = delayedData

        if samplingRate == 0:
            self._samplingRate = _feature.getADCsamplRate()[1]
        else:
            self._samplingRate = samplingRate


    def read(self, fname, cfrom=0, cto = -1):
        self._data         = read(self._fileName, self._headMode, self._byteMode,
                                  self._chX, self._chN, cfrom, cto, self._force)
        self._dataLen      = len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __getstate__(self):
        return [self._samplingRate, self._byteMode,
                self._fileName, self._cfrom, self._blkLen]

    def __setstate__(self, state):
        (self._samplingRate, self._byteMode,
         self._fileName, self._cfrom, self._blkLen) = state

    def __iter__(self):
        while (1):
            to = (self._cfrom + self._blkLen)
            if (to > self._dataLen):
                if (self._lastBlk == "truncated"):
                    raise StopIteration
                elif (self._lastBlk == "unmodified"):
                    final = self._data[self._cfrom:]
                elif (self._lastBlk == "stuffed"):
                    final = Numeric.zeros(self._blkLen, "s")
                    final[0:(self._dataLen-self._cfrom)] = self._data[self._cfrom:]
                else:
                    raise ValueError, 'Argument lastBlk has unrecognized value'
                yield final
                raise StopIteration
            else:
                yield self._data[self._cfrom:to]
            self._cfrom += self._blkLen


    def reset(self):
        self._cfrom = 0

    def zeroMean(self):
        print 'Zeroing mean ...'
        m  = sum(self._data)
        m  = int(m / self._dataLen)
        print 'Mean = ', m
        # self._data -= m

    def cut(self, cfrom, cto):
        cfrom         = int(cfrom)
        cto           = int(cto)
        print 'Cutting from ', cfrom, ' to ', cto
        self._data    = self._data[cfrom:cto]
        self._dataLen = len(self._data)

        print 'Data length = ', self._dataLen
        # print self._data[0:25]

    def addInterference(self, waveFile, dBLevel = 0.0, rolloff = 0, cut = 0):
        interference = read(waveFile, self._headMode, self._byteMode,
                            self._chX, self._chN, 0, -1, self._force)
        if len(interference) < self._dataLen:
            raise ValueError, "Length of interference is less than length of sound"
        if len(interference) > self._dataLen:
            interference = interference[:self._dataLen]
        level = 10.0**(dBLevel / 10.0)
        self._data = (self._data + (interference * level)).astype(Int)

        if (rolloff != 0 or cut != 0) and self._dataLen > 2 * rolloff + cut:
            window = ones(self._dataLen, Float)
            window[:cut] = zeros(cut,Float)
            window[cut:rolloff+cut] = hamming(2*rolloff)[:rolloff]
            window[self._dataLen-(rolloff+1):] = hamming(2*rolloff)[rolloff-1:]
            self._data = (self._data * window).astype(Int)

    def __str__(self):
        retval = """FeatureADC:
        File:\t%s
        Samplingrate:\t%s Hz
        Samples:\t%s,
        Byte mode:\t%s""" % (self._fileName, self._samplingRate,
                             len(self._data), self._byteMode)
        return retval


class OffsetCorrectedFileSoundSource(FileSoundSource):
    """
    Class to produce offset corrected Sound input with smoothing parameter alpha
    """
    def __init__(self, fname = "", headMode="auto", byteMode="auto",
                    chX=1, chN=1, cfrom=0, cto=-1, blkLen = 512, force=0, delay = 0, alpha = 0.02, lastBlk = "stuffed"):
        FileSoundSource.__init__(self, fname=fname, headMode=headMode, byteMode=byteMode, chX=chX, chN=chN, cfrom=cfrom, cto=cto, blkLen=blkLen, force=force, delay=delay, lastBlk=lastBlk)
        self._alpha = alpha
        self._beta  = 1-alpha
        self._mu    = 0

    def __iter__(self):
        while (1):
            to = (self._cfrom + self._blkLen)
            if (to > self._dataLen):
                if (self._lastBlk == "truncated"):
                    raise StopIteration
                elif (self._lastBlk == "unmodified"):
                    block = self._data[self._cfrom:]
                elif (self._lastBlk == "stuffed"):
                    block = Numeric.zeros(self._blkLen)
                    block[0:(self._dataLen-self._cfrom)] = self._data[self._cfrom:]
                else:
                    raise ValueError, 'Argument lastBlk has unrecognized value'
                for t in range(len(block)):
                    self._mu = self._beta * self._mu + self._alpha * block[t]
                    corr = block[t] - self._mu
                    if (abs(corr) < 2**15):
                        block[t] = corr
                    else:
                        block[t] = sign(corr) * ( 2**15 - 1 )
                yield block
                raise StopIteration
            else:
                block = self._data[self._cfrom:to]
                for t in range(len(block)):
                    self._mu = self._beta * self._mu + self._alpha * block[t]
                    corr = block[t] - self._mu
                    if (abs(corr) < 2**15):
                        block[t] = corr
                    else:
                        block[t] = sign(corr) * ( 2**15 - 1 )
                yield block
            self._cfrom += self._blkLen


class MCFileSoundSource:
    """
    class to read from multichannel wavefiles
    """
    def __init__(self, fname, cfrom=0, blkLen = 512, srate = 16000):
        self._WAV_HEADER_LEN = 44
        self._fname = fname
        self._cfrom = cfrom
        self._blkLen = blkLen
        self._srate = srate
        self._chIt = []
        self._chGen = []
        self._sf = AudioOpen(self._fname,
                       mode = 'r',
                       samplerate = self._srate,
                       verbose=1)
        self._chN = self._sf.nchannels
        for i in range(self._chN):
            self._chIt.append(MCIterator(chX = i,
                                         MCFSS = self,
                                         cfrom = self._cfrom,
                                         blkLen = self._blkLen))
        for j in self._chIt:
            self._chGen.append(iter(j))
        
        
    def chan(self, chN):
        """
        Iterates Generators for channel 0 to chN-1
        """
        ch = 0
        while (ch < self._chN) :
            yield self._chGen[ch]
            ch = ch + 1
        raise StopIteration
    

    def getData(self, chX, cfrom ):
        """
        Returns Sampleblock from channel chX the length of blkLen starting at cfrom
        """
        offset = self._sf.bytespersample * cfrom * self._chN + self._WAV_HEADER_LEN
        if(self._chN * self._sf.bytespersample * self._sf.totalsamples() > offset):
            self._sf.seek(offset)
            samples = self._sf.readsamples(self._blkLen)[:,:,chX]
            return samples
        else:
            raise StopIteration
        
class MCIterator(SoundSource):
    """
    Iterates through samples of one channel in a multichannel file
    """
    def __init__(self, chX, MCFSS, cfrom, blkLen):
        self._chX = chX
        self._mcfss = MCFSS
        self._cfrom = cfrom
        self._blkLen = blkLen
        self._j = 0
        
    def __iter__(self):
        while(1):
            yield self._mcfss.getData(self._chX, self._cfrom)
            self._cfrom += self._blkLen
 

# mute _featureADC.so
#_feature.adc_verbosity(0)
