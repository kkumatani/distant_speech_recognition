#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
# 
# Copyright (c) 2018 Kenichi Kumatani
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Basic classes to read/write a binary Kaldi ark file
"""

import struct, numpy

BFM_SYM = 'BFM '
BIV_SYM = 'B'
FEAT_BYTE_SIZE = '\x04'
NULLC = '\0'
WAV_SYM = 'RIFF'

class KaldiArkReader:
    """
    Base class for readling a Kaldi ark file
    """
    def __init__(self, store_image=False):
        """
        Constructor of KaldiArkReader

        :params store_image: Every utterance data in the ark file will be kept in RAM if True
        """
        self.arkfp = None
        self.curr_arkfile = None
        if store_image == True:# store all the utterance data into image
            self.arkdata = {} # arkdata[ID] = {matrix|vector}
            self.uttids = [] # remember the order of utterance IDs in an ark file
        else:
            self.arkdata = None
            self.uttids = None

    def __enter__(self):
        return self

    def __iter__(self):
        """
        Read each utterance from the ark file and return it

        :returns : Python dictionary that contains the utterance ID as a key and data as a value
        """
        raise NotImplemented('Implement this')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def accumulate(self, uttid, dataelem):
        """
        Store all the utterance data into the RAM if this is constructed with store_image = True.
        """
        if self.arkdata is None:
            self.arkdata = {}
            self.uttids = []

        self.arkdata[uttid] = dataelem
        self.uttids.append(uttid)

    def open(self, arkfile):
        """
        Set the ark file to be read later
        """
        if self.arkfp is not None:
            raise IOError('call close() first')

        self.arkfp = open(arkfile, 'rb')
        self.curr_arkfile = arkfile

    def close(self):
        """
        Close the file pointer if it is opened
        """
        if self.arkfp is not None:
            self.arkfp.close()
            self.arkfp = None
            self.curr_arkfile = None
        if self.arkdata is not None:
            self.arkdata = {}
            self.uttids = []

    def seek(self, position_in_bytes):
        """
        Skip the file pointer. You can pick up the file position from .scp file
        """
        if self.arkfp is not None:
            self.arkfp.seek(position_in_bytes, 0)


class KaldiFeatArkReader(KaldiArkReader):
    """
    Read a Kaldi .feat.ark file per utterance iteratively
    """
    def __init__(self, store_image=False):
        KaldiArkReader.__init__(self, store_image)

    def __iter__(self):
        uttid = ''

        while True:
            arkdata = self.arkfp.read(1)
            if arkdata == '':
                raise StopIteration('End of feat ark file')
            c = struct.unpack('<s', arkdata)[0]
            if c == ' ':
                arkdata = self.arkfp.read(1) # skip '\0'
                arkdata = self.arkfp.read(4) # read the end symbol 'BFM '
                endSym = struct.unpack('<4s', arkdata)[0]
                if endSym != BFM_SYM:
                    raise ValueError('ERROR: %s could not find BFM but %s' %(self.curr_arkfile, endSym))
                arkdata = self.arkfp.read(1) # skip one byte data '\x04'
                arkdata = self.arkfp.read(4) # read no. frames
                frameN = struct.unpack( '<I', arkdata )[0]
                arkdata = self.arkfp.read(1) # skip one byte data '\x04'
                arkdata = self.arkfp.read(4) # read the dimension
                featD = struct.unpack( '<I', arkdata )[0]
                coeffN = frameN * featD
                # read the coefficients
                arkdata = self.arkfp.read(coeffN * 4)
                feMat = numpy.reshape(struct.unpack('<%df' %(coeffN), arkdata), (frameN,featD))
                if self.arkdata is not None:
                    self.accumulate(uttid, feMat)
                uttid2data = {uttid:feMat}
                uttid = ''
                yield uttid2data
            else:
                uttid += str(c)


class KaldiIntVectorArkReader(KaldiArkReader):
    """
    Read a Kaldi integer-vector file per utterance iteratively
    """
    def __init__(self, store_image=False):
        KaldiArkReader.__init__(self, store_image)

    def __iter__(self):
        uttid = ''

        while True:
            arkdata = self.arkfp.read(1)
            if arkdata == '':
                break
            c = struct.unpack('<s', arkdata)[0]
            if c == ' ':
                arkdata = self.arkfp.read(1) # skip '\0'
                arkdata = self.arkfp.read(1) # read the end symbol 'B'
                endSym = struct.unpack('<s', arkdata)[0]
                if endSym != BIV_SYM:
                    raise ValueError('ERROR: %s: Unmatched symbol %s!=%s' %(self.curr_arkfile, endSym, BIV_SYM))
                arkdata = self.arkfp.read(1) # skip one byte data '\x04'
                arkdata = self.arkfp.read(4) # read no. frames
                frameN = struct.unpack('<i', arkdata)[0]
                # read the coefficients
                vals = []
                for i in range(frameN):
                    arkdata = self.arkfp.read(1)
                    arkdata = self.arkfp.read(4)
                    vals.append(struct.unpack('<i', arkdata)[0])
                intVec = numpy.array(vals)
                if self.arkdata is not None:
                    self.accumulate(uttid, intVec)
                uttid2data = {uttid:intVec}
                uttid = ''
                yield uttid2data
            else:
                uttid += str(c)


class KaldiWavArkReader(KaldiArkReader):
    """
    Read a Kaldi .wav.ark file per utterance iteratively
    """
    def __init__(self, store_image=False):
        KaldiArkReader.__init__(self, store_image)
        self.riff_header = None
        self.samplerate = None
        self.num_channels  = None

    def get_riff_header(self):
        return self.riff_header

    def get_samplerate(self):
        return self.samplerate

    def get_num_channel(self):
        return self.num_channels

    def __iter__(self):
        uttid = ''

        while True:
            arkdata = self.arkfp.read(1)
            if arkdata == '':
                raise StopIteration('End of wav ark file')

            c = struct.unpack('<s', arkdata)[0]
            if c == ' ':
                # read the 44 Byte header block of the RIFF file
                riff_header = self.arkfp.read(44) # skip '\0'
                endSym     = struct.unpack('<4s',riff_header[0:4])[0]
                dataLength = struct.unpack('<L', riff_header[40:44])[0]
                bitsPerSample = struct.unpack('<h', riff_header[34:36])[0]
                # nsamps = int(dataLength / (bitsPerSample/8)) # divide 2 (Byte)
                self.samplerate  = struct.unpack('<L', riff_header[24:28])[0]
                self.num_channels = struct.unpack('<h', riff_header[22:24])[0]
                if endSym != WAV_SYM:
                    raise ValueError('ERROR: %s: could not find %s but %s' %(self.curr_arkfile, WAV_SYM, endSym))
                if bitsPerSample != 16:
                    raise ValueError('ERROR: %s: expecting utterance with int16 format but %d bits per sample.' % (self.curr_arkfile, bitsPerSample))

                uttBinary = self.arkfp.read(dataLength)
                # expecting 16 bit per sample
                uttInt  = [struct.unpack('<h', uttBinary[i:i+2]) for i in numpy.arange(0,len(uttBinary), 2)]
                samples = numpy.array(numpy.int16(numpy.resize(uttInt, (len(uttInt),))))
                self.riff_header = riff_header
                if self.arkdata is not None:
                    self.accumulate(uttid, samples)
                uttid2data = {uttid:samples}
                uttid = ''
                yield uttid2data
            else:
                uttid += str(c)


class KaldiArkWriter:
    """
    Base class for writing a Kaldi ark file
    """
    def __init__(self):
        self.arkfp = None

    def __entry__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, arkfile):
        if self.arkfp is not None:
            raise IOError('call close() first')
        self.arkfp = open(arkfile, 'wb')

    def close(self):
        if self.arkfp is not None:
            self.arkfp.close()
            self.arkfp = None


class KaldiFeatArkWriter(KaldiArkWriter):
    """
    Write utterance data as a Kaldi .feat.ark file
    """
    def __init__(self):
        KaldiArkWriter.__init__(self)

    def write(self, uttid2feats, uttids=None):
        if uttids is None:
            uttids = uttid2feats.keys()

        for uttid in uttids:
            feMat = uttid2feats[uttid]
            frameN = len(feMat)
            featD  = len(feMat[0])
            outData = ''
            for c in uttid + ' ':
                outData += struct.pack('<c', c)
            outData += struct.pack('<c', NULLC)
            for c in BFM_SYM:
                outData +=struct.pack('<c', c)
            outData += struct.pack('<c', FEAT_BYTE_SIZE)
            outData += struct.pack('<I', frameN)
            outData += struct.pack('<c', FEAT_BYTE_SIZE)
            outData += struct.pack('<I', featD)
            self.arkfp.write(outData)
            outData = ''
            for frameX in range(frameN):
                for coeff in feMat[frameX]:
                    outData += struct.pack( '<f', coeff )
            self.arkfp.write( outData )

        self.arkfp.flush()


class KaldiIntVectorArkWriter(KaldiArkWriter):
    """
    Write utterance data as a Kaldi int-vector ark file
    """
    def __init__(self):
        KaldiArkWriter.__init__(self)

    def write(self, uttid2feats, uttids=None):
        if uttids is None:
            uttids = uttid2feats.keys()

        for uttid in uttids:
            intVec = uttid2feats[uttid]
            frameN = len(intVec)
            outData = ''
            for c in uttid + ' ':
                outData += struct.pack('<c', c)
            outData += struct.pack('<c', NULLC)
            for c in BIV_SYM:
                outData +=struct.pack('<c', c)
            outData += struct.pack('<c', FEAT_BYTE_SIZE)
            outData += struct.pack('<I', frameN)
            self.arkfp.write(outData)
            outData = ''
            for coeff in intVec:
                outData += struct.pack('<c', FEAT_BYTE_SIZE)
                outData += struct.pack('<i', coeff)
            self.arkfp.write(outData)

        self.arkfp.flush()


def correct_chunk_size(numSamples, riff_header):
    """
    Correct the data length in header information; see http://soundfile.sapp.org/doc/WaveFormat/ for details
    """
    bytesPerSample = struct.unpack( '<h', riff_header[34:36] )[0] / 8
    dataLength = numSamples * bytesPerSample
    totalChunkSize = 36 + dataLength

    return (riff_header[0:4] + struct.pack('<L', totalChunkSize) + riff_header[8:40] + struct.pack('<L', dataLength) + riff_header[44:])


class KaldiWavArkWriter(KaldiArkWriter):
    """
    Write utterance data as a Kaldi .wav.ark file
    """
    def __init__(self):
        KaldiArkWriter.__init__(self)

    def write(self, uttid2feats, uttid2headers, uttids=None):
        if uttids is None:
            uttids = uttid2feats.keys()

        for uttid in uttids:
            outData = ''
            for c in uttid + ' ':
                outData += struct.pack('<c', c)
            self.arkfp.write(outData)
            samples = uttid2feats[uttid]
            # write the corrected header information
            uttid2header = correct_chunk_size(len(samples), uttid2headers[uttid])
            self.arkfp.write(uttid2header)
            outData = ''
            for samp in samples:
                outData += struct.pack('<h', samp)
            self.arkfp.write(outData)

        self.arkfp.flush()

    def dump_riff_file(self, riff_file, uttid):
        """
        Dump the data in a RIFF file into the wav ark file
        """
        outData = ''
        for c in uttid + ' ':
            outData += struct.pack('<c', c)
        self.arkfp.write(outData)

        with open(riff_file, 'rb') as riffF:
            self.arkfp.write(riffF.read())
            self.arkfp.flush()


def test():
    import argparse

    def build_parser():
        parser = argparse.ArgumentParser(description='List utterance IDs in the ark file')
        parser.add_argument('-t', '--type', default='f', help='Ark file type (i/f/w)')
        parser.add_argument('input_ark', help='input ark path')
        parser.add_argument('output_ark', help='output ark path')

        return parser

    parser = build_parser()
    args, argv = parser.parse_known_args()

    if args.type == 'f':
        reader = KaldiFeatArkReader()
        writer = KaldiFeatArkWriter()
    elif args.type == 'w':
        reader = KaldiWavArkReader()
        writer = KaldiWavArkWriter()
    else:
        reader = KaldiIntVectorArkReader()
        writer = KaldiIntVectorArkWriter()

    reader.open(args.input_ark)
    writer.open(args.output_ark)

    for uttid2data in reader:
        print('uttid: %s' %uttid2data.keys()[0])
        if args.type == 'w':
            writer.write(uttid2data, {uttid2data.keys()[0]:reader.get_riff_header()})
        else:
            writer.write(uttid2data)

    reader.close()
    writer.close()

if __name__ == '__main__':

    test()
