"""
Do not use this script.

Every function will be moved into py_beamforming.py
"""
from __future__ import generators

import copy
import numpy


from btk.portnoff import *
from btk.beamformer import *
import pickle

SSPEED = 343740.0

def calcPolarTimeDelays(arrgeom, azimuth, elevation):
     """Calculate the delays to focus the beam on (x,y,z) in the
     medium field."""

     chanN  = len(arrgeom)

     delays = []
     c_x = - numpy.sin(elevation) * numpy.cos(azimuth)
     c_y = - numpy.sin(elevation) * numpy.sin(azimuth)
     c_z = - numpy.cos(elevation)
     for i in range(chanN):
         t = (c_x * arrgeom[i,0] + c_y * arrgeom[i,1] + c_z * arrgeom[i,2]) /SSPEED
         delays.append( t )

     delays = numpy.array(delays, numpy.float)

     return delays


def calcDelays(x, y, z, mpos):
    """Calculate the delays to focus the beam on (x,y,z) in the
    medium field."""

    chanN  = len(mpos)

    delays = []
    for i in range(chanN):
        t = numpy.sqrt((x-mpos[i,0])**2+(y-mpos[i,1])**2+(z-mpos[i,2])**2) / SSPEED
        delays.append( t )

    delays = numpy.array(delays, numpy.float)

    # Normalize by delay of the middle element
    mid = delays[chanN/2]
    for i in range(len(mpos)):
        delays[i] -= mid

    return delays


class SpectralSource:
    def __init__(self, fftLn = 512,  nBlks = 8, sbSmpRt = 2):
        self._fftLen      = fftLn
        self._fftLen2     = fftLn / 2
        self._nBlocks     = nBlks
        self._nBlocks2    = nBlks / 2
        self._subSampRate = sbSmpRt

    def __iter__(self):
        pass

    def fftLen(self):
        return self._fftLen

    def nBlocks(self):
        return self._nBlocks

    def subSampRate(self):
        return self.subSampRate

class AnalysisFB(SpectralSource):
    """
    Class to perform sub-band analysis and sub-sampling on a
    SoundSource.
    """
    def __init__(self, fftLn = 512,  nBlks = 8, sbSmpRt = 2):
        """Initialize the analysis filter bank."""
        SpectralSource.__init__(self, fftLn, nBlks, sbSmpRt)
        self.__analFB = AnalysisFilterBank(fftLn,  nBlks, sbSmpRt)

    def __iter__(self):
        """Return the next spectral sample."""
        sampleIter = self.__soundSource.__iter__()
        while True:
            try:
                sampleBlock = sampleIter.next()
                self.__analFB.nextSampleBlock(sampleBlock)
                for isub in range(self._subSampRate):
                    yield self.__analFB.getBlock(self.__nextR)
                    self.__nextR += self._fftLen / self._subSampRate
            except StopIteration:
                sampleBlock = numpy.zeros(self._fftLen)
                for i in range(self._nBlocks):
                    self.__analFB.nextSampleBlock(sampleBlock)
                    for isub in range(self._subSampRate):
                        yield self.__analFB.getBlock(self.__nextR)
                        self.__nextR += self._fftLen / self._subSampRate
                raise StopIteration

    def nextUtt(self, soundSource):
        """Set up to process the next utterance."""
        self.__nextR      = (-self._nBlocks2 * self._fftLen
                             + self._fftLen / self._subSampRate)
        self.__soundSource = soundSource
        self.__analFB.zeroAll()


class FileSpectralSource(SpectralSource):
    """
    Read spectral samples from a file.
    """
    def __init__(self, fftLn = 512,  nBlks = 8):
        """Initialize the analysis filter bank."""
        SpectralSource.__init__(self, fftLn, nBlks)

    def __iter__(self):
        for spec in self.__specSamples:
            yield spec

    def nextUtt(self, fileName):
        """Set up to process the next utterance."""
        fp = open(fileName, 'r')
        self.__specSamples = load(fp)
        fp.close()

class CepstralFB(SpectralSource):
    """
    Class to perform subband analysis and sub-sampling on a
    SoundSource. Supports arbitrary time shifts.
    """
    def __init__(self, fftLn = 512,  nBlks = 8):
        """Initialize the analysis filter bank."""
        SpectralSource.__init__(self, fftLn, nBlks)
        self.__cepstralFB = CepstralFilterBank(fftLn,  nBlks)

    def __iter__(self):
        """Return the next spectral sample."""
        sampleIter = self.__soundSource.__iter__()

        # initialization
        nShift = (self._fftLen * self._nBlocks2) / self.__shift
        if (self._fftLen * self._nBlocks2) % self.__shift != 0:
            nShift += 1
        samples = numpy.zeros(self._fftLen * self._nBlocks2, numpy.float)

        stillToCalculate = 0
        keepGoing        = 1
        try:
            for n in range(nShift):
                samples = numpy.concatenate((samples, sampleIter.next()))
                stillToCalculate += 1
        except StopIteration:
            keepGoing = 0
            for n in range(stillToCalculate, nShift):
                samples = numpy.concatenate((samples, numpy.zeros(self.__shift, numpy.float)))

        # iteration
        self._nextTime = 0
        while (stillToCalculate > 0):
            if keepGoing == 1:
                try:
                    samples = numpy.concatenate((samples[self.__shift:], sampleIter.next()))
                    stillToCalculate += 1
                except StopIteration:
                    keepGoing = 0

            if keepGoing == 0:
                samples = numpy.concatenate((samples[self.__shift:], numpy.zeros(self.__shift, numpy.float)))

            self._nextTime += self.__shift
            specSample = numpy.zeros(self._fftLen, numpy.complex)
            self.__cepstralFB.analyze(samples[:(self._fftLen * self._nBlocks)], self._nextTime, specSample)
            yield specSample
            stillToCalculate -= 1

        raise StopIteration

    def nextUtt(self, soundSource):
        """Set up to process the next utterance."""
        self.__soundSource = soundSource
        self.__shift       = soundSource._blkLen


class SynthesisFB(SpectralSource):
    """
    Class to resynthesize speech based on sub-band samples.
    """
    def __init__(self, spectralSource):
        """Initialize the synthesis filter bank."""
        SpectralSource.__init__(self,
                                spectralSource._fftLen,
                                spectralSource._nBlocks,
                                spectralSource._subSampRate)
        self.__synthFB = SynthesisFilterBank(spectralSource._fftLen,
                                             spectralSource._nBlocks,
                                             spectralSource._subSampRate)
        self.__spectralSource = spectralSource

    def __iter__(self):
        """Return the next block of resynthesized speech."""

        # Initialization for new utterance
        self.__synthFB.zeroAll()
        nextR = (-self._nBlocks2 * self._fftLen
                 + self._fftLen / self._subSampRate)
        nextX = 0

        for nextSpecSample in self.__spectralSource:
            self.__synthFB.nextSpectralBlock(nextSpecSample)
            if (nextR >= self._nBlocks2 * self._fftLen):
                nextSpeechSample = self.__synthFB.getBlock(nextX)
                yield nextSpeechSample
                nextX += self._fftLen / self._subSampRate

            nextR += self._fftLen / self._subSampRate


def calcBlockingMatrix(vs , NC = 1 ):
    """Calculate the blocking matrix for a distortionless beamformer,
    and return its Hermitian transpose."""
    vsize    = len(vs)
    bsize    = vsize - NC
    blockMat = numpy.zeros((vsize,bsize), numpy.complex)

    # Calculate the perpendicular projection operator 'PcPerp' for 'vs'.
    norm_vs  = numpy.inner( vs, numpy.conjugate(vs) )

    if norm_vs.real > 0.0:
        PcPerp   = numpy.eye(len(vs)) - numpy.outer( numpy.conjugate(vs), vs ) / norm_vs

        # Do Gram-Schmidt orthogonalization on the columns of 'PcPerp'.
        for idim in range(bsize):
            vec      = PcPerp[:,idim]
            for jdim in range(idim):
                rvec = blockMat[:,jdim]
                ip   = numpy.inner(numpy.conjugate(rvec), vec)
                vec -= rvec * ip
            norm_vec = numpy.sqrt( abs(numpy.inner(numpy.conjugate(vec),vec)) )
            blockMat[:,idim] = vec / norm_vec

    # Debugging:
    #print "len",len(vs),len(blockMat),len(blockMat[0])
    #print matrixmultiply(vs, blockMat)

    # return numpy.conjugate(numpy.transpose(blockMat))
    return blockMat

class SubbandBeamformer(SpectralSource):
    """
    Beamformer for processing spectral samples with fixed weights.
    """
    def __init__(self, spectralSources):
        """Initialize the sub-band beamformer."""
        SpectralSource.__init__(self,
                                spectralSources[0].fftLen(),
                                spectralSources[0].nBlocks(),
                                spectralSources[0].subSampRate())
        self._spectralSources = spectralSources
        self._nChan           = len(spectralSources)
        self._snapShotArray   = SnapShotArrayPtr(self._fftLen, self._nChan)

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        # Multiply frequency domain snapshot by conjugate of
        # array manifold vector to determine array output.
        while True:
            self.updateSnapShotArray(analysisFBs)
            nextSpecSample = numpy.zeros(self._fftLen2+1).astype(numpy.complex)
            #for m in range(self._fftLen2+1):
            #    val = numpy.dot(self._waHK[m], self.getSnapShot(m))
            #    nextSpecSample[m] = val

            yield nextSpecSample

    def reset(self):
        for source in self._spectralSources:
            source.reset()

    def getSnapShot(self, fbinX):
        return self._snapShotArray.getSnapShot(fbinX)

    def updateSnapShotArray(self, analysisFBs, chanX = 0):
        """Update frequency domain snapshots with new sensor outputs.
        Return the signal energy for channel 'chanX'."""
        ichan = 0
        for analFB in analysisFBs:
            subBandSample = numpy.array(analFB.next())
            self._snapShotArray.newSample(subBandSample, ichan)
            if ichan == chanX:
                sigmaK = abs(numpy.dot(numpy.conjugate(subBandSample), subBandSample))
            ichan += 1
        self._snapShotArray.update()
        return sigmaK

    def saveWeights(self, fileName):
        fp = open(fileName, 'w')
        pickle.dump(self._waHK, fp, 1)
        fp.close()

    def loadWeights(self, fileName):
        fp = open(fileName, 'r')
        self._waHK = pickle.load(fp)
        fp.close()

    def nextSpkr(self):
        pass

    def size(self):
        return self._fftLen

    def reset(self):
        for source in self._spectralSources:
            source.reset()

class SubbandBeamformerDS(SubbandBeamformer):
    """
    Sub-band implementation of a delay-and-sum beamformer. The signal's
    direction of arrival can be specified by specifying the appropriate
    set of array manifold vectors.
    """
    def __init__(self, spectralSources, halfBandShift = False ):
        """Initialize the sub-band beamformer."""
        SubbandBeamformer.__init__(self, spectralSources)
        self._halfBandShift = halfBandShift
        self._wp = [] # post-filter weights
        self.__isamp = 0
        
    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        # Multiply frequency domain snapshot by conjugate of
        # array manifold vector to determine array output.
        while True:
            self.updateSnapShotArray(analysisFBs)
            nextSpecSample = numpy.zeros(self._fftLen).astype(numpy.complex)
            for m in range(self._fftLen2+1):
                val = numpy.dot( numpy.conjugate(self._arrayManifold[m]), self.getSnapShot(m) )
                if self.__isamp < len(self._wp):
                    nextSpecSample[m] = self._wp[self.__isamp][m] * val
                else:
                    nextSpecSample[m] = val
                if m > 0 and m < self._fftLen2:
                    nextSpecSample[self._fftLen - m] = numpy.conjugate(nextSpecSample[m])
                    
            self.__isamp += 1
            yield nextSpecSample
            
    def calcArrayManifoldVectors(self, sampleRate, delays, normalize = True ):
        """@brief Calculate one (conjugate) array manifold vector for each frequancy bin or sub-band."""

        Delta_f = sampleRate / float(self._fftLen)
        if normalize == True:
            norm = self._nChan
        else:
            norm = 1.0

        J = (0+1j)
        self._arrayManifold = []
        if self._halfBandShift:
            for fbinX in range(self._fftLen2):
                vs = numpy.zeros(self._nChan).astype(numpy.complex)
                for chanX in range(self._nChan):
                    vs[chanX] = numpy.exp(-J*2.0*numpy.pi*(0.5+fbinX)*Delta_f*delays[chanX]) / norm
                self._arrayManifold.append(vs);
            for fbinX in range(self._fftLen2,0,-1):
                vs = numpy.zeros(self._nChan).astype(numpy.complex)
                for chanX in range(self._nChan):
                    vs[chanX] = numpy.exp(-J*2.0*numpy.pi*(0.5-fbinX)*Delta_f*delays[chanX]) / norm
                self._arrayManifold.append(vs);
        else:
            for fbinX in range(self._fftLen2+1):
                vs = numpy.zeros(self._nChan).astype(numpy.complex)
                for chanX in range(self._nChan):
                    vs[chanX] = numpy.exp(-J*2.0*numpy.pi*fbinX*Delta_f*delays[chanX]) / norm
                self._arrayManifold.append(vs);
            for fbinX in range(self._fftLen2-1,0,-1):
                vs = numpy.zeros(self._nChan).astype(numpy.complex)
                for chanX in range(self._nChan):
                    vs[chanX] = numpy.exp(J*2.0*numpy.pi*fbinX*Delta_f*delays[chanX]) / norm
                self._arrayManifold.append(vs);

        return self._arrayManifold

    def setWp(self, wp ):
        print 'set the post-filter %d' %(len(wp))
        for frameX in range(len(wp)):
            self._wp.append( wp[frameX] )

    def loadWq( self, fileName  ):
        """@brief load quiescent vectors from a file."""
        
        fp = open(fileName, 'r')
        woL = pickle.load(fp)
        fp.close()
        self._arrayManifold = copy.deepcopy( woL )
        
    def nextSpkr(self):
        self.__isamp         = 0
         
def calcArrayManifold_f( fbinX, fftLen, chanN, sampleRate, delays, halfBandShift ):
    """@brief Calculate one (conjugate) array manifold vector for each frequancy bin or sub-band."""

    Delta_f = sampleRate / float(fftLen)
    J = (0+1j)
    
    vs = numpy.zeros(chanN).astype(numpy.complex)
    fftLen2 = fftLen / 2
    
    if halfBandShift:
        if fbinX < fftLen2:
            for chanX in range(chanN):
                phase = -J*2.0*numpy.pi*(0.5+fbinX)*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase ) / chanN
        else:
            for chanX in range(chanN):
                phase = -J*2.0*numpy.pi*(0.5-fftLen+fbinX)*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase ) / chanN
    else:
        if fbinX <= fftLen2:
            for chanX in range(chanN):
                vs[chanX] = numpy.exp(-J*2.0*numpy.pi*fbinX*Delta_f*delays[chanX]) / chanN
        else:
            for chanX in range(chanN):
                vs[chanX] = numpy.exp(J*2.0*numpy.pi*(fftLen-fbinX)*Delta_f*delays[chanX]) / chanN
    return vs

def calcArrayManifoldWoNorm_f( fbinX, fftLen, chanN, sampleRate, delays, halfBandShift ):
    """@brief Calculate one (conjugate) array manifold vector for each frequancy bin or sub-band."""

    Delta_f = sampleRate / float(fftLen)
    J = (0+1j)
    
    vs = numpy.zeros(chanN,numpy.complex)
    fftLen2 = fftLen / 2

    if halfBandShift:
        if fbinX < fftLen2:
            for chanX in range(chanN):
                phase = -J*2.0*numpy.pi*(0.5+fbinX)*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase )
        else:
            for chanX in range(chanN):
                phase = -J*2.0*numpy.pi*(0.5-fftLen+fbinX)*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase )
    else:
        if fbinX <= fftLen2:
            for chanX in range(chanN):
                phase = -J*2.0*numpy.pi*fbinX*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase )
        else:
            for chanX in range(chanN):
                phase = J*2.0*numpy.pi*(fftLen-fbinX)*Delta_f*delays[chanX]
                vs[chanX] = numpy.exp( phase )

    return vs

def getInverseMat22( mat ):
    """ @brief calculate the inverse matrixof 2 x 2 matrix """
    
    MINDET_THRESHOLD = 1.0E-07
    det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    # compensate for non-invertibility
    if abs( det ) < MINDET_THRESHOLD :
        print "compensate for non-invertibility"
        betaI = 0.01 * numpy.eye( 2 )
        mat = mat + betaI
        det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        #del betaI

    invMat = numpy.zeros( (2,2), numpy.complex )
    invMat[0][0] = mat[1][1]/det
    invMat[0][1] = -mat[0][1]/det
    invMat[1][0] = -mat[1][0]/det
    invMat[1][1] = mat[0][0]/det

    return invMat


def calcNullBeamformer( w_t, w_j, NC ):
    """@brief calculate a weight vector which emphasizes a target and suppresses jammer signal"""
    """@param w_t[nChan] : a constraint for a target signal"""
    """@param w_j[nChan] : a constraint for a jammer signal"""
    """@param NC: the number of constraints (it must be 2)"""

    # there are 2 linear constraints
    if NC!=2:
        print "The number of constraints must be 2:",NC
        exit(1)

    nChan = len( w_t )
    C = numpy.zeros( ( nChan, NC ), numpy.complex )
    for chanX in range(nChan):
        C[chanX][0] = w_t[chanX]
        C[chanX][1] = w_j[chanX]
        
    CH  = C.conj().T
    g   = numpy.zeros( NC, numpy.float )
    g[0] = 1
    g[1] = 0
    V  = numpy.dot( getInverseMat22( numpy.dot( CH, C ) ), g )
    wq = numpy.dot( C, V )
    
    #del C
    #del CH
    #del g
    #del V
    
    return wq

# static GSC beamformer
class SubbandBeamformerGSC(SubbandBeamformerDS):
    def __init__(self, spectralSources, halfBandShift = False, NC = 1 ):
        SubbandBeamformerDS.__init__(self, spectralSources, halfBandShift )
        self._waHK = []
        self._NC = NC
        self.__isamp = 0
        
        #for m in range(self._fftLen2+1):
        if halfBandShift==True :
            myfftLen = self._fftLen
        else:
            myfftLen = self._fftLen
        for m in range(myfftLen):
            self._waHK.append(numpy.zeros(self._nChan-NC, numpy.complex))

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        self.sigma2 = 0.0
        self.frames = 0

        while True:
            self.updateSnapShotArray(analysisFBs)
            output = numpy.zeros(self._fftLen, numpy.complex)
            #for m in range(self._fftLen2+1):
            for m in range(self._fftLen2+1):
                # Get next snapshot and form output of blocking matrix.
                XKH = numpy.conjugate(self.getSnapShot(m))
                ZK  = numpy.dot(XKH, self._blockingMatrix[m])

                # Calculate output of upper branch.
                wq  = self._arrayManifold[m]
                YcK = numpy.dot(XKH, wq)

                # Calculate complete array output.
                val = numpy.conjugate( YcK - numpy.dot(ZK, self._waHK[m]) )
                output[m] = val
                if self.__isamp < len(self._wp):
                    output[m] = self._wp[self.__isamp][m] * val
                else:
                    output[m] = val
                if m > 0 and m < self._fftLen2:
                    output[self._fftLen - m] = numpy.conjugate(output[m])
                    
                if m == 100:
                    self.sigma2 += abs(val) * abs(val)
                    self.frames += 1
                  
            yield output
            self.__isamp += 1
            
    def calcArrayManifoldVectors(self, sampleRate, delays):
	"""Calculate one (conjugate) array manifold vector
        and one blocking matrix for each sub-band."""
        SubbandBeamformerDS.calcArrayManifoldVectors(self, sampleRate, delays)
        Delta_f = sampleRate / float(self._fftLen)
        J = (0+1j)
        self._blockingMatrix = []

        if self._halfBandShift==True :
            myfftLen = self._fftLen
        else:
            myfftLen = self._fftLen

        for fbinX in range(myfftLen):
            vs = self._arrayManifold[fbinX]
            self._blockingMatrix.append(calcBlockingMatrix(vs,self._NC));

    def calcArrayManifoldVectors2(self, sampleRate, delays1, delays2 ):
	"""Calculate one (conjugate) array manifold vector
        and one blocking matrix for each sub-band in the case that a null steered beamformer is used."""
        """@param  sampleRate[] sampling rate  """
        """@param  delays1[] TDOA of the target source """
        """@param  delays2[] TDOA of the interference source"""
        
        self._arrayManifold = []
        Delta_f = sampleRate / float(self._fftLen)
        J = (0+1j)
        self._blockingMatrix = []
        if self._halfBandShift==True :
            myfftLen = self._fftLen
        else:
            myfftLen = self._fftLen

        for fbinX in range(myfftLen):
            wds1= calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays1, self._halfBandShift)
            wds2= calcArrayManifoldWoNorm_f( fbinX, self._fftLen, self._nChan, sampleRate, delays2, self._halfBandShift)
            vs = calcNullBeamformer( wds1, wds2, self._NC )
            self._arrayManifold.append( vs );
            self._blockingMatrix.append(calcBlockingMatrix(vs,self._NC));
            #del wds1
            #del wds2

    def calcBlockingMatrix(self):
        """@brief calculate blocking matrices after quiescent weights of the upper branch are fixed"""
        """@notice quiescent weights should be set with loadWq() or ..."""
        
        self._blockingMatrix = []
        if self._halfBandShift==True :
            myfftLen = self._fftLen
        else:
            myfftLen = self._fftLen

        for fbinX in range(myfftLen):
            vs = self._arrayManifold[fbinX]
            self._blockingMatrix.append(calcBlockingMatrix(vs,self._NC));

    def setWa(self, waHK ):
        """@brief set active weight vectors """
        """@param waHK[fftLen][chanN-NC] """
        
        Len = self._nChan - self._NC
        if len( waHK[0] ) != Len :
            message =  'The length of an adaptive filter must be not %d but %d' %(waHK[0],Len)
            print message
            raise  ValueError , message

        if self._halfBandShift==True :
            myfftLen = self._fftLen
        else:
            myfftLen = self._fftLen
                
        for fbinX in range(myfftLen):
           for chanX in range( Len ):
               self._waHK[fbinX][chanX] = waHK[fbinX][chanX]


class SubbandBeamformerRLS(SubbandBeamformerGSC):
    """
    Recursive least squares (RLS) beamformer implemented in
    generalized sidelobe canceller (GSC) configuration with
    fixed diagonal loading.
    """
    def __init__(self, spectralSources, forgetFactor = 0.9999,
                 initDiagLoad = 1.0E+10, beta = 0.999, silThresh = 1.0E+10,
                 alpha2 = 0.001, plotting = False, halfBandShift = False):
        """Initialize the sub-band RLS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources, halfBandShift)
        self.__mu = forgetFactor
        self.__initDiagLoad  = initDiagLoad
        self.__beta          = beta
        self.__plotFlag      = plotting
        self.__sigmaK        = initDiagLoad
        self.__silThresh     = silThresh
        self.__alpha2        = alpha2
        self.__ttlUpdates    = 0
        self.__isamp         = 0

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs)
            output = numpy.zeros(self._fftLen, numpy.complex)

            if sigmaK > (self.__sigmaK/self.__silThresh):
                self.__ttlUpdates += 1
            for m in range(self._fftLen2+1):

                # Get next snapshot and form output of blocking matrix.
                XK = self.getSnapShot(m)
                ZK = numpy.dot(numpy.conjugate(self._blockingMatrix[m]),XK)

                # Calculate output of upper branch.
                wqH = numpy.conjugate(self._arrayManifold[m])
                YcK = numpy.dot(wqH, XK)

                if sigmaK > (self.__sigmaK/self.__silThresh) and self.__isamp % self._subSampRate == 0:
                    # Calculate gain vector and update precision matrix.
                    gzK   = numpy.dot(self.__PzK[m], ZK)
                    ip    = numpy.dot(numpy.conjugate(ZK), gzK)
                    gzK  /= (self.__mu + ip)
                    temp  = numpy.dot(numpy.conjugate(ZK), self.__PzK[m])
                    PzK   = (self.__PzK[m] - numpy.outer(gzK, temp)) / self.__mu

                    # Update active weight vector.
                    epK   = YcK - numpy.dot(self._waHK[m], ZK)
                    watHK = self._waHK[m] + numpy.conjugate(gzK) * epK
                    watK  = numpy.conjugate(watHK)

                    # Still under control? Apply quadratic constraint.
                    betaK = 0.0
                    norm_watK = abs(numpy.dot(watHK, watK))
                    if norm_watK > 10.0:
                        print 'Bailing out at sample %d' %(self.__isamp)
                        waHK = numpy.zeros(self._nChan-1)
                        PzK  = numpy.identity(self._nChan-1)/self.__initDiagLoad
                    elif norm_watK > self.__alpha2:
                        va   = numpy.dot(PzK, watK)
                        a    = abs(numpy.dot(va, numpy.conjugate(va)))
                        b    = -2.0 * (numpy.dot(numpy.conjugate(va), watK)).real
                        c    = norm_watK - self.__alpha2
                        arg  = b*b - 4.0*a*c
                        if arg > 0:
                            betaK = - (b + numpy.sqrt(arg)) / (2.0 * a)
                        else:
                            betaK = - b / (2.0 * a)
                        waHK = watHK - betaK * numpy.conjugate(va)
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if m == 100 and self.__isamp % 50 == 0:
                        print ''
                        print 'Sample %d' %(self.__isamp)
                        print 'SigmaK          = %8.4e' %(sigmaK)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        norm_gzK = abs(numpy.dot(numpy.conjugate(gzK), gzK))
                        print '||gzK||^2       = %8.4e' %(norm_gzK)
                        print '||Z^H P_z Z||^2 = %8.4e' %(abs(ip))
                        print 'betaK           = %8.4e' %(betaK)
                        print '||watK||^2      = %8.4e' %(norm_watK)
                        norm_waK = abs(numpy.dot(numpy.conjugate(waHK), waHK))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)
                        if self.__plotFlag:
                            from btk.plot import plotBeamPattern
                            plotBeamPattern(wqH - numpy.dot(waHK,self._blockingMatrix[m]))

                    # Store values for next iteration
                    self.__PzK[m]  = copy.deepcopy(PzK)
                    self._waHK[m] = copy.deepcopy(waHK)

                # Calculate complete array output.
                val = YcK
                if self.__isamp > 100:
                     val -= numpy.dot(self._waHK[m], ZK)
                output[m] = val
                if m > 0 and m < self._fftLen2:
                    output[self._fftLen - m] = numpy.conjugate(val)

            # Update the average power.
            self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta)*sigmaK
            yield output
            self.__isamp += 1

    def nextSpkr(self):
        """Reset precision matrices and weight vectors for new speaker."""
        if self.__isamp > 0:
            print 'Updated weight vectors on %d of %d total frames.' %(self.__ttlUpdates, self.__isamp)

        self.__sigmaK     = self.__initDiagLoad
        self.__ttlUpdates = 0
        self.__isamp      = 0
        self.__PzK        = []
        self._waHK       = []
        for i in range(self._fftLen2+1):
            self.__PzK.append(copy.deepcopy(numpy.identity(self._nChan-1)/self.__initDiagLoad))
            self._waHK.append(numpy.zeros(self._nChan-1, numpy.complex))


class SubbandBeamformerCovarSqRoot(SubbandBeamformerGSC):
    """
    Recursive least squares (RLS) beamformer implemented in
    generalized sidelobe canceller (GSC) configuration.
    The Cholesky decomposition (i.e., square-root) of the covariance
    matrix is propogated with every iteration.
    """
    def __init__(self, spectralSources, lam = 0.95, dBLoadLevel = -20.0,
                 initPower = 1.0E+08, beta = 0.98, silThresh = 1.0E+10,
                 subBandLoad = 0, plotting = False, halfBandShift = False):
        """Initialize the sub-band RLS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources, halfBandShift)
        self.__sqrt_lambda   = numpy.sqrt(lam)
        self.__subBandLoad   = subBandLoad
        self.__loadLevel     = 10.0**(dBLoadLevel / 10.0)
        self.__initPower     = initPower
        self.__initDiagLoad  = self.__initPower * self.__loadLevel
        self.__beta          = beta
        self.__plotFlag      = plotting
        self.__sigmaK        = initDiagLoad
        self.__silThresh     = silThresh
        self.__ttlUpdates    = 0
        self.__isamp         = 0

class SubbandBeamformerInfoSqRoot(SubbandBeamformerGSC):
    """
    Recursive least squares (RLS) beamformer implemented in
    generalized sidelobe canceller (GSC) configuration.
    The Cholesky decomposition (i.e., square-root) of the information
    matrix is propogated with every iteration. The diagonal loading
    of the information matrix is recursively updated.
    """
    def __init__(self, spectralSources, lam = 0.95, dBLoadLevel = -20.0,
                 initPower = 1.0E+08, beta = 0.98,
                 subBandLoad = False, plotting = False, halfBandShift = False):
        """Initialize the sub-band RLS beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources, halfBandShift)
        self.__init_sqrt_lambda = numpy.sqrt(lam)
        self.__subBandLoad      = subBandLoad
        self.__loadLevel        = 10.0**(dBLoadLevel / 10.0)
        self.__initPower        = initPower
        self.__initDiagLoad     = self.__initPower * self.__loadLevel
        self.__beta             = beta
        self.__plotFlag         = plotting
        self.__ttlUpdates       = 0


    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        M                 = 2048
        m                 = 4
        windowLen         = M
        sampleRate        = 44100
        timeIncrement     = (1.0 * windowLen) / sampleRate # *must* be float division

        ZK = numpy.zeros(self._nChan - 1, numpy.complex)
        currentTime = 0.0

        self.__isamp       = 0

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs) / self._fftLen
            output = numpy.zeros(self._fftLen, numpy.complex)

            largest_norm_waK   = 0.0
            largest_norm_waK_m = 0
            for m in range(self._fftLen2):
                
                # Calculate output of upper branch.
                wqH = self._arrayManifold[m]/self._nChan

                # Get next snapshot and form output of blocking matrix.
                XK = wqH * self.getSnapShot(m)

                YcK = numpy.sum(XK)

                for n in range(self._nChan - 1):
                    ZK[n] = XK[n] - XK[n+1]

                if isSilence(currentTime):

                    # Propagate 'sqrt_Pm_inv'
                    a_12 = copy.deepcopy(ZK)
                    a_22 = copy.deepcopy(YcK)

                    self.__sqrt_PzK_inv[m] *= self.__sqrt_lambda
                    a_21 = numpy.dot(self._waHK[m], self.__sqrt_PzK_inv[m])

                    propagateInfoSquareRoot_RLS(self.__sqrt_PzK_inv[m], a_12, a_21, a_22)

                    # Back substitution to calculate 'waHK(t)'
                    choleskyBackSubComplex(self.__sqrt_PzK_inv[m], a_21, self._waHK[m])

                    waHK = self._waHK[m]
                    norm_waK = abs(numpy.dot(waHK, numpy.conjugate(waHK)))

                    if norm_waK > largest_norm_waK:
                        largest_norm_waK   = norm_waK
                        largest_norm_waK_m = m

                    # Refresh diagonal loading
                    load = 0.0
                    if self.__subBandLoad:
                        load = self.__sub_sigmaK[m]
                    else:
                        load = self.__sigmaK
                    load *= self.__loadLevel

                    if self.__sqrt_lambda < 1.0:
                        load *= (1.0 - (self.__sqrt_lambda)**(2.0*(self._nChan-1)))
                    else:
                        load *= (self._nChan-1)
                        
                    addDiagonalLoading(self.__sqrt_PzK_inv[m], self.__loadIndex % (self._nChan-1), numpy.sqrt(load))
                    
                    # Update subband power
                    if self.__isamp % self._subSampRate == 0:
                        subBandPower = (numpy.dot(numpy.conjugate(ZK), ZK)).real / (self._nChan-1)
                        self.__sub_sigmaK[m] = self.__sub_sigmaK[m] * self.__beta + (1.0-self.__beta) * subBandPower

                    # Dump largest weight vector
                    if self.__isamp % 50 == 0:
                        print ''
                        print 'Sample %d' %(self.__isamp)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        print 'Subband SigmaK  = %8.4e' %(self.__sub_sigmaK[largest_norm_waK_m])
                        print '||waK||^2       = %8.4e' %(largest_norm_waK)
                        print 'largest         = %d'    %(largest_norm_waK_m)
                        print 'largest waK:'
                        print abs(self._waHK[largest_norm_waK_m])
                        if self.__plotFlag:
                            from btk.plot import plotBeamPattern
                            wqH = self._arrayManifold[largest_norm_waK_m]/self._nChan
                            plotBeamPattern2(wqH, wqH - matrixmultiply(self._waHK[largest_norm_waK_m], self._blockingMatrix[largest_norm_waK_m]))

                # Calculate complete array output.
                val = YcK
                if self.__isamp > 100:
                     val -= numpy.dot(self._waHK[m], ZK)
                output[m] = val

                self.__loadIndex += 1

            # Update the average power for all subbands
            if self.__isamp % self._subSampRate == 0:
                self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta) * sigmaK

            currentTime += timeIncrement

            yield output
            self.__isamp += 1

    def nextSpkr(self):
        """Reset precision matrices and weight vectors for new speaker."""
        if self.__isamp > 0:
            print 'Updated weight vectors on %d of %d total frames.' %(self.__ttlUpdates, self.__isamp)

        self.__sqrt_lambda  = self.__init_sqrt_lambda
        self.__sigmaK       = self.__initPower
        self.__sub_sigmaK   = []
        self.__ttlUpdates   = 0
        self.__isamp        = 0
        self.__sqrt_PzK_inv = []
        self._waHK          = []
        self.__loadIndex    = 0
        for m in range(self._fftLen2+1):
            self.__sqrt_PzK_inv.append(copy.deepcopy(numpy.identity(self._nChan-1, numpy.complex)*numpy.sqrt(self.__initDiagLoad)))
            self._waHK.append(numpy.zeros(self._nChan-1,numpy.complex))
            self.__sub_sigmaK.append(self.__initPower)


class SubbandBeamformerGriffithsJim(SubbandBeamformerGSC):
    """
    Least mean square (LMS) error beamformer implemented in
    generalized sidelobe canceller (GSC) configuration with
    fixed diagonal loading. Also known as a Griffiths-Jim
    beamformer.
    """
    def __init__(self, spectralSources, beta = 0.999, gamma = 0.0005,
                 initDiagLoad = 1.0E+10, floorSubBandSigmaK = 2.0E+7,
                 silThresh = 1.0E+10, alpha2 = 0.001, plotting = False,
                 staticAfter = 1000, halfBandShift = False ):
        """Initialize the sub-band Griffiths-Jim beamformer."""
        SubbandBeamformerGSC.__init__(self, spectralSources, halfBandShift )
        self.__beta               = beta
        self.__init_gamma         = gamma
        self.__initDiagLoad       = initDiagLoad
        self.__plotFlag           = plotting
        self.__floorSubBandSigmaK = floorSubBandSigmaK
        self.__sigmaK             = initDiagLoad
        self.__silThresh          = silThresh
        self.__alpha2             = alpha2
        self.__ttlUpdates         = 0
        self.__isamp              = 0
        self.__staticAfter        = staticAfter

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        while True:
            sigmaK = self.updateSnapShotArray(analysisFBs) / self._fftLen
            output = numpy.zeros(self._fftLen, numpy.complex)

            if self.__isamp == self.__staticAfter:
                self.__gamma /= 10.0
                print 'Setting gamma = %g after sample %d' %(self.__gamma, self.__staticAfter)

            if sigmaK > (self.__sigmaK/self.__silThresh):
                self.__ttlUpdates += 1
            for m in range(self._fftLen2+1):

                # Get next snapshot and form output of blocking matrix.
                XK = self.getSnapShot(m)
                ZK = numpy.dot(numpy.transpose(numpy.conjugate(self._blockingMatrix[m])), XK)

                # Calculate output of upper branch.
                wqH = numpy.conjugate(self._arrayManifold[m])
                YcK = numpy.dot(wqH,XK)

                if sigmaK > (self.__sigmaK/self.__silThresh) : # and self.__isamp % self._subSampRate == 0:
                    # Update active weight vector.
                    epa           = YcK - numpy.dot(self._waHK[m],ZK)
                    if self.__isamp > 0 :
                        subBandSigmaK = (self.__subBandSigmaK[m] * self.__beta +
                                         (1.0-self.__beta)*abs(numpy.dot(numpy.conjugate(XK), XK)))
                        #(1.0-self.__beta)*abs(numpy.dot(numpy.conjugate(ZK), ZK)))
                    else:
                        subBandSigmaK = abs(numpy.dot(numpy.conjugate(XK), XK))
                        # abs(numpy.dot(numpy.conjugate(ZK), ZK))
                        
                    if subBandSigmaK < self.__floorSubBandSigmaK:
                        subBandSigmaK = self.__floorSubBandSigmaK

                    alphaK = self.__gamma / subBandSigmaK
                    watHK  = self._waHK[m] + epa * numpy.conjugate(ZK) * alphaK

                    norm_watK = abs(numpy.dot(watHK, numpy.conjugate(watHK)))
                    if norm_watK > self.__alpha2:
                        cK = numpy.sqrt(self.__alpha2 / norm_watK)
                        waHK = cK * watHK
                    else:
                        waHK = watHK

                    # Dump debugging info.
                    if m == 100 and self.__plotFlag : #and self.__isamp % 50 == 0:
                        print ''
                        print 'Sample %d' %(self.__isamp)
                        print 'Sub-Band SigmaK = %8.4e' %(subBandSigmaK)
                        print 'SigmaK          = %8.4e' %(sigmaK)
                        print 'Avg. SigmaK     = %8.4e' %(self.__sigmaK)
                        norm_waK = abs(numpy.dot(waHK, numpy.conjugate(waHK)))
                        print '||waK||^2       = %8.4e' %(norm_waK)
                        print 'waHK:'
                        print abs(waHK)
                        #if self.__plotFlag:
                        #    from btk.plot import plotBeamPattern
                        #    plotBeamPattern(wqH - numpy.dot(waHK,self._blockingMatrix[m]))

                    # Store values for next iteration
                    self._waHK[m]          = copy.deepcopy(waHK)
                    self.__subBandSigmaK[m] = subBandSigmaK

                # Calculate complete array output.
                val = YcK - numpy.dot(self._waHK[m], ZK)
                if self.__isamp < len(self._wp):
                    output[m] = self._wp[self.__isamp][m] * val
                else:
                    output[m] = val
                    print  self.__isamp
                if m > 0 and m < self._fftLen2:
                    output[self._fftLen - m] = numpy.conjugate(output[m])

            # Update the average power for all subbands
            self.__sigmaK = self.__sigmaK * self.__beta + (1.0-self.__beta) * sigmaK

            yield output
            self.__isamp += 1

    def nextSpkr(self):
        """Reset weight vectors for new speaker."""
        if self.__isamp > 0:
            print 'Updated weight vectors on %d of %d total frames.' %(self.__ttlUpdates, self.__isamp)

        self.__gamma         = self.__init_gamma
        self.__sigmaK        = self.__initDiagLoad
        self.__ttlUpdates    = 0
        self._waHK           = []
        self.__subBandSigmaK = []
        for i in range(self._fftLen2+1):
            self._waHK.append(numpy.zeros(self._nChan-1, numpy.complex))
            self.__subBandSigmaK.append(self.__initDiagLoad)


class SubbandBeamformerMVDR(SubbandBeamformerDS):
    """Minimum power distortionless response (MPDR) beamformer
    based on sample matrix inversion (SMI)."""
    def __init__(self, spectralSources, diagLoad = 0.001,forgetFactor = 0.99, halfBandShift = True, end = 20, wp1L = [] ):
        SubbandBeamformerDS.__init__(self, spectralSources, halfBandShift )
        self._diagLoad = diagLoad * numpy.identity( self._nChan )
        self._forgetFactor = forgetFactor
        self._end = end
        self._frame = 0
        self._SxK  = []
        self._wH   = []
	self._wp1L = wp1L
        for m in range(self._fftLen):
            self._SxK.append(numpy.zeros((self._nChan,self._nChan), numpy.complex))
            self._wH.append(numpy.zeros( self._nChan, numpy.complex))
            
    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        # Update spatial spectral matrices, estimate MPDR sensor weights,
        # then calculate vector of output sub-band samples.
        while True:
            self.updateSnapShotArray(analysisFBs)
            nextSpecSample = numpy.zeros(self._fftLen, numpy.complex)

            for m in range(self._fftLen):
                #if self.__isamp % self._subSampRate == 0 and self.__frame <= 15 :
                sample = copy.deepcopy( self.getSnapShot(m) )
                if self._frame <=  self._end :
                    self.updateSx( m, sample )
                    Sx_inverse = inverse( self._SxK[m] + self._diagLoad )
		    temp = numpy.dot( Sx_inverse, self._arrayManifold[m] )
                    wo = temp / numpy.dot( numpy.conjugate(self._arrayManifold[m]), temp  )
                    
                    # Dump debugging info.
                    if m == 100 :
                        norm_wo = numpy.dot(numpy.conjugate(wo), wo)
                        print '||waK||^2       = %8.4e' %(abs(norm_wo))
                        print abs(wo)

                    # Store values for next iteration
                    self._wH[m] = copy.deepcopy( numpy.conjugate(wo) )

                val = numpy.dot(self._wH[m], sample )
		if len( self._wp1L ) > 0 :
		    val = self._wp1L[self._frame][m] * val
                if self._halfBandShift==True:
	            nextSpecSample[m] = val
                else:
                    if m < self._fftLen2:
                        nextSpecSample[m] = val
                        if m > 0 :
                            nextSpecSample[ self._fftLen - m] = numpy.conjugate( val )
                    elif m == self._fftLen2:
                        nextSpecSample[m] = val
                        break
            
            self._frame += 1
            yield nextSpecSample
        
    def updateSx( self, fbinX, sample ):
        if self._frame == 0:
            self._SxK[fbinX] = numpy.outer( sample, numpy.conjugate(sample) )
        else :
            newSx = numpy.outer( sample, numpy.conjugate(sample) )
            self._SxK[fbinX] = self._forgetFactor * self._SxK[fbinX] + ( 1.0 - self._forgetFactor ) * newSx

    def nextSpkr( self ):
        self._frame = 0
        self._SxK = []
        self._wH  = []
        for m in range(self._fftLen):
            self._SxK.append(numpy.zeros((self._nChan,self._nChan), numpy.complex))
            self._wH.append(numpy.zeros( self._nChan, numpy.complex))

class SubbandBeamformerMPDR(SubbandBeamformerDS):
    """Minimum power distortionless response (MPDR) beamformer
    based on sample matrix inversion (SMI)."""
    def __init__(self, spectralSources, diagLoad = 0.001,forgetFactor = 0.99, halfBandShift = True, end = 20  ):
        SubbandBeamformerDS.__init__(self, spectralSources, halfBandShift )
        self._diagLoad = diagLoad * numpy.identity( self._nChan - 1 )
        self._forgetFactor = forgetFactor
        self._end = end
        self._frame = 0
        self._SxK = []
        self._waK = []
        for m in range(self._fftLen):
            self._SxK.append(numpy.zeros((self._nChan,self._nChan), numpy.complex))
            self._waK.append(numpy.zeros( self._nChan-1, numpy.complex))

    def __iter__(self):
        """Return the next spectral sample."""
        analysisFBs = []
        for source in self._spectralSources:
            analysisFBs.append(source.__iter__())

        while True:
            self.updateSnapShotArray(analysisFBs)
            output = numpy.zeros(self._fftLen, numpy.complex)
            #for m in range(self._fftLen2+1):
            for m in range(self._fftLen):
                # Get next snapshot and form output of blocking matrix.
                sample = copy.deepcopy( self.getSnapShot(m) )

                XHK = numpy.conjugate( sample )
                ZHK = numpy.dot(XHK, self._blockingMatrix[m])

                # Calculate output of upper branch.
                wq = self._arrayManifold[m]
                YcK = numpy.dot(XHK, wq)
                self.updateActiveWeight( m, sample, wq, self._blockingMatrix[m] )

                # Calculate complete array output.
                #val = YcK
                ZHK = numpy.dot(ZHK, self._waK[m])
                output[m] = numpy.conjugate(YcK - ZHK)

                # Dump debugging info.
                if m == 100 and ( self._frame % 20 ) == 0 :
                    print 'Sample %d' %(self._frame)
                    norm_wH = numpy.dot(numpy.conjugate(self._waK[m]), self._waK[m])
                    print '||waK||^2       = %8.4e' %(abs(norm_wH))
                    print abs(self._waK[m])
                    print 'output'
                    print abs(YcK),abs(ZHK),abs(output[m])
                    #print self._SxK[m]

            self._frame += 1        
            yield output

    def calcArrayManifoldVectors(self, sampleRate, delays):
	"""Calculate one (conjugate) array manifold vector
        and one blocking matrix for each sub-band."""
        SubbandBeamformerDS.calcArrayManifoldVectors(self, sampleRate, delays)
        J = (0+1j)
        self._blockingMatrix = []
        #for fbinX in range(self._fftLen2+1):
        for fbinX in range(self._fftLen):
            vs = self._arrayManifold[fbinX]
            self._blockingMatrix.append( calcBlockingMatrix(vs,1) )

    def updateActiveWeight(self, fbinX, sample, wq, B ):
        self.updateSx( fbinX, sample )
        
        temp =  numpy.dot( self._SxK[fbinX], B )
        L = numpy.dot( numpy.conjugate( wq ), temp )
        R = numpy.dot( numpy.conjugate( numpy.transpose(  B ) ),  temp )
        Rinv = inverse( R + self._diagLoad )
        self._waK[fbinX] = numpy.conjugate( numpy.dot( L, Rinv ) )
        
    def updateSx( self, fbinX, sample ):
        if self._frame == 0:
            self._SxK[fbinX] = numpy.outer( sample, numpy.conjugate(sample) )
        elif self._frame <= self._end :
            newSx = numpy.outer( sample, numpy.conjugate(sample) )
            self._SxK[fbinX] = self._forgetFactor * self._SxK[fbinX] + ( 1.0 - self._forgetFactor ) * newSx
