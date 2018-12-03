import Numeric

# @class estimate the direction of arrival based on steering response power (SRP)
# @usage
# @note There is no band shift. 
class SRPbasedDOAEstimator_LA:
    def __init__( self, fftLen, sampleRate, sspeed = 343740.0 ):
        self._fftLen  = fftLen
        self._fftLen2 = fftLen/2
        self._sampleRate = sampleRate
        self._sspeed = sspeed
        self._XPositionsOfMicrophones = []
        self._deltaSin = 0.05
        self._arrayManifold = []
        self._nChan = 0
        self._steeringMatrix = [] # [self._fftLen2+1][self._nDOA][self._nChan]
        self._nDOA = 0
        self._Y2   = []
        self._maxY2    = 0.0
        self._argMaxY2 = 0
        self._minDOA = -1.5707963267948966
        self._maxDOA =  1.5707963267948966

    def setXPositionsOfMicrophones( self, mpos ):
        self._nChan = len(mpos)
        self._XPositionsOfMicrophones = []
        maxDist = 0.0
        for chanX in range(self._nChan):
            self._XPositionsOfMicrophones.append( mpos[chanX, 0] )
            if ( mpos[chanX, 1] != 0.0 ):
                print 'The Y component has to be zero'
            if ( mpos[chanX, 2] != 0.0 ):
                print 'The Z component has to be zero'
            dist = abs( mpos[0, 0] - mpos[chanX, 0] )
            if dist > maxDist:
                maxDist = dist
        
         self._deltaSin = 0.99 * self._sspeed / ( maxDist * self._sampleRate )

    def calcSteeringMatrix( self, baseMicX = -1 ):
        self._nDOA = 0
        sintheta = self._minDOA
        while sintheta <= self._maxDOA:
            self._nDOA  += 1
            sintheta += self._deltaSin

        if baseMicX < 0:
            baseMicX = self._nChan / 2
        phaseShifts = Numeric.zeros( self._nChan, Numeric.Float )
        J = (0+1j)
        Delta_f = J * 2.0 * Numeric.pi * self._sampleRate / float(self._fftLen)
        for chanX in range(0,self._nChan):
            if baseMicX != chanX :
                d = self._XPositionsOfMicrophones[chanX] - self._XPositionsOfMicrophones[baseMicX]
                phaseShifts[chanX] = - d * Delta_f

        self._steeringMatrix = Numeric.zeros( (self._fftLen2+1, self._nDOA, self._nChan) , Numeric.Complex )
        norm = self._nChan
        for fbinX in range(self._fftLen2+1):
            for dirX in range(self._nDOA):
                sintheta = self._minDOA + dirX * self._deltaSin
                for chanX in range(self._nChan):
                    #vs[chanX] = Numeric.exp(-J*2.0*Numeric.pi*fbinX*sampleDelays[chanX]) / norm
                    self._steeringMatrix[fbinX][dirX][chanX] = Numeric.exp( phaseShifts[chanX] * fbinX * sintheta ) / norm

        #self._Y2 = Numeric.zeros( self._nDOA, Numeric.Float )

        return 


    def calcSRP( self, X, lowerFBin = 1, upperFBin = -1 ):
        # @param X[fftLen/2+1][chanN] : input vector
        # @return DOA (radian)

        if upperFBin < 0:
            upperFBin = self._fftLen2 + 1

        self._Y2 = Numeric.zeros( self._nDOA, Numeric.Float )
        self._maxY2    = 0.0
        self._argMaxY2 = 0
        for dirX in range(self._nDOA):        
            for fbinX in range(lowerFBin,upperFBin):
                Y = abs( Numeric.matrixmultiply( Numeric.conjugate(self._steeringMatrix[fbinX][dirX]), X[fbinX] ) )
                self._Y2[dirX] += Y * Y
            if self._Y2[dirX] > maxY2:
                self._maxY2 = self._Y2[dirX]
                self._argMaxY2 = dirX

        return ( self._minDOA + self._argMaxY2 * self._deltaSin )


