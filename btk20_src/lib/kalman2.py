from btk.utils import *
from btk import dbase
from btk.localization import *
from btk.cepstralFrontend import *
from Numeric import *
from SrpPhatLocalizer import *


class KalmanFilter:
    """
    Class to perform SourceLocalization by stateEstimation via KalmanFilter
    """
    
    def __init__(self, arrgeom, state1 = [2000.0, -150.0, 0.0]):

        # Define current stateEstimate
        self.currentStateEst = state1

        # Define old stateEstimate
        self.oldStateEst = self.currentStateEst

        # Define future stateEstimate
        self.futureStateEst = self.currentStateEst

        # Define array geometrie
        self.arrgeom = arrgeom

        # Specify state transition matrix
        self.F_n = identity(3)

        # Specify linear measurement matrix
        self.C_n_l = identity(3)

        # Specify current error correlation matrix
        self.alpha = 1.0
        self.K_n = self.alpha * identity(3)

        # Specify old error correlation matrix
        self.K_n_old = self.K_n

        # Specify new error correlation matrix
        self.K_n_new = self.K_n

        # Specify correlation matrix of process noise vector
        self.beta = 1.0
        self.Q_1 = self.beta * identity(3)

        # Specify correlation matrix of measurement noise vector
        self.gamma = 2.0
        self.Q_2 = self.gamma * identity(3)

        # Specify measurement output y
        self.y = SrpPhatLocalizer(arrgeom)


    def getC_n_x(self, position_estimate):
        """Get nonlinear measurement matrix"""

        x = position_estimate[0]
        y = position_estimate[1]
        z = position_estimate[2]

        if (x!=0.0):
            C_n_x = array([arctan(y/x),z/sqrt(x**2 + y**2), sqrt(x**2+y**2+z**2)])
        print "C_n_x: ", C_n_x
        return C_n_x
        
    
    def getC_n(self, position_estimate):
        """Get linearized measurement matrix"""

        #x = position_estimate[0]
        #y = position_estimate[1]
        #z = position_estimate[2]

        #C_n = array([[-y/(x**2+y**2), x/(x**2+y**2), 0.0],
        #             [-2*x*z/((x*2+y**2+z**2)*sqrt(x**2+y**2)),-2*y*z/((x**2+y**2+z**2)*sqrt(x**2+y**2)),(x**2+y**2)/((x**2+y**2+z**2)*sqrt(x**2+y**2))],
        #             [x/sqrt(x**2+y**2+z**2), y/sqrt(x**2+y**2+z**2), z/sqrt(x**2+y**2+z**2)]])
        #return C_n

        return self.C_n_l

    
    def getMatrixInverse(self, matrix):
        """ Return inverse of matrix by CholeskyDecomposition """

        lMatrix = cholesky_decomposition(matrix)
        inverseLMatrix = inverse(lMatrix)

        return matrixmultiply(transpose(inverseLMatrix), inverseLMatrix)
        
        
    def getKalmanGain(self):
        """ Calculate new KalmanGain """

        #C_n  = self.getC_n(self.currentStateEst)
        C_n_T = transpose(self.C_n_l)
        G_n   = matrixmultiply(matrixmultiply(self.K_n, C_n_T),self.getMatrixInverse(matrixmultiply(matrixmultiply(self.C_n_l,self.K_n),C_n_T)+self.Q_2))
                                  
        

        return G_n

    def getInnovationFactor(self, spectralSample, sampleNr):
        """ Calculate new InnovationFactor """
       
        y_n = self.y.getPostitionEstimation(spectralSample, sampleNr)
        #x_n = self.getC_n_x(self.oldStateEst)
        x_n = matrixmultiply(self.C_n_l, self.oldStateEst)

        print "Position y: ",y_n
        print "PositionEstimation x: ",x_n

        # No Innovation if SrpPhatLocalizer fails to localize speaker
        if (y_n[1]!=-10000):
            innovation = y_n - x_n
        else:
            innovation = array([0.0, 0.0, 0.0])

        print "Innovation: ", innovation
            
        return innovation
        

    def getCurrentStateEstimate(self, kalmanGain, innovation):
        """ Calculate StateEstimate at time n """

        #print "oldStateEst: ", self.oldStateEst
        
        self.currentStateEst = self.oldStateEst + matrixmultiply(kalmanGain,innovation)
        print "CurrentStateEst: " ,self.currentStateEst
        return self.currentStateEst


    def getFutureStateEstimate(self, currentState, kalmanGain, innovation):
        """ Calculate StateEstimate at time n+1 """

        self.futureStateEst = matrixmultiply(self.F_n, currentState)+matrixmultiply(kalmanGain, innovation)
        
        return self.futureStateEst
    

    def getCurrentErrorCorrelationMatrix(self, kalmanGain, currentStateEst):
        """ Calculate error correlation matrix at time n """

        self.K_n = self.K_n_old-matrixmultiply(kalmanGain, matrixmultiply(self.C_n_l,self.K_n_old))
        #print "CurrentErrorCorrelation: " ,self.K_n
        return self.K_n


    def getFuturErrorCorrelationMatrix(self):
        """ Calculate error correlation matrix at time n+1 """

        self.K_n_new = matrixmultiply(self.F_n, matrixmultiply(self.K_n,transpose(self.F_n))+self.Q_1)
        #print "FuturCorrelationMatrix: ", self.K_n_new
        return self.K_n_new


    def getNextIteration(self, spectralSample, sampleNr):
        """ Calls the different methods to do one IterationStep """
        
        # new time step...
        self.oldStateEst = self.futureStateEst
        self.K_n_old = self.K_n_new
        stateEst = []
        
        kalmanGain   = self.getKalmanGain()
        innovation   = self.getInnovationFactor(spectralSample, sampleNr)
        currentState = self.getCurrentStateEstimate(kalmanGain, innovation)

        stateEst.append(currentState)

        futureStateEstimate = self.getFutureStateEstimate(currentState, kalmanGain, innovation)

        stateEst.append(futureStateEstimate)
        
        print "NewStateEstimate: " ,futureStateEstimate

        
        
        self.getCurrentErrorCorrelationMatrix(kalmanGain, currentState)
        self.getFuturErrorCorrelationMatrix()

        return stateEst
        
    
def doSourceLocalization():
    """ Method to test the KalmanFilter """
    
    # Specify the geometry of the array
    arrgeom = array([[   0. , -307.5,    0. ],
                     [   0. , -266.5,    0. ],
                     [   0. , -225.5,    0. ],
                     [   0. , -184.5,    0. ],
                     [   0. , -143.5,    0. ],
                     [   0. , -102.5,    0. ],
                     [   0. ,  -61.5,    0. ],
                     [   0. ,  -20.5,    0. ]])
    

    # Specify paths for array data and database.
    inDataDir   = '/project/smartroom180/array/interference/'
    dataBaseDir = '/home/jmcd/BeamformingExpts/1000/'
    spkLst      = FGet('spk.DB2006.test')
    outDataDir = '/home/klee/experimente/Kalman/testOut/'
    nChan = len(arrgeom)
    fftLen      = 512
    nBlocks     = 4
    subSampRate = 2
    
    #db = dbase.DB200x('/home/jmcd/BeamformingExpts/1000/DB2006-spk', '/home/jmcd/BeamformingExpts/1000/DB2006-utt')
    
    # Build analysis chain
    analysisFBs = []
    for i in range(nChan):
        analysisFBs.append(AnalysisFB(fftLen, nBlocks, subSampRate))

        
    kalman = KalmanFilter(arrgeom)

            
    # Get SpectralSamples from next utterance for each Channel
    ttlSamplesIn = 0

    spectralPerChan = []
            
    for i in range(nChan):
        nextFile = ("%sif-talk.wav.%s" %(inDataDir, (i)))
        print nextFile
        soundSource = safeIO(lambda x: FileSoundSource(x, blkLen = fftLen), nextFile)
                
        if ttlSamplesIn == 0:
            ttlSamplesIn = soundSource._dataLen
        else:
            if ttlSamplesIn != soundSource._dataLen:
                print 'File %s contains %d samples instead of %d' %(nextFile, soundSource._dataLen, ttlSamplesIn)
        analysisFBs[i].nextUtt(soundSource)
                
        spectralSample = []
                
        for frame in analysisFBs[i]:
            spectralSample.append(frame)

        spectralPerChan.append(spectralSample)

    for m in range(len(spectralSample)):
        stateEstimates = kalman.getNextIteration(spectralPerChan, m)
            
    
doSourceLocalization()
