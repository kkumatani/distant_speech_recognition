"""
 @file tdoa.py
 @brief Calculation of TDOA features for various microphone configurations.
 @author John McDonough and Kenichi Kumatani
"""
from btk.feature import *
from numpy.fft import *
import numpy
import math

class MicrophonePairObservation:

    def __init__(self, pairX, firstMicX, secondMicX, observation):
        self.pairX 		= pairX
        self.firstMicX		= firstMicX
        self.secondMicX		= secondMicX
        self.observation	= observation


# Implement the phase transform for a given microphone pair
class PHATFeature:

    def __init__(self, source1, source2, fftLen):
        self.source1	= source1
        self.source2	= source2
        self.fftLen	= fftLen

    def next(self, frameX):
        block1		= self.source1.next(frameX)
        block2 		= self.source2.next(frameX)

        crossSpectrum    = block1 * numpy.conjugate(block2) / numpy.abs(block1 * numpy.conjugate(block2))
        crossCorrelation = ifft(crossSpectrum)

        return crossCorrelation


# Perform peak picking to find the TDOA from the (generalized) cross-correlation
class TDOAFeature:
    def __init__(self, src, fftLen, f_s):
        self.src	= src
        self.fftLen	= fftLen
        self.fftLen2	= fftLen / 2
        self.T_s	= 1.0 / f_s

    def next(self, frameX):
        block = self.src.next(frameX)

        highestPeak = 0.0
        for n in range(len(block)):
            height = abs(block[n])
            if height > highestPeak:
                if n < self.fftLen2:
                    delay	= float(n) * self.T_s
                else:
                    delay	= float(n - self.fftLen) * self.T_s

                # pack as [tau_d, peakHeight]
                bestPeak	= [delay, height]
                highestPeak	= height
                
        if highestPeak == 0.0:
            return None
        else:
            return bestPeak


class MicrophonePairSource:

    def __init__(self, pairX, firstMicX, secondMicX, source):
        # @param pairX[string list] a pair of mic indices (start from '1')
        # @param firstMicX[int] mic index (start from '0')
        # @param secondMicX[int] mic index (start from '0')
        # @param observation[obj] TDOAFeature()
        self.pairX 		= pairX
        self.firstMicX		= firstMicX
        self.secondMicX		= secondMicX
        self.source		= source


# Concatenate together those TDOAs whose cross-correlation peak is above a given threshold
class TDOAFeatureVector:
    def __init__(self, microphonePairs, microphonePositions, minimumPairs = 2, threshold = 0.2, c = 343000.0):
        self.microphonePairs		= microphonePairs
        self.microphonePositions	= microphonePositions
        self.minimumPairs		= minimumPairs
        self.threshold			= threshold
        self.c				= c

    # Calculate true time delays of arrival based on predicted state estimate
    def tdoa(self, micPair, x_cart):
        diff1 = x_cart - self.microphonePositions[micPair.firstMicX]
        diff2 = x_cart - self.microphonePositions[micPair.secondMicX]
        dist1 = sqrt(numpy.dot(diff1, diff1))
        dist2 = sqrt(numpy.dot(diff2, diff2))
        return (dist1 - dist2) / self.c


    # Calculate the linearized time delay
    def calcLinearizedObservation(self, xk_predict, H, observation):
        # print 'Calculating linearized observation ...'

        yk_len = len(observation)
        yk = numpy.zeros(len(observation), numpy.float)
        n = 0
        for micPair in observation:
            yk[n] = observation[n].observation - (self.tdoa(micPair, xk_predict) - numpy.dot(H[n,:], xk_predict))

            n += 1

        return yk


    # Linearize the observation functional about xk_predict
    def linearize(self, xk_predict, observation):
        # print 'Linearizing observation functional ...'

        H = numpy.zeros([len(observation), len(xk_predict)], numpy.float)
        rowX = 0
        for obs in observation:
            m1 = self.microphonePositions[obs.firstMicX]
            m2 = self.microphonePositions[obs.secondMicX]
            diff1 = xk_predict - m1
            diff2 = xk_predict - m2

            D1 = sqrt(numpy.dot(diff1, diff1))
            D2 = sqrt(numpy.dot(diff2, diff2))

            H[rowX,:] = ((diff1 / D1) - (diff2 / D2)) / self.c
            rowX += 1

        return H


    def next(self, frameX):

        # Determine which microphone pairs had a correlation peak above the threshold
        observation	= []
        self.tdoabuf = {}
        for micPair in self.microphonePairs:

            obs = micPair.source.next(frameX) # [delay, height(cc)]
            #print 'TDOA' , obs
            self.tdoabuf['%d %d' %(micPair.firstMicX, micPair.secondMicX)] = obs[0]
            if obs[1] > self.threshold:
                micPairObs = MicrophonePairObservation(micPair.pairX, micPair.firstMicX, micPair.secondMicX, obs[0])
                observation.append(micPairObs)

        if len(observation) < self.minimumPairs:
            return None
        else:
            return observation

    def getInstantaneousPosition(self,frameX):

        return

    def getTDOAs(self):
        return self.tdoabuf


# Concatenate together those TDOAs whose cross-correlation peak is above a given threshold
# in the case of the linear microphone array under the far-field assumption
class FarfieldLinearArrayTDOAFeatureVector:
    def __init__(self, microphonePairs, microphonePositions, minimumPairs = 2, threshold = 0.3, c = 343000.0):
        self.microphonePairs		= microphonePairs
        self.microphonePositions	= microphonePositions # [nMicrophones][position value]
        self.minimumPairs		= minimumPairs
        self.threshold			= threshold
        self.c				= c

        if len(self.microphonePositions[0]) > 1 :
            print 'You have to set sensor positions on the axis'
            raise

    # Calculate true time delays of arrival based on predicted state estimate
    def tdoa(self, micPair, azimuth):
        dist	= abs( self.microphonePositions[micPair.secondMicX] - self.microphonePositions[micPair.firstMicX] )
        tau	= dist * numpy.cos(azimuth) / self.c
        return numpy.array([tau],numpy.float)


    # Calculate the linearized time delay
    def calcLinearizedObservation(self, azimuthk_predict, H, observation):
        # print 'Calculating linearized observation ...'

        yk_len	= len(observation)
        yk	= numpy.zeros(len(observation), numpy.float)
        n	= 0
        for micPair in observation:
            yk[n] = observation[n].observation - (self.tdoa(micPair, azimuthk_predict) - numpy.inner(H[n,:], azimuthk_predict))

            n += 1

        return yk


    # Linearize the observation functional about azimuthk_predict
    def linearize(self, azimuthk_predict, observation):
        # print 'Linearizing observation functional ...'

        H = numpy.zeros([len(observation), len(azimuthk_predict)], numpy.float)
        rowX = 0
        for obs in observation:
            dist	= abs( self.microphonePositions[obs.secondMicX] - self.microphonePositions[obs.firstMicX] )
            grad	= - dist * numpy.sin(azimuthk_predict) / self.c
            H[rowX,:]	= numpy.array([grad], numpy.float)
            rowX       += 1

        return H


    def next(self, frameX):

        # Determine which microphone pairs had a correlation peak above the threshold
        observation	= []
        self.tdoabuf	= {}
        for micPair in self.microphonePairs:

            obs = micPair.source.next(frameX) # [delay, height(cc)]
            dist = abs( self.microphonePositions[micPair.firstMicX] - self.microphonePositions[micPair.secondMicX] )            
            #print 'TDOA' , obs , numpy.arccos( obs[0] * self.c / dist  ) * 180 /numpy.pi 
            self.tdoabuf['%d %d' %(micPair.firstMicX, micPair.secondMicX)] = obs[0]
            if obs[1] > self.threshold:
                micPairObs = MicrophonePairObservation(micPair.pairX, micPair.firstMicX, micPair.secondMicX, obs[0])
                observation.append(micPairObs)

        if len(observation) < self.minimumPairs:
            return None
        else:
            return observation

    def getInstantaneousPosition(self, frameX):
        
        aveAzimuth = 0.0
        nSamples = 0
        areAllOverThreshold = True
        for micPair in self.microphonePairs:
            obs = micPair.source.next(frameX) # [delay, height(cc)]
            dist = abs( self.microphonePositions[micPair.firstMicX] - self.microphonePositions[micPair.secondMicX] )
            if obs[1] > self.threshold:
                val = obs[0] * self.c / dist
                if val >= -1 and val <= 1:
                    azimuth = numpy.arccos( val ) 
                    aveAzimuth += azimuth
                    nSamples += 1
                else:
                    #print 'Invalid time delay %e btw. mic %d and %d (cc=%0.4f)' %(val,micPair.firstMicX,micPair.secondMicX,obs[1])
                    areAllOverThreshold = False
            else:
                areAllOverThreshold = False

        if nSamples and areAllOverThreshold == True:
            aveAzimuth = aveAzimuth / nSamples
            return aveAzimuth
        #print 'Could not find a source'
        return -1e10

    def getTDOAs(self):
        return self.tdoabuf


# Concatenate together those TDOAs whose cross-correlation peak is above a given threshold
# for the case of a circular microphone array under a far-field assumption
class FarfieldCircularArrayTDOAFeatureVector:

    def __init__(self, microphonePairs, microphonePositions, minimumPairs = 2, threshold = 0.3, c = 343000.0):
        self.microphonePairs		= microphonePairs
        self.microphonePositions	= microphonePositions # [nMicrophones][position value]
        self.minimumPairs		= minimumPairs
        self.threshold			= threshold
        self.c				= c

        # if len(self.microphonePositions[0]) > 1 :
        #     print 'You need to set the sensor positions'
        #     raise

    # Calculate true time delays of arrival based on predicted state estimate
    def tdoa(self, micPair, polarX):
        theta	= polarX[0]	# polar angle
        phi	= polarX[1]	# azimuth
        offset	= self.microphonePositions[micPair.secondMicX] - self.microphonePositions[micPair.firstMicX]
        u	= numpy.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        tau	= numpy.dot(u, offset) / self.c
        return numpy.array([tau], numpy.float)


    # Calculate the linearized time delay
    def calcLinearizedObservation(self, polarX, H, observation):
        # print 'Calculating linearized observation ...'

        yk_len	= len(observation)
        yk	= numpy.zeros(len(observation), numpy.float)
        n	= 0
        for micPair in observation:
            yk[n] = observation[n].observation - (self.tdoa(micPair, polarX) - numpy.dot(H[n,:], polarX))

            n += 1

        return yk


    # Linearize the TDOA functional about polarX
    def linearize(self, polarX, observation):
        # print 'Linearizing observation functional ...'

        H = numpy.zeros([len(observation), len(polarX)], numpy.float)
        rowX = 0
        for obs in observation:
            theta	= polarX[0]	# polar angle
            phi		= polarX[1]	# azimuth
            offset	= self.microphonePositions[obs.secondMicX] - self.microphonePositions[obs.firstMicX]
            du_dtheta	= numpy.array([numpy.cos(theta) * numpy.cos(phi), numpy.cos(theta) * numpy.sin(phi), - numpy.sin(theta)])
            du_dphi	= numpy.array([- numpy.sin(theta) * numpy.sin(phi), numpy.sin(theta) * numpy.cos(phi), 0.0])
            dtau_dtheta	= numpy.dot(du_dtheta, offset) / self.c
            dtau_dphi	= numpy.dot(du_dphi,   offset) / self.c
            H[rowX,:]	= numpy.array([dtau_dtheta, dtau_dphi])
            rowX       += 1

        return H


    def next(self, frameX):

        # Determine which microphone pairs had a correlation peak above the threshold
        observation	= []
        self.tdoabuf	= {}
        for micPair in self.microphonePairs:

            obs = micPair.source.next(frameX) # [delay, height(cc)]
            dist = abs( self.microphonePositions[micPair.firstMicX] - self.microphonePositions[micPair.secondMicX] )            
            #print 'TDOA' , obs , numpy.arccos( obs[0] * self.c / dist  ) * 180 /numpy.pi 
            self.tdoabuf['%d %d' %(micPair.firstMicX, micPair.secondMicX)] = obs[0]
            if obs[1] > self.threshold:
                micPairObs = MicrophonePairObservation(micPair.pairX, micPair.firstMicX, micPair.secondMicX, obs[0])
                observation.append(micPairObs)

        if len(observation) < self.minimumPairs:
            return None
        else:
            return observation

    def getTDOAs(self):
        return self.tdoabuf

