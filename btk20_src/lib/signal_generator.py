#!/usr/bin/python

#=================================================================#
# Author: Uwe Mayer                                               #
# Date:   2004-10-06                                              #
#                                                                 #
# Contains a collection of classes and functions to create        #
# basic signals for debugging and demonstration purposes          #
#=================================================================#

from numpy import *                   # sin, zeros, etc
from numpy.fft import  *
from types import *


#================================================================#
# Helper functions                                               #
#================================================================#
ComplexTypes = (ComplexType, Complex, Complex0, Complex8, Complex16, Complex32, Complex64)

def REAL(complex_struct):
    """returns the real part of all elements in the list or matrix"""
    real = lambda a: complex(a).real

    # test if is matrix
    if (type(complex_struct[0]) in [ArrayType, ListType]):
        matrix = []
        for row in range(len(complex_struct)):
            matrix.append(map(real, complex_struct[row]))
        return matrix
    # if is vector
    else:
        return map(real, complex_struct)


def IMAG(complex_struct):
    """returns the imaginary part of all elements in the list"""
    imag = lambda a: complex(a).imag

    # test if is matrix
    if (type(complex_struct[0]) in [ArrayType, ListType]):
        matrix = []
        for row in range(len(complex_struct)):
            matrix.append(map(imag, complex_struct[row]))
        return matrix
    # if is vector
    else:
        return map(imag, complex_struct)


def spectrum(complex_list):
    """calculates the spectrum of all elements in the list"""
    return abs(fft(complex_list,1024))


def logSpectrum(complex_list):
    """calculates the log-spectrum of all elements in the list"""
    spec = spectrum(complex_list)
    epsilon = max(spec)* 10**(-20)
    return log10(spec+epsilon)


#=========================================================#
# Sequence                                                #
#=========================================================#


class __Sequence:
    """number generator"""
    def __init__(self, start=0, stop=None, step=1, loop=False):
        """generator for creating (infinite) number ranges [start, stop)

        You can create infinite loops with this, so check our ranges and
        step value.

        Param: start    [default: %d]
               stop     [default: %s]
               step     [default: %d]
               loop     when reaching the last value wether to start from
                        the beginning
        """%(start, str(stop), step)
        assert(type(start) in [IntType, FloatType, LongType])
        assert(type(step) in [IntType, FloatType, LongType])

        self.__start = start
        self.__stop = stop
        self.__step = step
        self.__loop = loop
        self.__pos = self.__start

    def __iter__(self):
        while ((self.__stop == None) or (self.__pos < self.__stop)):
            yield self.__pos
            self.__pos += self.__step
            # restart if necessary
            if ((self.__pos >= self.__stop) and self.__loop):
                self.reset()
            
    def reset(self):
        self.__pos = self.__start
    

# abstract factroy for sequence objects
def Sequence(start=0, stop=None, step=1, loop=False):
    """returns a sequence iterator"""
    return iter(__Sequence(start, stop, step, loop))



#=====================================================================#
# FunctionFeature                                                     #
#=====================================================================#


class AbstractFeatureType:
    """base class for feature types

    Only declares interface for objects that return some kind of
    input feature. All methods raise a NotImplementedError().
    """
    def __init__(self):
        pass

    def __iter__(self):
        """returns an iterator"""
        raise NotImplementedError()

    def reset(self):
        """resets internal counters so that the iterator starts from the beginning again"""
        raise NotImplementedError()

    def size(self):
        """returns the size of generated features"""
        raise NotImplementedError()
    
    def getFeature(self):
        """returns a copy of the last feature returned by next() without modifying any counters"""
        raise NotImplementedError()
        

class FeatureAdapter(AbstractFeatureType):
    """convert input features to another type

    Takes a class which adheres to the FeatureStream protocoll
    and returns its data as NumPy array with a specific typecode.

    Param: inputSignal    forreign signal class
           typecode       NumPy typecode of returned data
    """
    def __init__(self, inputSignal, typecode=Float64):
        self.inputSignal = inputSignal
        self.inputIter = iter(inputSignal)
        self.typecode = typecode
        self.data = zeros(self.inputSignal.size())

    def __iter__(self):
        while (1):
            self.data = asarray(self.inputIter.next(), typecode=self.typecode)
            yield self.data

    def reset(self):
        self.inputSignal.reset()

    def size(self):
        return self.inputSignal.size()

    def getFeature(self):
        return self.data

        

class RawFileFeature:
    """read raw audio data in 16000 Hz, 16 signed

    Param: filename      path to input file
           windowLen     length of window to return
           windowShift   number of samples to shift window
    """
    def __init__(self, filename, windowLen, windowShift):
        self.__filename = filename
        self.__windowLen = windowLen
        self.__windowShift = windowShift
        self.__bps = 2
        self.data = []
        self.__f = open(filename)

    def __iter__(self):
        self.__getNextBlock()
        while (len(self.data) == self.__windowLen):
            yield self.data
            self.__getNextBlock()

        raise StopIteration

    def __getNextBlock(self):
        self.data = asarray(unpack("%dh"%self.__windowLen, self.__f.read(self.__windowLen*self.__bps)),
                            typecode=Int16)
        self.__f.seek(self.__bps*(self.__windowShift -self.__windowLen), 1)

    def getFeature(self):
        return self.data

    def reset(self):
        self.__f.seek(0)

    def size(self):
        return self.__windowLen



class FunctionFeature(AbstractFeatureType):
    """creates a generalised function feature

    Creates a sample feature of length <windowLen> by evaluating:

    y = a(x) *f(b(x)*x) +c(x)
    """
    def __init__(self, windowLen, a=lambda _: 1, f=lambda _: 0,
                 b=lambda _: 1, x=Sequence(), c=lambda _: 0, typecode=Int16):
        """class for function features

        The features have the form of:
        y(x) = a(x)* f(b(x)*x) +c(x)

        Param: windowLen  sample length; corresponds to the sampling
                          frequency of the signal f
               a          amplitude as a function of x [default: 1]
               f          function to be evaluated [default: 0]
               b          frequency as a function of x [default: 1]
               x          as a gernerator of a numerical type
                          [default: sequence of natural numbers, starting
                          from 0]
               c          y-axis interception as a function of x
                          [default: 0]
               typecode   datatype [default: Int16]           
        """
        AbstractFeatureType.__init__(self)

        assert(type(a) is FunctionType)
        assert(type(f) in [FunctionType, type(sin)])
        assert(type(b) is FunctionType)
        assert(type(x) is GeneratorType)
        assert(type(c) is FunctionType)
        
        self.__windowLen = windowLen        
        self.__a = a
        self.__f = f
        self.__b = b
        self.__x = x
        self.__c = c
        self.__typecode = typecode
        # current value of the feature
        self.__feature = zeros(self.__windowLen, self.__typecode)


    def __iter__(self):
        v = zeros(self.__windowLen, self.__typecode)
        self.__feature = v
        for t in self.__x:
            for i in xrange(self.__windowLen):
                v[i] = self.__a(t+i) *self.__f(1.0*self.__b(t+i)*(t+i)) +self.__c(t+i)
            yield v

    def reset(self): pass #self.__x.reset()   # reset WaveFeature
    def size(self): return self.__windowLen # return window length
    def getFeature(self): return self.__feature[:]



class BufferFeature(AbstractFeatureType):
    """simple 1:1 vector buffer"""
    def __init__(self, source):
        """initialise buffer feature

        Takes a generator type as input, buffers and returns the
        value 1:1 of the source.
        """
        AbstractFeatureType.__init__(self)        
        self.__source = source
        self.__value = [0]

    def __iter__(self):
        while (1):
            self.__value = self.__source.next()
            yield self.__value

    def getFeature(self):
        return self.__value

    def size(self):
        return self.__source.size()

    def reset(self):
        self.__value = zeros(self.size())
        self.__source.reset()



#==========================================================================#
# implementations of various usefull signals from signaltheory             #
#==========================================================================#


def WaveFeature(windowLen, amplitude=100, frequency=1,
                f=sin, x=Sequence(), typecode=Int16):
    """abstract factory for trigonometric feature vectors

    A WaveFeatures is a FunctionFeature for trigonometric functions.
    They are periodic in 2*PI, frequency and amplitude are constants
    and the sample length is normalized to the sample length (i.e at
    frequency 1 Hz they have one period over the sample feature length.
    i.e. default: 1*sin(2*pi*x)

    Param: windowLen    length of the feature vector
           amplitude    amplitude [default: 100]
           frequency    frequence [default: 1]
           f            function [default: sin]
           x            input time [default: Sequence()]
           typecode     datatype [default: Int16]
    """
    return FunctionFeature(windowLen,
                           a=lambda _: amplitude, 
                           f=lambda a: f(2*pi*a/windowLen),
                           b=lambda _: frequency,
                           x=x,
                           c=lambda _: 0,
                           typecode=typecode)



def ImpulseFeature(windowLen, amplitude=1, delta=0, x=Sequence(), typecode=Int16):
    """abstract factory for a unit impulse

    This returns a FunctionFeature for the unit impulse which is
    1 at x=0 and 0 everywhere else.
    This implementation is periodic. For a stationary signal set
    x to a constant.

    Param: windowLen    length of the feature vector
           amplitude    amplitude [default: 1]
           delta        shift: delta < 0 in positive x-direction
                        [default: 0]
           x            input time [default: Sequence()]
           typecode     datatype [default: Int16]
    """
    unitImpulse = lambda t: (t == 0) and 1 or 0
    
    return FunctionFeature(windowLen,
                           a=lambda _: amplitude,
                           f=lambda a: unitImpulse( (a+delta) %windowLen ),
                           b=lambda _: 1,
                           x=x,
                           c=lambda _: 0,
                           typecode=typecode)



def ImpulseTrainFeature(windowLen, amplitude=1, spaceing=1, delta=0, x=Sequence(),
                 typecode=Int16):
    """abstract factory for an impulse train

    This returns a FunctionFeature for an impulse train. This implementation
    is periodic. For a stationary signal set x to a constant.

    Param: windowLen     length of the feature vector
           amplitude     amplitude [default: 1]
           spaceing      number of samples between two impulses [default: 1]
           delta         shift: delta < 0 in positive x-direction
                         [default: 0]
           x             input time [default: Sequence()]
           typecode      datatype [default: Int16]
    """
    impulseTrain = lambda t,d: (int(t) %d == 0) and 1 or 0

    return FunctionFeature(windowLen,
                           a=lambda _: amplitude,
                           f=lambda a: impulseTrain( (a+delta)%windowLen, spaceing ),
                           b=lambda _: 1,
                           x=x,
                           c=lambda _: 0,
                           typecode=typecode)



def TriangleFeature(windowLen, spread=1.0, height=1.0, delta=0, x=Sequence(), typecode=Int16):
    """abstract factory for a triangular feature

    This returns a FunctinFeature for a triangular signal. This implementation
    is not periodic. For a stationary signal set x to a constant.

    Param: windowLen     length of the feature vector
           spread        the +/- intersection with the x-axis [default: 1.0]
           height        height [default: 1.0]
           delta         shift in x-direction [default: 0]
           x             input time [default: Sequence()]
           typecode      datatype [default: Int16]
    """
    triangle = lambda t: (-spread < t < 0) and (height/spread *t +height) or \
                         (0 <= t < spread) and (-height/spread *t +height) or 0
    
    return FunctionFeature(windowLen,
                           a=lambda _: 1,
                           f=lambda a: triangle( (a+delta) ),
                           b=lambda _: 1,
                           x=x,
                           c=lambda _: 0,
                           typecode=typecode )


def RectFeature(windowLen, width=6, height=1.0, delta=0, x=Sequence(), typecode=Int16):
    rect = lambda t: (-width/2 < t < width/2) and height or 0

    return FunctionFeature(windowLen,
                           a=lambda _: 1,
                           f=lambda a: rect( (a+delta)%windowLen ),
                           b=lambda _: 1,
                           x=x,
                           c=lambda _: 0,
                           typecode=typecode )



#-- test -----------------------------------------------------------------------
if (__name__ == "__main__"):
    from btk.multiplot import *         # import plot tools
    import Gnuplot                      # plot types

    WINLEN = 2**6                      # window length of the feature 

    #-- function feature test --------------------------------------------------
    fsinFeatureIter = iter(FunctionFeature(windowLen=WINLEN,
                                       a=lambda _: 200,
                                       f=lambda a: sin(2*pi*a /WINLEN),
                                       b=lambda _: 2,
                                       x=Sequence(),
                                       c=lambda _: 0))
    fsinFeature_plot = FeaturePlot(fsinFeatureIter, with='lines 1', title="function feature sin")


    #-- wave feature test ------------------------------------------------------
    sinFeatureIter = iter(WaveFeature(windowLen=WINLEN, f=cos))
    sinFeature_plot = FeaturePlot(sinFeatureIter, with='lines 2', title="wave feature sin")
    
    #-- exponential feature
    expFeatureIter = iter(WaveFeature(WINLEN, frequency=2.1,
                                      f=lambda a: exp(1j*a), typecode=Complex64))
    expFeatureRE_plot = FeaturePlot(None, with='lines 5', title="complex exponential RE")
    expFeatureIM_plot = FeaturePlot(None, with='lines 6', title="complex exponential IM")


    #-- impulse test -----------------------------------------------------------
    impFeatureIter = iter(ImpulseFeature(WINLEN, delta=-30))
    impFeature_plot = FeaturePlot(impFeatureIter, with='impulses', title="impulse")


    #-- impulse train test -----------------------------------------------------
    impTrainFeatureIter = iter(ImpulseTrainFeature(WINLEN, spaceing=5))
    impTrainFeature_plot = FeaturePlot(impTrainFeatureIter, with='impulses', title='impulse train')


    #-- triangle test ----------------------------------------------------------
    triangleFeatureIter = iter(TriangleFeature(WINLEN, spread=10.0, delta=-WINLEN/2.0, x=Sequence(step=0), typecode=Float16))
    triangleFeature_plot = FeaturePlot(triangleFeatureIter, with='lines 3', title='triangle')

    #-- triangle spectrum
    triangleSpectrum_plot = FeaturePlot(None, with='lines 3', title='triangle spectrum')
    

    #-- chirp signal -----------------------------------------------------------
    chirpFeature = FunctionFeature(WINLEN,
                                   a=lambda _: 1,
                                   f=lambda t: sin(2*pi*t /WINLEN),
                                   b=lambda a: 0.05*a,
                                   x=Sequence(),
                                   c=lambda a: 0,
                                   typecode=Float16)
    chirpFeatureIter = iter(chirpFeature)
    chirpFeature_plot = FeaturePlot(chirpFeatureIter, with='lines 4', title="chirp")


    #-- composition ------------------------------------------------------------
    compFeatureIter1 = iter(WaveFeature(WINLEN, frequency=2))
    compFeatureIter2 = iter(WaveFeature(WINLEN, frequency=3, f=cos))
    compFeature_plot = FeaturePlot(None, with='lines', title="2*sin(f1*x)+4*cos(f2*x)")
    

    G = Multiplot((5,2), persist=1)
    G('set grid')

    G[(0,0)] = sinFeature_plot, fsinFeature_plot
    G[(0,1)] = sinFeature_plot, fsinFeature_plot
    G[(1,0)] = expFeatureRE_plot, expFeatureIM_plot
    G[(1,1)] = expFeatureRE_plot, expFeatureIM_plot
    G[(2,0)] = impFeature_plot
    G[(2,1)] = impTrainFeature_plot
    G[(3,0)] = triangleFeature_plot
    G[(3,1)] = triangleSpectrum_plot
    G[(4,0)] = chirpFeature_plot
    G[(4,1)] = compFeature_plot


    for _ in range(10):
        fsinFeature_plot.update()
        sinFeature_plot.update()

        tmp = expFeatureIter.next()
        expFeatureRE_plot.update(REAL(tmp))
        expFeatureIM_plot.update(IMAG(tmp))

        impFeature_plot.update()
        impTrainFeature_plot.update()

        triangleData = triangleFeatureIter.next()
        triangleFeature_plot.update(triangleData)

        triangleSpectrum = spectrum(triangleData)
        triangleSpectrum_plot.update(triangleSpectrum)

        chirpFeature_plot.update()

        compFeature_plot.update( compFeatureIter1.next()+compFeatureIter2.next() )

        G.plot(0.3)
