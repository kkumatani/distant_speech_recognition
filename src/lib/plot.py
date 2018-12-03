# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.plot
#   Purpose: Plotting of signals and spectra.
#   Author:  Fabian Jakobs

import Gnuplot
from Numeric import *

def plotData(data, cfrom=0, cto=-1, title=None):
    """
    This plots the contents of the NumPy array data from cfrom to cto
    using Gnuplot. 
    """
    if cto == -1:
        cto = len(data)
    if title == None:
        title = "unknown data (%s, %s)" % (cfrom, cto)
    g.clear()
    dat = Gnuplot.Data(arange(cfrom, cto),
                       data[cfrom:cto])
    dat.set_option(with="lines")
    dat.set_option(title=title)
    g.plot(dat)

def plotCorrelation(delays, correlation, title='Correlation Function'):
    """
    Plot time delays vs. cross-correlation. 
    """
    g.clear()
    dat = Gnuplot.Data(delays, correlation)
    dat.set_option(with="lines")
    dat.set_option(title=title)
    # g('set yrange [0:0.5]')
    g.plot(dat)

def plotSpectrum(data, cfrom=0, cto=-1, title=None):
    """
    Plot a log-spectrum. 
    """
    if cto == -1:
        cto = len(data)
    if title == None:
        title = "Spectrum (omega / pi)"
    g.clear()
    dat = Gnuplot.Data(arange(cfrom, cto) * 2.0 / cto,
                       20*log10(abs(data[cfrom:cto]) + 0.1) - 20*log10(abs(data[0])))
    dat.set_option(with="lines")
    dat.set_option(title=title)
    g.plot(dat)

def plotBeamPattern(wH):
    """
    Plot a beam pattern. 
    """
    nPoint  = 1000
    nPoint2 = nPoint / 2
    nChan   = len(wH)
    nChan2  = (nChan - 1) / 2.0
    vk = zeros(nChan).astype(complex)
    pd = zeros(nPoint).astype(complex)
    x  = zeros(nPoint).astype(float)
    J = (0+1j)
    for i in range(nPoint):
        x[i] = 1.0*(i - nPoint2)/nPoint2
        for n in range(nChan):
            vk[n] = exp(-pi*J*(n-nChan2)*x[i])
        pd[i] = innerproduct(wH,vk)

    g.title('Uniform Array Beam Pattern')
    g.xlabel('psi / pi')
    g.ylabel('Magnitude (dB)')
    g('set xrange [-1:1]')
    g('set yrange [-30:5]')
    g('set xtics -1,0.25,1')
    g('set ytics -30,5,5')
    g('set grid')

    logMagnitude = 20.* log10(abs(pd))
    dat = Gnuplot.Data(x, logMagnitude)
    dat.set_option(with="lines")
    g.plot(dat)

g = Gnuplot.Gnuplot(debug=0)

def plotBeamPattern2(wH1, wH2):
    """
    Plot two beam patterns simultaneously. 
    """
    nPoint  = 1000
    nPoint2 = nPoint / 2
    nChan   = len(wH1)
    nChan2  = (nChan - 1) / 2.0
    vk = zeros(nChan).astype(complex)
    pd1 = zeros(nPoint).astype(complex)
    pd2 = zeros(nPoint).astype(complex)
    x  = zeros(nPoint).astype(float)
    J = (0+1j)
    for i in range(nPoint):
        x[i] = 1.0*(i - nPoint2)/nPoint2
        for n in range(nChan):
            vk[n] = exp(-pi*J*(n-nChan2)*x[i])
        pd1[i] = innerproduct(wH1,vk)
        pd2[i] = innerproduct(wH2,vk)

    g.title('Uniform Array Beam Pattern')
    g.xlabel('psi / pi')
    g.ylabel('Magnitude (dB)')
    g('set xrange [-1:1]')
    g('set yrange [-30:5]')
    g('set xtics -1,0.25,1')
    g('set ytics -30,5,5')
    g('set grid')

    logMagnitude1 = 20.* log10(abs(pd1))
    logMagnitude2 = 20.* log10(abs(pd2))
    dat1 = Gnuplot.Data(x, logMagnitude1, title='D&S',      with='lines')
    dat2 = Gnuplot.Data(x, logMagnitude2, title='Adaptive', with='lines')
    g.plot(dat1, dat2)

g = Gnuplot.Gnuplot(debug=0)

