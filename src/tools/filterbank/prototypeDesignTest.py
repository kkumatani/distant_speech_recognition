from btk.modulated import *
from pygsl.multimin import *
from pygsl._callback import *
from Numeric import *
#from btk.plot import *
from FFT import *

import pickle

# M      = 8
M      = 256
m      = 4
N      = 2 * M * m
fs     = 1.0 / (2.0 * M)
design = CosineModulatedPrototypeDesign(M = M, N = N, fs = fs)
ndim   = N / 4

prototypeFileName = './prototype_M'+str(M)+'_m'+str(m)

print 'ndim = ', ndim

startpoint = zeros(ndim, Float)
# startpoint[0]      = 1.0

def fun(x, design):
    return design_f(x, design)

def dfun(x, design):
    # print x
    grad = zeros(ndim, Float)
    design_df(x, design, grad)
    return grad

def fdfun(x, design):
    f    = design_f(x, design)
    grad = zeros(ndim, Float)
    design_df(x, design, grad)
    return (f, grad)

minfunc    = gsl_multimin_function_init_fdf((fun, dfun, fdfun, design, ndim))
minim      = gsl_multimin_fdfminimizer_alloc(cvar.gsl_multimin_fdfminimizer_conjugate_pr, ndim)
gsl_multimin_fdfminimizer_set(minim, minfunc, startpoint, 1e-05, 1e-05)

MinItns   = 10
MaxItns   = 100
Tolerance = 1.0E-04

curval = 1.0E+30
for icnt in range(MaxItns):
    try:
        gsl_multimin_fdfminimizer_iterate(minim)
    except:
        print 'Minimization failed or stopped prematurely.'
        break

    oldval = curval
    curval = gsl_multimin_fdfminimizer_minimum(minim)
    print 'step ', icnt, ' : ', curval
    if icnt >= MinItns and (oldval - curval) /  (oldval + curval) < Tolerance:
        break

h = design.proto()

print 'len(h) = ', len(h)
print 'N/2 = ', N/2

# Form the complete filter prototype
f = zeros(N, Float)
f[0:(N/2)] = h[::-1]
f[(N/2):]  = h

# Write the prototype to disk
fp = open(prototypeFileName, 'w')
pickle.dump(f, fp, 1)
fp.close()

# # Plot its power spectrum
# a = fft(f, 1024)
# print f
# print curval
# plotSpectrum(a)

# # Check the power complementarity of 'G_k(z)' and 'G_{k+M}(z)'
# k       = 2
# gk      = array([f[k],f[k+2*M],f[k+4*M]])
# gMplusk = array([f[k+M],f[k+3*M],f[k+5*M]])
# sk      = fft(gk, 128)
# sMplusk = fft(gMplusk, 128)
# spec = conjugate(sk) * sk + conjugate(sMplusk) * sMplusk
# plotSpectrum(spec)
