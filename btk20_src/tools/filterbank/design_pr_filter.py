from btk.modulated import *
from pygsl import multiminimize
import pygsl.errors as errors
import os
from Numeric import *
#from btk.plot import *
from FFT import *

import pickle

M      = 4
#M      = 2048
m      = 8
N      = 2 * M * m
fs     = 1.0 / (2.0 * M)
design = CosineModulatedPrototypeDesign(M = M, N = N, fs = fs)
ndim   = N / 4

prototypeDir      = 'prototype.PR'
prototypeFileName = '%s/M=%d-m=%d.txt' %(prototypeDir,M,m)
prototypeMFile    = '%s/M=%d-m=%d.m'   %(prototypeDir,M,m)

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

sys    = multiminimize.gsl_multimin_function_fdf( fun, dfun, fdfun, design, ndim )
solver = multiminimize.conjugate_pr_eff( sys, ndim )
STEPSIZE  = 0.01
TOLERANCE = 1.0E-04
STOPTOLERANCE = 1.0E-03
solver.set(startpoint, STEPSIZE, TOLERANCE )

MinItns   = 10
MaxItns   = 100

for icnt in range(MaxItns):
    try:
        status1 = solver.iterate()
    except:
        print 'Minimization failed or stopped prematurely.'
        print msg
        break
    
    gradient = solver.gradient()
    #waAs = solver.getx()
    #mi   = solver.getf()
    status2 = multiminimize.test_gradient( gradient, STOPTOLERANCE )

    del gradient
    if status2 == 0:
        #print 'MI Converged %d %d %f' %(fbinX, itera,mi)
        break

h = design.proto()

print 'len(h) = ', len(h)
print 'N/2 = ', N/2

# Form the complete filter prototype
f = zeros(N, Float)
f[0:(N/2)] = h[::-1]
f[(N/2):]  = h

# Write the prototype to disk
if not os.path.exists(os.path.dirname(prototypeFileName)):
    os.makedirs(os.path.dirname(prototypeFileName))

fp = open(prototypeFileName, 'w')
pickle.dump(f, fp, 1)
fp.close()

fp = open(prototypeMFile, 'w')
for elem in f:
    fp.write('%e ' %elem)
fp.write('\n')
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
