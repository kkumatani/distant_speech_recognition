#!/usr/bin/python
"""
Design the de Haan filter prototypes.

Reference:
Jan Mark De Haan, Nedelko Grbic, Ingvar Claesson, Sven E Nordholm, "Filter bank design for subband adaptive microphone arrays", IEEE TSAP 2003.

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""
import os.path
import pickle
import sys
import numpy

from btk20.modulated import *

def design_de_haan_analysis_filter( M, m, r, wpW):
    """
    Generate an analysis filter prototype

    :param M: number of subbands
    :type M: integer
    :param m: Filter length factor
    :type m: integer
    :param r: exponential decimation factor
    :type r: integer
    :param wpW: cut-off frequency factor
    :type wpW: float
    :returns: coefficients of analysis filter prototype
    """
    analysis = AnalysisOversampledDFTDesignPtr(M = M,  m = m, r = r, wp = wpW)
    h = analysis.design()
    (alpha, beta, sum_alpha_beta) = analysis.calcError()

    return h


def design_de_haan_synthesis_filter(h, M, m, r, v, wpW):
    """
    Create a synthesis prototype, given the analysis filter

    :param h: Analysis filter prototype
    :type h: vector
    :param M: Number of subbands
    :type M: integer
    :param m: Filter length factor
    :type m: integer
    :param r: exponential decimation factor
    :type r: integer
    :param v: Weight for the total response error and residual aliasing distortion
    :type v: float
    :param wpW: Cut-off frequency factors (< M)
    :type wpW: float
    :returns: Coefficients of synthesis filter prototype
    """
    synthesis = SynthesisOversampledDFTDesignPtr(h, M = M, m = m, r = r, v = v, wp = wpW)
    g = synthesis.design()
    (gamma, epsir, sum_gamma_epsir) = synthesis.calcError()

    return g


def main( M, m, r, v, wpW, outputdir):

    D    = M / 2**r # frame shift
    print('M=%d m=%d r=%d D=%d v=%f wp=pi/%d*M' %(M, m, r, D, v, wpW))

    h = design_de_haan_analysis_filter( M, m, r, wpW)
    analysis_path = '%s/h-M=%d-m=%d-r=%d-v=%0.4f-w=%0.2f.pickle' %(outputdir, M, m, r, v, wpW)
    with open(analysis_path, 'w') as fp:
        pickle.dump(h, fp)

    g = design_de_haan_synthesis_filter(h, M, m, r, v, wpW)
    synthesis_path = '%s/g-M=%d-m=%d-r=%d-v=%0.4f-w=%0.2f.pickle' %(outputdir, M, m, r, v, wpW)
    with open(synthesis_path, 'w') as fp:
        pickle.dump(g, fp)


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description='generate coefficients of the de Haan filter prototypes.')
    parser.add_argument('-M', dest='M',
                        default=256, type=int,
                        help='no. of subbands')
    parser.add_argument('-m', dest='m',
                        default=4, type=int,
                        help='Prototype filter length factor')
    parser.add_argument('-r', dest='r',
                        default=1, type=int,
                        help='Decimation factor')
    parser.add_argument('-w', dest='wpW',
                        default=1.0, type=float,
                        help='cut-off frequency factor')
    parser.add_argument('-v', dest='v',
                        default=100.0, type=float,
                        help='Weight factor for the total response error and residual aliasing distortion')
    parser.add_argument('-o', dest='outputdir',
                        default='./prototype.dh',
                        help='output directory name (default: .)')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    main(args.M, args.m, args.r, args.v, args.wpW, args.outputdir)
