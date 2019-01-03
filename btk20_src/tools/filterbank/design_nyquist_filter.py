#!/usr/bin/python
"""
Design the Nyquist(M) filter prototypes.

Reference:
Kenichi Kumatani, John McDonough, Stefan Schacht, Dietrich Klakow, Philip N Garner, Weifeng Li, "Filter bank design based on minimization of individual aliasing terms for minimum mutual information subband adaptive beamforming", ICASSP 2018.

.. moduleauthor:: Kenichi Kumatani <k_kumatani@ieee.org>
"""
import numpy
import numpy.linalg
import os
import pickle

def mynull(A, num=0, datatype='d'):
    """
    Find a null space projectoin matrix

    :param A: matrix
    :type A: numpy matrix
    :param num: number of bases for the null space
    :type num: integer
    :param datatype: 'd' or 's' for tolereance
    :type datatype: char
    :returns: null space projection matrix of A and singular weights
    """

    [U,W,VH] = numpy.linalg.svd(A)
    V = VH.transpose()
    (rowN, colN) = A.shape
    if num > 0:
        sX = colN - num
        val  = numpy.zeros(num, numpy.float)
    else:
        if rowN > 1:
            s = numpy.diag(W)
        elif rowN == 1:
            s = numpy.array([[W[0]]])

        if datatype == 'd': # double precision accuracy
            tol = max(rowN, colN) * s.max() * 2.2204e-16
        else: # single precision accuracy
            tol = max(rowN, colN) * s.max() * 1.1921e-07

        print('Threshold for nullspace: %e' %tol)
        sX = numpy.sum(s  > tol)
        val = numpy.zeros(colN-sX, numpy.float)

    y = numpy.array(V[:, sX:colN:1])
    for i in range(len(val)):
        val[i] = W[sX+i]

    return (y, val)


def design_Nyquist_analyasis_filter_prototype(M, m, D, wpW=1):
    """
    Design an analysis filter prototype

    :param M: Number of subbands
    :type M: integer
    :param m: Filter length factor
    :type m: integer
    :param D: Decimation factor
    :type D: integer
    :returns: Coefficients of analysis filter prototype and inband aliasing distortion
    """
    L_h   = M * m   # length of the prototype filter
    md    = L_h / 2 if m != 1 else 0 # group delay offset
    tau_h = L_h / 2 # group delay of analysis fb
    w_p   = numpy.pi / (wpW * M)  # passband cut-off frequency

    A = numpy.zeros((L_h, L_h), numpy.float) # A is (L_h x L_h) hermitian matrix
    b = numpy.zeros((L_h, 1),   numpy.float)
    C = numpy.zeros((L_h, L_h), numpy.float)

    for i in range(L_h):
        for j in range(L_h):
            factor = float(D - 1) if ((j - i) % D) == 0 else -1.0

            if (j - i) == 0:
                C[i][j] = factor / D
            else:
                C[i][j] = factor * (numpy.sin(numpy.pi * (j - i) / D)) / (numpy.pi * (j - i))

            if (j - i) == 0:
                A[i][j] = 1.0
            else:
                A[i][j] = numpy.sin(w_p * (j - i)) / (w_p * (j - i))

        if (tau_h - i) == 0:
            b[i] = 1.0
        else:
            b[i] = numpy.sin(w_p * (tau_h - i)) / (w_p * (tau_h - i))

    # delete the rows and columns of C corresponding to the components of h = 0
    delC = numpy.zeros((L_h - m + 1, L_h - m + 1), numpy.float)
    delA = numpy.zeros((L_h - m + 1, L_h - m + 1), numpy.float)
    delb = numpy.zeros((L_h - m + 1, 1),           numpy.float)
    i = 0
    for k in range(L_h):
        if k == md or (k % M) != 0:
            j = 0
            for l in range(L_h):
                if l == md or (l % M) != 0:
                    delA[i][j] = A[k][l]
                    delC[i][j] = C[k][l]
                    j += 1

            delb[i] = b[k]
            i += 1

    rank_delC = numpy.linalg.matrix_rank(delC)
    if rank_delC == len(delC):
        # take an eigen vector corresponding to the smallest eigen value.
        eVal, eVec = numpy.linalg.eig(delC)
        # take eigen vectors as basis
        minX = numpy.argmin(eVal)
        print('nmin eigen val: {}'.format(eVal[minX]))
        rh = eVec[:,minX]; # eigen values are sorted in the ascending order.
        # flip the sign if all the coefficients are negative
        all_negative = True
        for val in rh:
            if val > 0:
                all_negative = False
        if all_negative:
            rh = - rh
    else:
        nulldelC, _w = mynull( delC )
        if len(nulldelC[0]) == 0:
            raise ArithmeticError('No. null space bases of is 0')
        print( 'No. null space bases of C is %d' %len(nulldelC[0]))
        # In general, null(delP) is not a square matrix.
        # We don't want to use a peseude inversion matrix as much as possible.
        T1    = numpy.dot(delA, nulldelC)
        T1_2  = numpy.dot(nulldelC.transpose(), T1)
        rank_T = numpy.linalg.matrix_rank(T1_2)
        if rank_T == len(T1_2):
            x = numpy.linalg.solve(T1_2, numpy.dot(nulldelC.transpose(), delb))
        else:
            print('Use pseudo-inverse matrix because %d < %d' %(rank_T, len(T1_2)))
            x = numpy.dot(linalg.pinv(T1), delb)
        rh = numpy.dot(nulldelC, x)

    # re-assemble the complete prototype
    h = numpy.zeros((L_h, 1), numpy.float)
    k = 0
    for m in range(L_h):
        if m != md and (m % M) == 0:
            h[m] = 0
        else:
            h[m] = rh[k]
            k += 1

    # Pass-band error: h' * A * h - 2 * h' * b + 1
    # alpha = numpy.dot(h.transpose(), numpy.dot(A, h)) - 2 * numpy.dot(h.transpose(), b) + 1
    # Inband aliasing distortion: h' * C * h
    beta  = numpy.dot(h.transpose(), numpy.dot(C, h))

    return (h, beta)


def design_Nyquist_synthesis_filter_prototype(h, M, m, D, wpW=1):
    """
    Design a synthesis filter prototype

    :param h: Analysis filter prototype
    :type h: 1 x Mm matrix
    :param M: Number of subbands
    :type M: integer
    :param m: Filter length factor
    :type m: integer
    :param D: Decimation factor
    :type D: integer
    :returns: Coefficients of synthesis filter prototype and residual aliasing distortion
    """
    L_h   = len(h)       # length of the analysis prototype filter
    L_g   = M * m        # length of the synthesis prototype filter
    L_max = max(L_g, L_h)
    md    = L_h / 2 if m != 1 else 0
    tau_g = L_g / 2 # group delay of synthesis fb
    tau_t = md + tau_g # total filterbank delay
    w_p   = numpy.pi / (wpW * M)  # cut-off frequency

    E = numpy.zeros((L_g, L_g), numpy.float)
    f = numpy.zeros((L_g, 1),   numpy.float)
    P = numpy.zeros((L_g, L_g), numpy.float)

    for i in range(L_g):
        for j in range(L_g):
            for k in range(0, 2*m+1):
                kM = k * M
                if (kM - i) >= 0 and (kM - j) >= 0 and (kM - i) < L_h and (kM - j) < L_h:
                    E[i][j] += h[kM-i][0] * h[kM-j][0]

            factor = D - 1 if ((i - j) % D) == 0 else -1
            for l in range(-L_max, L_max+1):
                if (l+i) >= 0 and (l+j) >= 0 and (l+i) < L_h and (l+j) < L_h:
                    P[i][j] += h[l+j][0] * h[l+i][0] * factor

        if (tau_t - i) >= 0 and (tau_t - i) < L_h:
            f[i] = h[tau_t-i][0]

    E = ((M * M) / float(D / D)) * E
    f = (M / (numpy.pi * D)) * f
    P = (M / float(D * D)) * P

    # Shift a time-reversed version of h and make a matrix.
    # The k-th row of the matrix indicates h_k
    rowN = 2 * m - 1;
    H  = numpy.zeros((rowN, L_g), numpy.float ) # a row vector corresponds to h_k in the report
    sX = M
    eX = sX - L_g + 1
    for i in range(rowN):
        s = sX;
        if s < 1:
            s = 1
        elif s > L_g:
            s = L_g
        e = eX;
        if e < 1:
            e = 1
        elif e > L_g:
            e = L_g
        H[i, e-1:s] = numpy.array([h[j-1, 0] for j in range(s,e-1,-1)])
        sX += M
        eX += M

    C0 = numpy.zeros((rowN, 1), numpy.float)
    C0[m-1][0] = D * 1.0 / M # C0(m) = h(md+1);

    sizeP = len(P)
    rank_P = numpy.linalg.matrix_rank(P);
    if rank_P == sizeP:
        print('Use Lagrange multiplier...')
        invP = numpy.linalg.inv( P );
        H_invP_HT = numpy.dot(numpy.dot(H, invP), H.transpose())
        g = numpy.dot(numpy.dot(numpy.dot(invP, H.transpose()), numpy.linalg.inv(H_invP_HT)), C0)
    elif rank_P <= (sizeP - rowN):
        print('Use the null space...')
        nullP, _w = mynull(P)
        print('No. null space bases of P is %d' %len(nullP[0]))
        y = numpy.dot(numpy.linalg.pinv(numpy.dot(H, nullP)), C0)
        g = numpy.dot(nullP, y)
    else:
        # will not find enough bases of the null space
        print('Use SVD (rank(P)=%d)...' %rank_P)
        [UP,WP,VP] = numpy.linalg.svd( P );
        pnullP = VP[:,(sizeP-rowN):sizeP]
        y = numpy.linalg.solve(numpy.dot(H, pnullP), C0)
        g = numpy.dot(pnullP, y)

    # Total response error: g' * E * g - 2 * g' * f + 1
    # gamma = numpy.dot(g.transpose(), numpy.dot(E, g)) - 2 * numpy.dot(g.transpose(), f) + 1
    # Residual aliasing distortion: g' * P * g
    epsir = numpy.dot(g.transpose(), numpy.dot(P, g))

    return (g, epsir)


def main(M, m, r, outputdir):
    D     = M // (2 ** r) # window shift
    if D == 0:
        D = 1

    (h, beta) = design_Nyquist_analyasis_filter_prototype(M, m, D)
    print('Inband aliasing error: %f dB' %(10*numpy.log(beta)))
    analysis_filename = os.path.join(outputdir, 'h-M%d-m%d-r%d.pickle' %(M, m, r))
    with open(analysis_filename, 'wb') as hfp:
        pickle.dump(h.flatten(), hfp, True)

    (g, epsir) = design_Nyquist_synthesis_filter_prototype(h, M, m, D)
    print('Residual aliasing distortion: %f dB' %(10*numpy.log(epsir)))
    synthesis_filename = os.path.join(outputdir, 'g-M%d-m%d-r%d.pickle' %(M, m, r))
    with open(synthesis_filename, 'wb') as gfp:
        pickle.dump(g.flatten(), gfp, True)

    coeff_filename = os.path.join(outputdir, 'M=%d-m=%d-r=%d.m' %(M, m, r))
    with open(coeff_filename, 'w') as cfp:
        for coeff in h:
            cfp.write('%e ' %(coeff))
        cfp.write('\n')
        for coeff in g:
            cfp.write('%e ' %(coeff))


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description='generate coefficients of the Nqyuist(M) filter prototypes.')
    parser.add_argument('-M', dest='M',
                        default=256, type=int,
                        help='no. of subbands')
    parser.add_argument('-m', dest='m',
                        default=4, type=int,
                        help='Prototype filter length factor')
    parser.add_argument('-r', dest='r',
                        default=1, type=int,
                        help='Decimation factor')
    parser.add_argument('-o', dest='outputdir',
                        default='./prototype.ny',
                        help='output directory name (default: .)')

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    main(args.M, args.m, args.r, args.outputdir)
