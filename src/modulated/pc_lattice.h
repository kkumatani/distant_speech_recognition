/*
  File Name: PC_lattice.c
  Last Modification Date:	3/9/94	10:15:31
  Current Version: PC_lattice.c	1.3
  File Creation Date: Fri Aug 27 19:37:04 1993
  Author: Ramesh Gopinath  <ramesh@dsp.rice.edu>

  Copyright: All software, documentation, and related files in this distribution
  are Copyright (c) 1993  Rice University

  Permission is granted for use and non-profit distribution providing that this
  notice be clearly maintained. The right to distribute any portion for profit
  or as part of any commercial product is specifically reserved for the author.
*/


/* This is a library of programs for power complementary lattice manipulations:
There are three functions viz., pclat, dpclat and ddpclat */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PC_LATTICE_H
#define PC_LATTICE_H

void Pclat(double* h0, double* h1, const double* k, int k_ord);
void Dpclat(double* h0, double* h1, const double* k, int k_ord, int iii);
void Ddpclat(double* h0, double* h1, const double* k, int k_ord,
             int iii, int jjj);

#endif // _PC_lattice_h_

#ifdef __cplusplus
}
#endif
