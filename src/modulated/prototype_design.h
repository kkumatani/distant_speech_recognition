/*
  File Name: grad.c
  Last Modification Date:	3/9/94	10:03:45
  Current Version: grad.c	1.7
  %File Creation Date: 26/10/91
  Author: Ramesh Gopinath  <ramesh@dsp.rice.edu>

  Copyright: All software, documentation, and related files in this distribution
  are Copyright (c) 1993  Rice University

  Permission is granted for use and non-profit distribution providing that this
  notice be clearly maintained. The right to distribute any portion for profit
  or as part of any commercial product is specifically reserved for the author.
*/

/*
#################################################################################
grad.mex4:
function [g,f,h] = grad(k,M,N,fs); Computes the gradient of stopband energy of the
                   linear-phase prototype filter in a cosine-modulated filter bank
		   with respect to lattice parameters of the J lattices.

      Input:
            k : Vector of denormalized lattice parameters for the J lattices of
	        CMFB
            M : M-channel, M-band etc
            N : 2Mm, length of the prototype filter
           fs : Stopband edge (as a fraction of pi)
      Output: 
            f : Stopband Energy
            h : Prototype filter (second half)
#################################################################################
*/

#ifndef PROTOTYPE_DESIGN_H
#define PROTOTYPE_DESIGN_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include "common/refcount.h"
#include "common/jexception.h"


// ----- definition for class `CosineModulatedPrototypeDesign' -----
//
class CosineModulatedPrototypeDesign {
 public:
  CosineModulatedPrototypeDesign(int M = 256, int N = 3072, double fs = 1.0);
  ~CosineModulatedPrototypeDesign();

  void fcn(const double* x, double* f);
  void grad(const double* x, double* g);

  int M() const { return _M; }
  int N() const { return _N; }
  int m() const { return _m; }
  int J() const { return _J; }

  // return (one-half) of the prototype filter
  const gsl_vector* proto();

 private:
  const int					_M;	// No. of channels
  const int					_N;	// filter length 2 * _M * _m
  const int					_M2;	// 2 * _M
  const int					_m;
  const int					_Mm;
  const int					_J;
  const int					_Jm;
  const int					_oddM;
  const int					_oddm;

  double*					_sinews;
  double*					_h;
  double*					_hh;
  double*					_Ph;
  double*					_jac;

  int*						_index;

  gsl_vector*					_proto;
};

double design_f(const gsl_vector* v, void* params);
void   design_df(const gsl_vector* v, void* params, gsl_vector* df);
void   design_fdf(const gsl_vector* v, void* params, double* f, gsl_vector* df);


// ----- definition for class `PrototypeDesignBase' -----
//
class PrototypeDesignBase {
public:
  PrototypeDesignBase(int M, int m, unsigned r, double wp, int tau);
  ~PrototypeDesignBase();

protected:
  virtual void _calculateAb();
  void _calculateC();
  void _svd(gsl_matrix* U, gsl_matrix* V, gsl_vector* S, gsl_vector* workSpace);
  gsl_matrix* _nullSpace(const gsl_matrix* A, double tolerance);
  gsl_vector* _pseudoInverse(const gsl_matrix* A, const gsl_vector* b, double tolerance);
  void _solveNonSingular(const gsl_matrix* H, const gsl_vector* c0, const gsl_matrix* P, double tolerance);
  void _solveSingular(gsl_matrix* Ac, gsl_vector* bc, gsl_matrix* K, gsl_vector* dp, double tolerance);
  double _condition(gsl_matrix* H, gsl_matrix* P);
  void _printMatrix(const gsl_matrix* A, FILE* fp = stdout) const;
  void _printVector(const gsl_vector* b, FILE* fp = stdout) const;

  void save(const String& fileName);

  const int					_M;	// the number of subbands
  const int					_m;
  const	double					_wp;
  const int					_R;
  const int					_D;
  const int					_L;
  	int					_tau;

  gsl_matrix*					_A;
  gsl_vector*					_b;
  gsl_matrix*					_cpA;
  gsl_vector*					_cpb;
  gsl_matrix*					_C;
  gsl_matrix*					_cpC;

  gsl_vector*					_singularVals;
  gsl_vector*					_scratch;
  gsl_vector*					_workSpace;

  gsl_vector*					_prototype;
};


// ----- definition for class `AnalysisOversampledDFTDesign' -----
//
typedef Countable AnalysisOversampledDFTDesignCountable;
class AnalysisOversampledDFTDesign : public AnalysisOversampledDFTDesignCountable, protected PrototypeDesignBase {
public:
  AnalysisOversampledDFTDesign(int M = 512, int m = 2, int r = 1, double wp = 1.0, int tau_h = -1 );
  ~AnalysisOversampledDFTDesign();

  // design analysis prototype
  const gsl_vector* design(double tolerance = 1.0E-07);

  // calculate distortion measures
  const gsl_vector* calcError(bool doPrint = true);

  void save(const String& fileName) { PrototypeDesignBase::save(fileName); }

protected:
  virtual void _solve(double tolerance);
  double _passbandResponseError();
  double _inbandAliasingDistortion();

  gsl_vector*					_error;	// [0] the total response error
                                                        // [1] the residual aliasing distortion
                                                        // [2] objective fuction
};

typedef refcountable_ptr<AnalysisOversampledDFTDesign> AnalysisOversampledDFTDesignPtr;


// ----- definition for class `SynthesisOversampledDFTDesign' -----
//
typedef Countable SynthesisOversampledDFTDesignCountable;
class SynthesisOversampledDFTDesign : public SynthesisOversampledDFTDesignCountable, protected PrototypeDesignBase {
public:
  SynthesisOversampledDFTDesign(const gsl_vector* h, int M = 512, int m = 2, int r = 1,
				double v = 0.01, double wp = 1.0, int tau_T = -1 );
  ~SynthesisOversampledDFTDesign();

  // design synthesis prototype
  const gsl_vector* design(double tolerance = 1.0E-07);

  // calculate distortion measures
  virtual const gsl_vector* calcError(bool doPrint = true);

  void save(const String& fileName) { PrototypeDesignBase::save(fileName); } 

protected:
  void _calculateEfP();
  virtual void _solve(double tolerance);
  double _totalResponseError();
  double _residualAliasingDistortion();

        gsl_vector*				_h;
  const double					_v;

  gsl_vector*					_singularVals;
  gsl_vector*					_scratch;
  gsl_vector*					_workSpace;

  gsl_matrix*					_E;
  gsl_matrix*					_P;
  gsl_vector*					_f;

  gsl_matrix*					_cpE;
  gsl_matrix*					_cpP;
  gsl_vector*					_cpf;
  gsl_vector*					_error;	// [0] the total response error
                                                        // [1] the residual aliasing distortion
                                                        // [2] objective fuction
};

typedef refcountable_ptr<SynthesisOversampledDFTDesign> SynthesisOversampledDFTDesignPtr;


// ----- definition for class `AnalysisNyquistMDesign' -----
//
class AnalysisNyquistMDesign : public AnalysisOversampledDFTDesign {
public:
  AnalysisNyquistMDesign(int M = 512, int m = 2, int r = 1, double wp = 1.0, int tau_h = -1 );
  ~AnalysisNyquistMDesign();

private:
  virtual void _solve(double tolerance);
  void _calculateFd();
  void _calculateKdp();

  gsl_matrix*					_F;
  gsl_vector*					_d;
  gsl_matrix*					_K;
  gsl_vector*					_dp;
};

typedef Inherit<AnalysisNyquistMDesign, AnalysisOversampledDFTDesignPtr>	AnalysisNyquistMDesignPtr;


// ----- definition for class `SynthesisNyquistMDesign' -----
//
class SynthesisNyquistMDesign : public SynthesisOversampledDFTDesign {
public:
  SynthesisNyquistMDesign(const gsl_vector* h, int M = 512, int m = 2, int r = 1,
			  double wp = 1.0, int tau_g = -1);
  ~SynthesisNyquistMDesign();

  virtual const gsl_vector* calcError(bool doPrint = true);

private:
  virtual void _solve(double tolerance);

  double _passbandResponseError();
  void _calculateHc0();
  void _calculateJcp();

  gsl_matrix*					_H;
  gsl_vector*					_c0;
  gsl_matrix*					_J;
  gsl_vector*					_cp;
};

typedef Inherit<SynthesisNyquistMDesign, SynthesisOversampledDFTDesignPtr> SynthesisNyquistMDesignPtr;


// ----- definition for class `SynthesisNyquistMDesignCompositeResponse' -----
//
class SynthesisNyquistMDesignCompositeResponse : public SynthesisNyquistMDesign {
public:
  SynthesisNyquistMDesignCompositeResponse(const gsl_vector* h, int M = 512, int m = 2, int r = 1,
					   double wp = 1.0, int tau = -1);
  ~SynthesisNyquistMDesignCompositeResponse();

private:
  virtual void _calculateAb();
  inline double _sinc(int m);

  double* 					_sincValues;
};

typedef Inherit<SynthesisNyquistMDesignCompositeResponse, SynthesisNyquistMDesignPtr> SynthesisNyquistMDesignCompositeResponsePtr;

double SynthesisNyquistMDesignCompositeResponse::_sinc(int m)
{
  int am     = abs(m);

  if (am >= 2 * _L)
    throw jindex_error("Problem: index %d > 2 * L = %d", am, 2 * _L);

  double val = _sincValues[am];
  if (val != -HUGE) return val;

  if (m == 0)
    val = _sincValues[am] = 1.0;
  else
    val = _sincValues[am] = sin(_wp * am) / (_wp * am);

  return val;
}

#endif
