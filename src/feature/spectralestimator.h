/**
 * @file spectralestimator.h
 * @brief Speech recognition front end.
 * @author John McDonough, Matthias Woelfel, Kenichi Kumatani
 */

#ifndef SPECTRALESTIMATOR_H
#define SPECTRALESTIMATOR_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector_complex.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <algorithm>

#include "matrix/gslmatrix.h"
#include "stream/stream.h"
#include "common/mlist.h"

#include <stdio.h>
#include <iostream>

/**
   @brief calculate the auto-correlation vector, r, with the time-discrete signal, x[n].
          Each component, r[k], represents the auto-correlation function of the input sample for lag, k.
          Accordingly, r[k] can be expressed as \cE{ x[n] x[n-k] }.
   @param const gsl_vector *samples[in] time-discrete signal which corresponds to x[n].
   @param gsl_vector *autoCorrelationVector[out] auto-correlation vector which corresponds to r. 
   @note  the size of "autoCorrelationVector" should be "order+1" in order to calculate the LP model of the order, "order".
 */
void calcAutoCorrelationVector( const gsl_vector_float *samples, gsl_vector *autoCorrelationVector );

/**
   @brief calculate the linear prediction coefficients (LPC) with the Levinson-Durbin Algorithm.
   @param gsl_vector_float *autoCorrelationVector[in] 
   @param gsl_vector_float *lpcs[out] lpcs[0,...,p-1] where p is the order of the model.
   @return prediction error
   @notice The estimated signal is defined as 
           \hat{x}[n] = sum_{k=1}^p lpcs[k-1] * x[n-k].
           That is different from the definition in MATLAB lpc()
	   where it is \hat{x}(n) = 1.0 - sum_{k=1}^p lpcs(k+1) * x(n-k).
 */
float calcLPCswithLevinsonDurbin( gsl_vector_float *autoCorrelationVector, gsl_vector_float *lpcs );

// ----- definition for class `LPCSpectrumEstimator' -----
//

/**
   @class calculate LPCs with the Levinson-Durbin algorithm, and estimate the LPC envelope.
   @note  You can feed samples to this object with calcLPCs( const gsl_vector_float* samples ), 
	  and the LPCs are then computed.
 */
class LPCSpectrumEstimator : public VectorFloatFeatureStream {
public:
  LPCSpectrumEstimator( const VectorFloatFeatureStreamPtr& samp, unsigned order, unsigned fftLen, const String& nm = "LPCSpectrumEstimator");
  ~LPCSpectrumEstimator();

  const gsl_vector_float* next(int frame_no = -5);
  void reset();

  /**
     @brief get the auto-correlation vector, r[0,...,order-1], 
            where r[k] represents the auto-correlation function of the input sample for lag, k+1.
     @note you have to call calcLPCs() before you call this method.
   */
  const gsl_vector_float* getAutoCorrelationVector(){
    return (const gsl_vector_float*)_r;
  }
  /**
     @brief get the prediction error of the LP model. 
     @note you have to call calcLPCs() before you call this method.
   */
  float getPredictionError(){
    return _e;
  }
  /**
     @brief get the LPCs.
     @note you have to call calcLPCs() before you call this method.
   */
  const gsl_vector_float* getLPCs(){
    return _lpcs; //_vector;
  }

  gsl_vector_complex* getFTLPCs(){
    return _ftlpcs;
  }

  const gsl_vector_float* calcLPCs( const gsl_vector_float* samples );
  void calcLPEnvelope();

private:
  void complexPackedArrayUnpack(const double* halfcomplex_coefficient, gsl_vector_complex* complex_coefficient) const;
private:
  static const double             _epsilon;
  VectorFloatFeatureStreamPtr        _samp;
  unsigned                          _order; /* the LP model of the order */
  unsigned                         _fftLen;
  unsigned                        _fftLen2;
  float                                 _e; /* prediction error */
  gsl_vector_float*                     _r; /* _order x 1 auto-correlation vector _r[1,...,p]; See [1] */
  gsl_vector*             _autoCorrelation; /* (_order+1) x 1 auto-correlation vector, where the first element corresponds to the power */
  gsl_vector_float*                  _lpcs; /* LPCs */
  double*                          _wslpcs; /* workspace for the LPCs */
  gsl_vector_complex*              _ftlpcs;
};

typedef Inherit<LPCSpectrumEstimator, VectorFloatFeatureStreamPtr> LPCSpectrumEstimatorPtr;


// ----- definition for class `CepstralSpectrumEstimator' -----
//
class CepstralSpectrumEstimator : public VectorFloatFeatureStream {
public:
  CepstralSpectrumEstimator(const VectorComplexFeatureStreamPtr& source, unsigned order = 14, unsigned fftLen = 256,
			    double logPadding = 1.0, const String& nm = "CepstralSpectrumEstimator");
  ~CepstralSpectrumEstimator();

  const gsl_vector_float* next(int frame_no = -5);
  void reset() { _source->reset();  VectorFloatFeatureStream::reset(); }

private:
  VectorComplexFeatureStreamPtr			_source;
  const unsigned				_order;
  const unsigned				_fftLen;
  const double					_logPadding;

  double*					_samples;
  gsl_vector_complex*				_spectrum;
  gsl_vector_complex*				_cepstrum;
};

typedef Inherit<CepstralSpectrumEstimator, VectorFloatFeatureStreamPtr> CepstralSpectrumEstimatorPtr;


// ----- definition for class `SEMNB' -----
//

/**
   @class 
 */

class SEMNB : public Countable {
public:
  SEMNB( unsigned order, unsigned fftLen, const String& nm = "SEMNB");
  ~SEMNB();

  const gsl_vector*	calcDerivativeOfDeviation( LPCSpectrumEstimatorPtr &lpcSEPtr );
  virtual void		reset();
  const gsl_vector*	getLPEnvelope() const { return _lpEnvelope; }

private:
  void			_calcEigen();
  void			_calcLambda_bar();

  double		_dLambdak_dPm(unsigned k, unsigned m);
  void			_dLambdabar_dPm(unsigned m);
  void			_da_dPm(unsigned m);
  float			_dSigma_dP(const gsl_vector_float* lpcs, double epsilonp, unsigned m);

  double		_dSp_dan(const gsl_vector_float* lpcs, unsigned m, unsigned n);
  double		_dSp_dPm(const gsl_vector_float* lpcs, unsigned m);
  double		_depsilonp_dPm(const gsl_vector_float* lpcs, unsigned m);
  void			_calc_dr_dPm(unsigned m);
  double		_dPhat_dP(const gsl_vector_float* lpcs, double epsilonp, unsigned m);


  const unsigned				_order;
  const unsigned				_fftLen;
  const unsigned				_fftLen2;

  gsl_vector*					_derivative;
  gsl_matrix*					_R;
  gsl_matrix*					_Rcopy;
  gsl_eigen_symmv_workspace*			_workSpace;
  gsl_vector*					_evalues;
  gsl_matrix*					_evectors;
  gsl_vector*	                                _lpEnvelope;
  gsl_vector_complex*                           _ftlpcs;
  
  gsl_vector*					_r;
  gsl_vector*					_dr_dPm;
  gsl_vector*					_dak_dPm;
  gsl_matrix*					_Lambda_bar;
  gsl_matrix*					_dLambdamatrix;

  gsl_matrix*					_C;
  gsl_matrix*					_D;
};

typedef refcountable_ptr<SEMNB> SEMNBPtr;

#endif

