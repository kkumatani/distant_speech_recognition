/**
 * @file lpc.h
 * @brief Speech recognition front end.
 * @author John McDonough, Matthias Woelfel, Kenichi Kumatani
 */

#include <gsl/gsl_vector.h>
#include "feature/feature.h"

#ifndef LPC_H
#define LPC_H

// ----- definition for class `BaseFeature' -----
//
class BaseFeature {
protected:
  BaseFeature(unsigned order, unsigned dim);
  virtual ~BaseFeature();

  void fftPower(float* power);
  //virtual void autoCorrelation(const float* X, float* LP, float* E, float warp ) = 0;

  unsigned					_order;
  unsigned					_dim;

private:
  unsigned					_log2Length;

  unsigned 					_npoints;
  unsigned					_npoints2;
  double* 					_temp;
};


// ----- definition for class `WarpFeature' -----
//
class WarpFeature : protected BaseFeature {
protected:
  WarpFeature(unsigned order, unsigned dim);
  virtual ~WarpFeature();

  void autoCorrelation(const float* X, float* LP, float* E, float warp );

private:
  float*            				_K;
  float*            				_R;
  float*            				_WX;
  float*            				_WXTEMP;
  gsl_matrix_float* 				_tmpA;
};


// ----- definition for class `BurgFeature' -----
//
class BurgFeature : protected BaseFeature {
protected:
  BurgFeature(unsigned order, unsigned dim);
  virtual ~BurgFeature();

  void autoCorrelation(const float* X, float* LP, float* E, float warp );

private:
  float*					_EF;
  float*					_EB;
  float*					_EFP;
  float*					_EBP;
  float*					_A_flip;
  float*					_K;
};


// ----- definition for class `MVDRFeature' -----
//
template <class AutoCovariance>
class MVDRFeature : public VectorFeatureStream, private AutoCovariance {
 public:
  MVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, const String& nm = "MVDR");
  virtual ~MVDRFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { _src->reset(); VectorFeatureStream::reset(); }

private:
  VectorFloatFeatureStreamPtr			_src;
  unsigned					_temp_order;
  unsigned					_correlate;
  float						_warp;
  float*					_tmpR;
  float*					_tmpA;
  float*					_tmpPC;
  float*					_tmpPA;
  float*					_E;
};


// ----- methods for class `MVDRFeature' -----
//
template <class AutoCovariance>
MVDRFeature<AutoCovariance>::
MVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm)
  : VectorFeatureStream(src->size()/2+1, nm), AutoCovariance(order, src->size()),
    _src(src), _temp_order(2*order+1),
    _correlate(correlate), _warp(warp),
    _tmpR(new float[order+1]), _tmpA(new float[order+1]),
    _tmpPC(new float[_temp_order]), _tmpPA(new float[AutoCovariance::_dim+1]),
    _E(new float[order+1])
{
  if (_correlate < 10) _correlate = AutoCovariance::_dim;
  if (AutoCovariance::_order >= AutoCovariance::_dim/2+1)
    throw jparameter_error("Order (%d) and dimension (%d) do not match.", AutoCovariance::_order, AutoCovariance::_dim/2+1);
}

template <class AutoCovariance>
MVDRFeature<AutoCovariance>::~MVDRFeature()
{
  delete[] _tmpR;  delete[] _tmpA;  delete[] _tmpPC;  delete[] _tmpPA;  delete[] _E;
}

template <class AutoCovariance>
const gsl_vector* MVDRFeature<AutoCovariance>::MVDRFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = _src->next(frame_no_ + 1);
  increment_();

  const float* X  = block->data;
  float* A  = _tmpA;
  float* PC = _tmpPC;
  float* PA = _tmpPA;

  AutoCovariance::autoCorrelation(X, _tmpA, _E, _warp);

  for (int i = 0; i <= AutoCovariance::_order; i++) {
    double temp = 0;
    for (int ii = 0; ii <= (AutoCovariance::_order-i); ii++)
      temp += (float)(int(AutoCovariance::_order)+1-i-2*ii)*A[ii]*A[ii+i];
    if (_E[0] > 0) {
      PC[AutoCovariance::_order+i] = -temp;
      // instead of -temp/E[order] in the paper by B. Musicus IEEE 1985
    } else {
      PC[AutoCovariance::_order+i] = 10000000;
    }
  }

  for (int i = 1; i <= AutoCovariance::_order; i++) PC[AutoCovariance::_order-i] = PC[AutoCovariance::_order+i];

  PA[0] = 0;
  for (unsigned i = 1; i <= _temp_order; i++)
    PA[i] = PC[i-1];  // [..-1] because of fft

  for (unsigned i = _temp_order+1; i <= AutoCovariance::_dim; i++)
    PA[i] = 0;

  /* ------------------------------------------------------------------------
     Calculate MVDR Power
     (also called Capon Max. Likelihood Method)
  */

  AutoCovariance::fftPower(PA);

  for (unsigned i = 0; i<= AutoCovariance::_dim/2; i++) {
    double temp = sqrt(PA[i]);
    if (temp > 0) {
      temp = _E[0]/temp;
    } else {
      temp = 10000000;
    }

    gsl_vector_set(vector_, i, temp);
  }

  return vector_;
}

typedef MVDRFeature<WarpFeature> WarpMVDRFeature;
typedef Inherit<WarpMVDRFeature, VectorFeatureStreamPtr> WarpMVDRFeaturePtr;

typedef MVDRFeature<BurgFeature> BurgMVDRFeature;
typedef Inherit<BurgMVDRFeature, VectorFeatureStreamPtr> BurgMVDRFeaturePtr;

// ----- definition for class `WarpedTwiceFeature' -----
//
class WarpedTwiceFeature : protected BaseFeature {
protected:
  WarpedTwiceFeature(unsigned order, unsigned dim );
  virtual ~WarpedTwiceFeature();

  void autoCorrelation(const float* X, float* LP, float* E, float warp, float rewarp=0.0 );

private:
  float*     _K;
  float*     _R;
  float*     _WX;
  float*     _WXTEMP;
  float**    _tmpA;
};

// ----- definition for class `WarpedTwiceMVDRFeature' -----
//
class WarpedTwiceMVDRFeature : public VectorFeatureStream, private WarpedTwiceFeature {
public:
  WarpedTwiceMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, bool warpFactorFixed=false, float sensibility = 0.1, const String& nm = "WTMVDR");
  virtual ~WarpedTwiceMVDRFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { _src->reset(); VectorFeatureStream::reset(); }

private:
  VectorFloatFeatureStreamPtr			_src;
  unsigned					_correlate;
  unsigned					_temp_order;
  float						_warp;
  float*					_tmpR;
  float*					_tmpA;
  float*					_tmpPC;
  float*					_tmpPA;
  float*					_E;
  float                                         _rewarp;
  bool                                          _warpFactorFixed;
  float                                         _sensibility;
};

typedef Inherit<WarpedTwiceMVDRFeature,VectorFeatureStreamPtr> WarpedTwiceMVDRFeaturePtr;

// ----- definition for class `LPCFeature' -----
//
template <class AutoCovariance>
class LPCFeature : public VectorFeatureStream, private AutoCovariance {
 public:
  LPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm = "LPC");
  virtual ~LPCFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { _src->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  unsigned					_correlate;
  float						_warp;
  float*					_tmpA;
  float*					_tmpPA;
  float*					_E;
  const String					_lpmethod;
};


// ----- methods for class `LPCFeature' -----
//
template <class AutoCovariance>
LPCFeature<AutoCovariance>::
LPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm)
  : VectorFeatureStream(src->size()/2+1, nm),
    AutoCovariance(order, src->size()),
    _src(src),
    _correlate(correlate),
    _warp(warp),
    _tmpA(new float[order+1]),
    _tmpPA(new float[AutoCovariance::_dim+1]),
    _E(new float[order+1])
{
  if (_correlate < 10) _correlate = AutoCovariance::_dim;
  if (AutoCovariance::_order >= AutoCovariance::_dim/2+1)
    throw jparameter_error("Order (%d) and dimension (%d) do not match.", AutoCovariance::_order, AutoCovariance::_dim/2+1);
}

template <class AutoCovariance>
LPCFeature<AutoCovariance>::~LPCFeature()
{
  delete[] _tmpA;  delete[] _tmpPA;  delete[] _E;
}

template <class AutoCovariance>
const gsl_vector* LPCFeature<AutoCovariance>::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = _src->next(frame_no_ + 1);
  increment_();

  const float* X = block->data;
  float* A  = _tmpA;
  float* PA = _tmpPA;

  AutoCovariance::autoCorrelation(X, _tmpA, _E, _warp);

  PA[0] = 0;
  for (unsigned i = 1; i <= AutoCovariance::_order+1; i++)
    PA[i] = A[i-1];  // [..-1] because of fft

  for (unsigned i = AutoCovariance::_order+2; i <= AutoCovariance::_dim; i++)
    PA[i] = 0;

  AutoCovariance::fftPower(PA);

  for (unsigned i = 0; i <= AutoCovariance::_dim/2; i++) {
    double temp = PA[i];
    if (temp > 0) {
      temp = (2*_E[0])/(temp*AutoCovariance::_dim);
    } else {
      temp = 10000000;
    }
    gsl_vector_set(vector_, i, temp);
  }

  return vector_;
}

typedef LPCFeature<WarpFeature> WarpLPCFeature;
typedef Inherit<WarpLPCFeature, VectorFeatureStreamPtr> WarpLPCFeaturePtr;

typedef LPCFeature<BurgFeature> BurgLPCFeature;
typedef Inherit<BurgLPCFeature, VectorFeatureStreamPtr> BurgLPCFeaturePtr;


// ----- definition for class `SpectralSmoothing' -----
//
class SpectralSmoothing : public VectorFeatureStream {
 public:
  SpectralSmoothing(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom,
		    const String& nm = "Spectral Smoothing");
  virtual ~SpectralSmoothing();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { _adjustTo->reset(); _adjustFrom->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFeatureStreamPtr			_adjustTo;
  VectorFeatureStreamPtr			_adjustFrom;
  float*					_R;
};

typedef Inherit<SpectralSmoothing, VectorFeatureStreamPtr> SpectralSmoothingPtr;

#endif

