/**
 * @file lms.cc
 * @brief Implementation of LMS algorithms.
 * @author John McDonough
 */

#include <gsl/gsl_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "lms/lms.h"
#include "matrix/gslmatrix.h"
#include <gsl/gsl_blas.h>


// ----- methods for class `FastBlockLMSFeature' -----
//
FastBlockLMSFeature::
FastBlockLMSFeature(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha, float gamma, const String& nm)
  : VectorFloatFeatureStream(samp->size() / 2, nm),
    _desired(desired), _samp(samp),
    _overlapSave(new OverlapSave(samp)),
    _N(samp->size()),
    _M(samp->size() / 2),
    _alpha(alpha),
    _gamma(gamma),
    _e(new double[_N]),
    _u(new double[_N]),
    _phi(new double[_N]),
    _U(gsl_vector_complex_alloc(_N)),
    _E(gsl_vector_complex_alloc(_N)),
    _Phi(gsl_vector_complex_alloc(_N)),
    _D(gsl_vector_alloc(_N))
{
  gsl_vector_set_zero(_D);
}

FastBlockLMSFeature::~FastBlockLMSFeature()
{
  delete[] _e;  delete[] _u;  delete[] _phi;

  gsl_vector_complex_free(_U);
  gsl_vector_complex_free(_E);
  gsl_vector_complex_free(_Phi);

  gsl_vector_free(_D);
}

const gsl_vector_float* FastBlockLMSFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = _overlapSave->next(frame_no_);
  gsl_vector_float_memcpy(vector_, block);

  increment_();
  return vector_;
}

void FastBlockLMSFeature::update()
{
  // form 'E'
  const gsl_vector_float* d = _desired->next(frame_no_);
  const gsl_vector_float* y = _overlapSave->next(frame_no_);
  for (unsigned i = 0; i < _M; i++)
    _e[i] = 0.0;
  for (unsigned i = _M; i < _N; i++)
    _e[i] = gsl_vector_float_get(d, i - _M) - gsl_vector_float_get(y, i - _M);
  gsl_fft_real_radix2_transform(_e, /*stride=*/ 1, _N);
  halfComplexUnpack_(_E, _e);

  // form 'U'
  const gsl_vector_float* u = _samp->next(frame_no_);
  for (unsigned i = 0; i < _N; i++)
    _u[i] = gsl_vector_float_get(u, i);
  gsl_fft_real_radix2_transform(_u, /*stride=*/ 1, _N);
  halfComplexUnpack_(_U, _u);

  // form 'E x U^H'
  for (unsigned i = 0; i < _N; i++) {
    gsl_complex val = gsl_complex_mul(gsl_vector_complex_get(_E, i), gsl_complex_conjugate(gsl_vector_complex_get(_U, i)));
    val = gsl_complex_mul_real(val, gsl_vector_get(_D, i));
    gsl_vector_complex_set(_E, i, val);
  }

  // update 'D'
  for (unsigned i = 0; i < _N; i++)
    gsl_vector_set(_D, i, _gamma * gsl_vector_get(_D, i) + (1.0 - _gamma) * gsl_complex_abs2(gsl_vector_complex_get(_U, i)));

  // apply gradient constraint
  halfComplexPack_(_phi, _E);
  gsl_fft_halfcomplex_radix2_inverse(_phi, /* stride= */ 1, _N);
  for (unsigned i = _M; i < _N; i++)
    _phi[i] = 0.0;
  gsl_fft_real_radix2_transform(_phi, /*stride=*/ 1, _N);
  halfComplexUnpack_(_Phi, _phi);

  // perform update
  for (unsigned i = 0; i < _N; i++)
    gsl_vector_complex_set(_Phi, i, gsl_complex_mul_real(gsl_vector_complex_get(_Phi, i), _alpha));
  _overlapSave->update(_Phi);
}

void FastBlockLMSFeature::halfComplexUnpack_(gsl_vector_complex* tgt, const double* src)
{
  for (unsigned m = 0; m <= _M; m++) {
    if (m == 0 || m == _M) {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], 0.0));
    } else {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], src[_N-m]));
    }
  }
}

void FastBlockLMSFeature::halfComplexPack_(double*  tgt, const gsl_vector_complex* src)
{
  unsigned len  = src->size;
  unsigned len2 = (len+1) / 2;

  gsl_complex entry = gsl_vector_complex_get(src, 0);
  tgt[0]    = GSL_REAL(entry);

  for (unsigned m = 1; m < len2; m++) {
    entry      = gsl_vector_complex_get(src, m);
    tgt[m]     = GSL_REAL(entry);
    tgt[len-m] = GSL_IMAG(entry);
  }

  if ((len & 1) == 0) {
    entry     = gsl_vector_complex_get(src, len2);
    tgt[len2] = GSL_REAL(entry);
  }
}

