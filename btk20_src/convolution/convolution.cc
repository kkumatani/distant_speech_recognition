/*
 * @file convolution.cc
 * @brief Block convolution realization of an LTI system with the FFT.Time delay estimation
 * @author John McDonough
 */

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "common/jpython_error.h"
#include "convolution/convolution.h"

#ifdef HAVE_CONFIG_H
#include <btk.h>
#endif
#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif


// ----- methods for class `OverlapAdd' -----
//
OverlapAdd::OverlapAdd(VectorFloatFeatureStreamPtr& samp,
                       const gsl_vector* impulseResponse, unsigned fftLen, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp),
    L_(samp->size()), P_(impulseResponse->size), N_(check_fftLen_(L_, P_, fftLen)), N2_(N_/2),
    section_(new double[N_]), frequencyResponse_(gsl_vector_complex_alloc(N2_+1)),
    buffer_(gsl_vector_float_alloc(L_+P_-1))
{
  set_impulse_response_(impulseResponse);
}

OverlapAdd::~OverlapAdd()
{
  delete[] section_;
  gsl_vector_complex_free(frequencyResponse_);
  gsl_vector_float_free(buffer_);
}

void OverlapAdd::set_impulse_response_(const gsl_vector* impulseResponse)
{
  if (impulseResponse == NULL) {
    gsl_vector_complex_set_zero(frequencyResponse_);
    return;
  }

  for (unsigned i = 0; i < N_; i++)
    section_[i] = 0.0;
  for (unsigned i = 0; i < P_; i++)
    section_[i] = gsl_vector_get(impulseResponse, i);
  gsl_fft_real_radix2_transform(section_, /* stride= */ 1, N_);
  unpack_half_complex(frequencyResponse_, N2_, section_, N_);

  for (unsigned i = 0; i < N_; i++)
    section_[i] = 0.0;
  gsl_vector_float_set_zero(buffer_);
}

// check consistency of FFT length
unsigned OverlapAdd::check_fftLen_(unsigned sectionLen, unsigned irLen, unsigned fftLen)
{
  if (fftLen == 0) {

    fftLen = 1;
    while (fftLen < sectionLen + irLen - 1)
      fftLen *= 2;

    return fftLen;

  } else {

    if (fftLen < sectionLen + irLen - 1)
      throw jdimension_error("Section (%d) and impulse response (%d) lengths inconsistent with FFT length (%d).",
			     sectionLen, irLen, fftLen);
    return fftLen;

  }
}

const gsl_vector_float* OverlapAdd::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);

  assert(block->size == L_);

  // forward FFT on new data
  for (unsigned i = 0; i < N_; i++)
    section_[i] = 0.0;
  for (unsigned i = 0; i < L_; i++)
    section_[i] = gsl_vector_float_get(block, i);
  gsl_fft_real_radix2_transform(section_, /*stride=*/ 1, N_);

  // multiply with frequency response
  for (unsigned i = 0; i <= N2_; i++) {
    if (i == 0 || i == N2_) {
      section_[i] = section_[i] * GSL_REAL(gsl_vector_complex_get(frequencyResponse_, i));
    } else {
      gsl_complex val = gsl_complex_mul(gsl_complex_rect(section_[i], section_[N_-i]), gsl_vector_complex_get(frequencyResponse_, i));
      section_[i]     = GSL_REAL(val);
      section_[N_-i]  = GSL_IMAG(val);
    }
  }

  // inverse FFT
  gsl_fft_halfcomplex_radix2_inverse(section_, /* stride= */ 1, N_);

  // add contribution of new section to buffer
  for (unsigned i = 0; i < L_ + P_ - 1; i++)
    gsl_vector_float_set(buffer_, i, gsl_vector_float_get(buffer_, i) + section_[i]);

  // copy section length 'L' from buffer onto output
  for (unsigned i = 0; i < L_; i++)
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(buffer_, i));

  // shift down buffer
  for (unsigned i = 0; i < P_ - 1; i++)
    gsl_vector_float_set(buffer_, i, gsl_vector_float_get(buffer_, i + L_));
  for (unsigned i = P_ - 1; i < L_ + P_ - 1; i++)
    gsl_vector_float_set(buffer_, i, 0.0);

  increment_();
  return vector_;
}

void OverlapAdd::reset()
{
  samp_->reset();  VectorFloatFeatureStream::reset();

  for (unsigned i = 0; i < L_ + P_ - 1; i++)
    gsl_vector_float_set(buffer_, i, 0.0);
}


// ----- methods for class `OverlapSave' -----
//
OverlapSave::OverlapSave(VectorFloatFeatureStreamPtr& samp,
			 const gsl_vector* impulseResponse, const String& nm)
  : VectorFloatFeatureStream(check_output_size_(impulseResponse->size, samp->size()), nm), samp_(samp),
    L_(check_L_(impulseResponse->size, samp->size())), L2_(L_/2), P_(impulseResponse->size),
    section_(new double[L_]), frequencyResponse_(gsl_vector_complex_alloc(L_/2+1))
{
  set_impulse_response_(impulseResponse);
}

OverlapSave::~OverlapSave()
{
  delete[] section_;
  gsl_vector_complex_free(frequencyResponse_);
}

void OverlapSave::set_impulse_response_(const gsl_vector* impulseResponse)
{
  if (impulseResponse == NULL) {
    gsl_vector_complex_set_zero(frequencyResponse_);
    return;
  }

  for (unsigned i = 0; i < L_; i++)
    section_[i] = 0.0;

  for (unsigned i = 0; i < P_; i++)
    section_[i] = gsl_vector_get(impulseResponse, i);
  gsl_fft_real_radix2_transform(section_, /* stride= */ 1, L_);
  unpack_half_complex(frequencyResponse_, L2_, section_, L_);

  for (unsigned i = 0; i < L_; i++)
    section_[i] = 0.0;
}

unsigned OverlapSave::check_output_size_(unsigned irLen, unsigned sampLen)
{
  if (irLen >= sampLen)
    throw jdimension_error("Cannot have P = %d and L = %d", irLen, sampLen);

  return (sampLen - irLen);
}

// check consistency of FFT length
unsigned OverlapSave::check_L_(unsigned irLen, unsigned sampLen)
{
  // should check that _L is a power of 2
  if (irLen >= sampLen)
    throw jdimension_error("Cannot have P = %d and L = %d", irLen, sampLen);

  return sampLen;
}

const gsl_vector_float* OverlapSave::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);

  // forward FFT on new data
  for (unsigned i = 0; i < L_; i++)
    section_[i] = gsl_vector_float_get(block, i);
  gsl_fft_real_radix2_transform(section_, /*stride=*/ 1, L_);

  // multiply with frequency response
  for (unsigned i = 0; i <= L2_; i++) {
    if (i == 0 || i == L2_) {
      section_[i] = section_[i] * GSL_REAL(gsl_vector_complex_get(frequencyResponse_, i));
    } else {
      gsl_complex val = gsl_complex_mul(gsl_complex_rect(section_[i], section_[L_-i]), gsl_vector_complex_get(frequencyResponse_, i));
      section_[i]    = GSL_REAL(val);
      section_[L_-i] = GSL_IMAG(val);
    }
  }

  // inverse FFT
  gsl_fft_halfcomplex_radix2_inverse(section_, /* stride= */ 1, L_);

  // pick out linearly convolved portion
  for (unsigned i = P_ ; i < L_; i++)
    gsl_vector_float_set(vector_, i - P_, section_[i]);

  increment_();
  return vector_;
}

void OverlapSave::reset()
{
  samp_->reset();  VectorFloatFeatureStream::reset();
}

void OverlapSave::update(const gsl_vector_complex* delta)
{
  if (delta->size != L_)
    throw jdimension_error("Dimension of udpate vector (%d) does not match frequency response (%d).",
			   delta->size, L_);

  for (unsigned i = 0; i < L_; i++)
    gsl_vector_complex_set(frequencyResponse_, i, gsl_complex_add(gsl_vector_complex_get(frequencyResponse_, i), gsl_vector_complex_get(delta, i)));
}
