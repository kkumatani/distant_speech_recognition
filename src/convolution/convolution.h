/**
 * @file convolution.h
 * @brief Block convolution realization of an LTI system with the FFT.Time delay estimation
 * @author John McDonough
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"

// ----- definition for class `OverlapAdd' -----
//
class OverlapAdd : public VectorFloatFeatureStream {
 public:
  OverlapAdd(VectorFloatFeatureStreamPtr& samp,
             const gsl_vector* impulseResponse = NULL, unsigned fftLen = 0,
             const String& nm = "OverlapAdd");

  ~OverlapAdd();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:
  void set_impulse_response_(const gsl_vector* impulseResponse);
  unsigned check_fftLen_(unsigned sectionLen, unsigned irLen, unsigned fftLen);

  const VectorFloatFeatureStreamPtr			samp_;
  const unsigned					L_;
  const unsigned					P_;
  const unsigned					N_;
  const unsigned					N2_;

  double*						section_;
  gsl_vector_complex*					frequencyResponse_;
  gsl_vector_float*					buffer_;
};

typedef Inherit<OverlapAdd, VectorFloatFeatureStreamPtr> OverlapAddPtr;


// ----- definition for class `OverlapSave' -----
//
class OverlapSave : public VectorFloatFeatureStream {
 public:
  OverlapSave(VectorFloatFeatureStreamPtr& samp,
              const gsl_vector* impulseResponse = NULL, const String& nm = "OverlapSave");

  ~OverlapSave();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  void update(const gsl_vector_complex* delta);

 private:
  void set_impulse_response_(const gsl_vector* impulseResponse);
  unsigned check_output_size_(unsigned irLen, unsigned sampLen);
  unsigned check_L_(unsigned irLen, unsigned sampLen);

  const VectorFloatFeatureStreamPtr			samp_;
  const unsigned					L_;
  const unsigned					L2_;
  const unsigned					P_;

  double*						section_;
  gsl_vector_complex*					frequencyResponse_;
};

typedef Inherit<OverlapSave, VectorFloatFeatureStreamPtr> OverlapSavePtr;

#endif // CONVOLUTION_H
