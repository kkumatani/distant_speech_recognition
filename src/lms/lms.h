/**
 * @file lms.h
 * @brief Implementation of LMS algorithms.
 * @author John McDonough
 */

#ifndef LMS_H
#define LMS_H

#include "convolution/convolution.h"


// ----- definition for class `FastBlockLMSFeature' -----
//
class FastBlockLMSFeature : public VectorFloatFeatureStream {
 public:
  FastBlockLMSFeature(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha, float gamma, const String& nm = "Fast Block LMS Feature");
  virtual ~FastBlockLMSFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

  void update();

 private:
  void halfComplexPack_(double*  tgt, const gsl_vector_complex* src);
  void halfComplexUnpack_(gsl_vector_complex* tgt, const double* src);

  VectorFloatFeatureStreamPtr			_desired;
  VectorFloatFeatureStreamPtr			_samp;
  OverlapSavePtr				_overlapSave;

  const unsigned				_N;
  const unsigned				_M;

  float						_alpha;
  float						_gamma;

  double*					_e;
  double*					_u;
  double*					_phi;

  gsl_vector_complex*				_U;
  gsl_vector_complex*				_E;
  gsl_vector_complex*				_Phi;

  gsl_vector*					_D;
};

typedef Inherit<FastBlockLMSFeature, VectorFloatFeatureStreamPtr> FastBlockLMSFeaturePtr;


#endif
