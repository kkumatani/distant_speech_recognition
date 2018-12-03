/**
 * @file sad_feature.h
 * @brief Feature for voice activity detection.
 * @author John McDonough
 */

#ifndef SAD_FEATURE_H
#define SAD_FEATURE_H

#include "stream/stream.h"
#include "common/mlist.h"


// ----- definition for class `BrightnessFeature' -----
//
class BrightnessFeature : public VectorFloatFeatureStream {
 public:
  BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness");
  virtual ~BrightnessFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  bool						weight_;
  float						samplerate_;
  float						max_;
  float						df_;
  float*					frs_;
};

typedef Inherit<BrightnessFeature, VectorFloatFeatureStreamPtr> BrightnessFeaturePtr;


// ----- definition for class `EnergyDiffusionFeature' -----
//
class EnergyDiffusionFeature : public VectorFloatFeatureStream {
 public:
  EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "EnergyDiffusion");
  virtual ~EnergyDiffusionFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
};

typedef Inherit<EnergyDiffusionFeature, VectorFloatFeatureStreamPtr> EnergyDiffusionFeaturePtr;


// ----- definition for class `BandEnergyRatioFeature' -----
//
class BandEnergyRatioFeature : public VectorFloatFeatureStream {
 public:
  BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~BandEnergyRatioFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  float						samplerate_;
  float						max_;
  float						df_;
  float						threshF_;
  int						threshX_;
};

typedef Inherit<BandEnergyRatioFeature, VectorFloatFeatureStreamPtr> BandEnergyRatioFeaturePtr;


// ----- definition for class `NormalizedFluxFeature' -----
//
class NormalizedFluxFeature : public VectorFloatFeatureStream {
 public:
  NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~NormalizedFluxFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  float*					win0_;
  float*					win1_;
};

typedef Inherit<NormalizedFluxFeature, VectorFloatFeatureStreamPtr> NormalizedFluxFeaturePtr;


// ----- definition for class `NegativeEntropyFeature' -----
//
class NegativeEntropyFeature : public VectorFloatFeatureStream {
 public:
  NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy");
  virtual ~NegativeEntropyFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  float*					win_;
};

typedef Inherit<NegativeEntropyFeature, VectorFloatFeatureStreamPtr> NegativeEntropyFeaturePtr;


// ----- definition for class `SignificantSubbandsFeature' -----
//
class SignificantSubbandsFeature : public VectorFloatFeatureStream {
 public:
  SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~SignificantSubbandsFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  float						thresh_;
  float*					win_;
};

typedef Inherit<SignificantSubbandsFeature, VectorFloatFeatureStreamPtr> SignificantSubbandsFeaturePtr;


// ----- definition for class `NormalizedBandwidthFeature' -----
//
class NormalizedBandwidthFeature : public VectorFloatFeatureStream {
 public:
  NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~NormalizedBandwidthFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			src_;
  float						thresh_;
  float						samplerate_;
  float						df_;
  float*					frs_;
  float*					win_;
};

typedef Inherit<NormalizedBandwidthFeature, VectorFloatFeatureStreamPtr> NormalizedBandwidthFeaturePtr;

#endif

