/**
 * @file binauralprocessing.h
 * @brief Binaural processing
 * @author Kenichi Kumatani
 */
#ifndef BINAURALPROCESSING_H
#define BINAURALPROCESSING_H

#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "postfilter/spectralsubtraction.h"
#include "beamformer/spectralinfoarray.h"
#include "beamformer/beamformer.h"

class BinaryMaskFilter : public VectorComplexFeatureStream {
public:
  BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha,
		       float dEta = 0.01, const String& nm = "BinaryMaskFilter" );
  ~BinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void set_threshold( float threshold ){ threshold_ = threshold; }
  void set_thresholds( const gsl_vector *thresholds );
  double threshold(){ return threshold_; }
  gsl_vector *thresholds(){ return threshold_per_freq_; }

#ifdef ENABLE_LEGACY_BTK_API
  void setThreshold( float threshold ){ set_threshold(threshold); }
  void setThresholds( const gsl_vector *thresholds ){ set_thresholds(thresholds); }
  double getThreshold(){ return threshold(); }
  gsl_vector *getThresholds(){ return thresholds(); }
#endif

protected:
  VectorComplexFeatureStreamPtr srcL_; /* left channel */
  VectorComplexFeatureStreamPtr srcR_; /* right channel */
  unsigned          chanX_;  /* want to extract the index of a channel */
  gsl_vector_float *prevMu_; /* binary mask at a previous frame */
  float             alpha_;  /* forgetting factor */
  float             dEta_;        /* flooring value */
  float             threshold_;   /* threshold for the ITD */
  gsl_vector        *threshold_per_freq_;
};

typedef Inherit<BinaryMaskFilter, VectorComplexFeatureStreamPtr> BinaryMaskFilterPtr;

/**
   @class Implementation of binary masking based on C. Kim's Interspeech2010 paper 
   @brief binary mask two inputs based of the threshold of the interaural time delay (ITD)
   @usage
 */
class KimBinaryMaskFilter : public BinaryMaskFilter {
public:
  KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha,
		       float dEta = 0.01, float dPowerCoeff = 1/15.0, const String& nm = "KimBinaryMaskFilter" );
  ~KimBinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );

protected:
  float             dpower_coeff_; /* power law non-linearity */
};

typedef Inherit<KimBinaryMaskFilter, BinaryMaskFilterPtr> KimBinaryMaskFilterPtr;

/**
   @class Implementation of estimating the threshold for C. Kim's ITD-based binary masking
   @brief binary mask two inputs based of the threshold of the interaural time delay (ITD)
   @usage
 */
class KimITDThresholdEstimator : public KimBinaryMaskFilter {
public:
  KimITDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 0.02, float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15.0, const String& nm = "KimITDThresholdEstimator" );
  ~KimITDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void   reset();
  virtual double calc_threshold();
  const gsl_vector* cost_function();

#ifdef ENABLE_LEGACY_BTK_API
  virtual double calcThreshold(){ return calc_threshold(); }
  const gsl_vector* getCostFunction(){ return cost_function(); }
#endif

protected:
  virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );

  /* for restricting the search space */
  float min_threshold_;
  float max_threshold_;
  float width_;
  unsigned min_fbinX_;
  unsigned max_fbinX_;
  /* work space */
  double *cost_func_values_;
  double *sigma_T_;
  double *sigma_I_;
  double *mean_T_;
  double *mean_I_;
  unsigned int nCand_;
  unsigned int nSamples_;
  gsl_vector *buffer_; /* for returning values from the python */
  bool cost_func_computed_;
};

typedef Inherit<KimITDThresholdEstimator, KimBinaryMaskFilterPtr> KimITDThresholdEstimatorPtr;

/**
	@class
 */
class IIDBinaryMaskFilter : public BinaryMaskFilter {
public:
	IIDBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
						unsigned M, float threshold, float alpha,
						float dEta = 0.01, const String& nm = "IIDBinaryMaskFilter" );
	~IIDBinaryMaskFilter();
	virtual const gsl_vector_complex* next(int frame_no = -5);
	virtual void reset();
	
	virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );
};

typedef Inherit<IIDBinaryMaskFilter, BinaryMaskFilterPtr> IIDBinaryMaskFilterPtr;

/**
 @class binary masking based on a difference between magnitudes of two beamformers' outputs.
 @brief set zero to beamformer's output with the smaller magnitude over frequency bins
 */
class IIDThresholdEstimator : public KimITDThresholdEstimator {
public:
	IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
							 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
							 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 0.5, const String& nm = "IIDThresholdEstimator" );
	~IIDThresholdEstimator();
	virtual const gsl_vector_complex* next(int frame_no = -5);
	virtual void reset();
	virtual double calc_threshold();

#ifdef ENABLE_LEGACY_BTK_API
	virtual double calcThreshold(){ return calc_threshold(); }
#endif

protected:
        virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );

private:
        double *_Y4_T;
        double *_Y4_I;
        double _beta;
};

typedef Inherit<IIDThresholdEstimator, KimITDThresholdEstimatorPtr> IIDThresholdEstimatorPtr;

/**
 @class binary masking based on a difference between magnitudes of two beamformers' outputs at each frequency bin.
 @brief set zero to beamformer's output with the smaller magnitude at each frequency bin
 */
class FDIIDThresholdEstimator : public BinaryMaskFilter {
public:
  FDIIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
			   float minThreshold = 0, float maxThreshold = 0, float width = 1000,
			   float dEta = 0.01, float dPowerCoeff = 1/15.0, const String& nm = "FDIIDThresholdEstimator" );
  ~FDIIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void   reset();
  virtual double calc_threshold();
  const gsl_vector* cost_function(unsigned freqX);

#ifdef ENABLE_LEGACY_BTK_API
  virtual double calcThreshold(){ return calc_threshold(); }
  const gsl_vector* getCostFunction( unsigned freqX ){ return cost_function(freqX); }
#endif

protected:
  virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );

  /* for restricting the search space */
  float min_threshold_;
  float max_threshold_;
  float width_;
  float dpower_coeff_;

  /* work space */
  double **cost_func_values_;
  double **_Y4;
  double **_sigma;
  double **_mean;
  double _beta;
  unsigned int nCand_;
  unsigned int nSamples_;
  gsl_vector *buffer_; /* for returning values from the python */
  bool cost_func_computed_;
};

typedef Inherit<FDIIDThresholdEstimator, BinaryMaskFilterPtr> FDIIDThresholdEstimatorPtr;

#endif
