#ifndef SPECTRALSUBTRACTION_H
#define SPECTRALSUBTRACTION_H

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
#include "modulated/modulated.h"

class PSDEstimator {

 public:
  PSDEstimator(unsigned fftLen2);
  ~PSDEstimator();

  bool read_estimates( const String& fn );
  bool write_estimates( const String& fn );

  const gsl_vector* estimate() const {
    return estimates_;
  }

 protected:
  gsl_vector* estimates_; /* estimated noise PSD */
};

typedef refcount_ptr<PSDEstimator> PSDEstimatorPtr;

/**
   @class estimate the noise PSD for spectral subtraction (SS)
   1. construct an object 
   2. add a noise sample
   3. get noise estimates
 */
class AveragePSDEstimator : public PSDEstimator {
 public:
  /** 
      @brief A constructor for SS
      @param unsigned fftLen2[in] the half of the FFT point
      @param double alpha[in] the forgetting factor for recursive averaging. 
                              If this is negative, the average is used
                              as the noise PSD estimate.
   */
  AveragePSDEstimator(unsigned fftLen2, float alpha = -1.0 );
  ~AveragePSDEstimator();

  void clear_samples();
  const gsl_vector* average();
  bool add_sample( const gsl_vector_complex *sample );
  void clear();

 protected:
  float alpha_;
  bool sample_added_;
  list<gsl_vector *> sampleL_;
};

typedef Inherit<AveragePSDEstimator, PSDEstimatorPtr> AveragePSDEstimatorPtr;

class SpectralSubtractor : public VectorComplexFeatureStream {
public:
  SpectralSubtractor(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor");
  ~SpectralSubtractor();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void clear(){
    for(NoisePSDIterator_ itr = noisePSDList_.begin(); itr != noisePSDList_.end(); itr++)
      (*itr)->clear();
  }

  void set_noise_over_estimation_factor(float ft){ ft_ = ft; }
  void set_channel(VectorComplexFeatureStreamPtr& chan, double alpha=-1 );
  void start_training(){ training_started_ = true; }
  void stop_training();
  void clear_noise_samples();
  void start_noise_subtraction(){ start_noise_subtraction_ = true; }
  void stop_noise_subtraction(){ start_noise_subtraction_ = false; }
  bool read_noise_file(const String& fn, unsigned idx=0);
  bool write_noise_file(const String& fn, unsigned idx=0){ return noisePSDList_.at(idx)->write_estimates(fn); }

#ifdef ENABLE_LEGACY_BTK_API
  void setNoiseOverEstimationFactor(float ft){ set_noise_over_estimation_factor(ft); }
  void setChannel(VectorComplexFeatureStreamPtr& chan, double alpha=-1){ set_channel(chan, alpha); }
  void startTraining(){ start_training(); }
  void stopTraining(){ stop_training(); }
  void clearNoiseSamples(){ clear_noise_samples(); }
  void startNoiseSubtraction(){ start_noise_subtraction(); }
  void stopNoiseSubtraction(){ stop_noise_subtraction(); }
  bool readNoiseFile( const String& fn, unsigned idx=0 ){ return read_noise_file(fn, idx); }
  bool writeNoiseFile( const String& fn, unsigned idx=0 ){ return write_noise_file(fn, idx); }
#endif

 protected:
  typedef list<VectorComplexFeatureStreamPtr>	ChannelList_;
  typedef ChannelList_::iterator		ChannelIterator_;
  typedef vector<AveragePSDEstimatorPtr>	NoisePSDList_;
  typedef NoisePSDList_::iterator		NoisePSDIterator_;

  ChannelList_		channelList_;
  NoisePSDList_         noisePSDList_;
  unsigned		fftLen_;
  unsigned		fftLen2_;
  bool			halfBandShift_;
  bool                  training_started_;
  unsigned              totalTrainingSampleN_;
  float			ft_;
  float			flooringV_;
  bool                  start_noise_subtraction_;
};

typedef Inherit<SpectralSubtractor, VectorComplexFeatureStreamPtr> SpectralSubtractorPtr;

// ----- definition for class 'WienerFilter' -----
// 
class WienerFilter : public VectorComplexFeatureStream {
 public:
  WienerFilter( VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter");
  ~WienerFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

  void    set_noise_amplification_factor(double beta){ beta_ = beta; }
  void    start_updating_noise_PSD(){ update_noise_PSD_ = true; }
  void    stop_updating_noise_PSD(){ update_noise_PSD_ = false; }

#ifdef ENABLE_LEGACY_BTK_API
  void    setNoiseAmplificationFactor(double beta){ set_noise_amplification_factor(beta); }
  void    startUpdatingNoisePSD(){ start_updating_noise_PSD(); }
  void    stopUpdatingNoisePSD(){ stop_updating_noise_PSD(); }
#endif

 private:
  gsl_vector *prev_PSDs_;
  gsl_vector *prev_PSDn_;
  VectorComplexFeatureStreamPtr target_signal_;
  VectorComplexFeatureStreamPtr noise_signal_;
  bool                          halfBandShift_;
  float			alpha_; // forgetting factor
  float			flooringV_;
  float                 beta_; // amplification coefficient for a noise signal
  bool                  update_noise_PSD_;
  unsigned		fftLen_;
  unsigned		fftLen2_;
};

typedef Inherit<WienerFilter, VectorComplexFeatureStreamPtr> WienerFilterPtr;

#endif
