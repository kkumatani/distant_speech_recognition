/*
 * @file modulated.h
 * @brief Cosine modulated analysis and synthesis filter banks.
 * @author John McDonough and Kenichi Kumatani
 */
#ifndef MODULATED_H
#define MODULATED_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"
//#include "btk.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif

inline int powi(int x, int p)
{
  if(p == 0) return 1;
  if(x == 0 && p > 0) return 0;
  if(p < 0) {assert(x == 1 || x == -1); return (-p % 2) ? x : 1;}

  int r = 1;
  for(;;) {
    if(p & 1) r *= x;
    if((p >>= 1) == 0)	return r;
    x *= x;
  }
}

/**
* \defgroup FilterBanks Filter Banks
* This hierarchy of classes provides the capability to divide
* signal into 'M' subbands and then resynthesize the original time-domain signal.
*/
/*@{*/

// ----- definition for class `BaseFilterBank' -----
//
class BaseFilterBank {
 public:
  virtual ~BaseFilterBank();

  virtual void reset() = 0;

 protected:
  class RealBuffer_ {
  public:
    /*
      @brief Construct a circular buffer to keep samples periodically.
             It keeps nsamp arrays which is completely updated with the period 'nsamp'.
             Each array holds actual values of the samples.
      @param unsigned len [in] The size of each array
      @param unsigned nsamp [in] The period of the circular buffer
    */
    RealBuffer_(unsigned len, unsigned nsamp)
      : len_(len), nsamp_(nsamp), zero_(nsamp_ - 1), samples_(new gsl_vector*[nsamp_])
    {
      for (unsigned i = 0; i < nsamp_; i++)
	samples_[i] = gsl_vector_calloc(len_);
    }
    ~RealBuffer_()
    {
      for (unsigned i = 0; i < nsamp_; i++)
	gsl_vector_free(samples_[i]);
      delete[] samples_;
    }

    const double sample(unsigned timeX, unsigned binX) const {
      unsigned idx = index_(timeX);
      const gsl_vector* vec = samples_[idx];
      return gsl_vector_get(vec, binX);
    }

    void nextSample(const gsl_vector* s = NULL, bool reverse = false) {
      zero_ = (zero_ + 1) % nsamp_;

      gsl_vector* nextBlock = samples_[zero_];

      if (s == NULL) {
	gsl_vector_set_zero(nextBlock);
      } else {
	if (s->size != len_)
	  throw jdimension_error("'RealBuffer_': Sizes do not match (%d vs. %d)", s->size, len_);
	assert( s->size == len_ );
	if (reverse)
	  for (unsigned i = 0; i < len_; i++)
	    gsl_vector_set(nextBlock, i, gsl_vector_get(s, len_ - i - 1));
	else
	  gsl_vector_memcpy(nextBlock, s);
      }
    }

    void nextSample(const gsl_vector_float* s) {
      zero_ = (zero_ + 1) % nsamp_;

      gsl_vector* nextBlock = samples_[zero_];

      assert( s->size == len_ );
      for (unsigned i = 0; i < len_; i++)
	gsl_vector_set(nextBlock, i, gsl_vector_float_get(s, i));
    }

    void nextSample(const gsl_vector_short* s) {
      zero_ = (zero_ + 1) % nsamp_;

      gsl_vector* nextBlock = samples_[zero_];

      assert( s->size == len_ );
      for (unsigned i = 0; i < len_; i++)
	gsl_vector_set(nextBlock, i, gsl_vector_short_get(s, i));
    }

    void zero() {
      for (unsigned i = 0; i < nsamp_; i++)
	gsl_vector_set_zero(samples_[i]);
      zero_ = nsamp_ - 1;
    }

  private:
    unsigned index_(unsigned idx) const {
      assert(idx < nsamp_);
      unsigned ret = (zero_ + nsamp_ - idx) % nsamp_;
      return ret;
    }

    const unsigned				len_;
    const unsigned				nsamp_;
    unsigned					zero_; // index of most recent sample
    gsl_vector**				samples_;
  };

  BaseFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false);

  const unsigned				M_;
  const unsigned				Mx2_;
  const unsigned				m_;
  const unsigned				mx2_;
  const unsigned				r_;
  const unsigned				R_;
  const unsigned				Rx2_;
  const unsigned				D_;
};


// ----- definition for class `NormalFFTAnalysisBank' -----
//
/**
   @class do FFT on time discrete samples multiplied with a window.
*/
class NormalFFTAnalysisBank
  : protected BaseFilterBank, public VectorComplexFeatureStream {
 public:
  NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
                        unsigned fftLen,  unsigned r = 1, unsigned windowType = 1,
                        const String& nm = "NormalFFTAnalysisBank");
  ~NormalFFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  unsigned fftLen() const { return N_; }

protected:
  void update_buf_();
  virtual void update_buffer_(int frame_no);

#ifdef HAVE_LIBFFTW3
  fftw_plan                          fftwPlan_;
#endif
  const VectorFloatFeatureStreamPtr  samp_;
  int                                winType_; // 1 = hamming, 2 = hann window
  unsigned                           N_;       // FFT length
  const unsigned	             processing_delay_;
  unsigned			     framesPadded_;
  RealBuffer_			     buffer_;
  gsl_vector*			     convert_;
  RealBuffer_		             gsi_;
  double*			     output_;
  const gsl_vector*		     prototype_;
};

typedef Inherit<NormalFFTAnalysisBank, VectorComplexFeatureStreamPtr> NormalFFTAnalysisBankPtr;

/**
* \defgroup OversampledFilterBank Oversampled Filter Bank
*/
/*@{*/


// ----- definition for class `OverSampledDFTFilterBank' -----
//
class OverSampledDFTFilterBank : public BaseFilterBank {
public:
  ~OverSampledDFTFilterBank();

  virtual void reset();

  protected:
  OverSampledDFTFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false, unsigned delayCompensationType=0, int gainFactor=1 );

  double polyphase(unsigned m, unsigned n) const {
    return gsl_vector_get(prototype_, m + M_ * n);
  }

  unsigned				laN_; /*>! the number of look-ahead */
  const unsigned			N_;
  unsigned				processing_delay_;
  const gsl_vector*			prototype_;
  RealBuffer_				buffer_;
  gsl_vector*				convert_;
  RealBuffer_				gsi_;
  const int                             gain_factor_;
};

/*@}*/

/**
* \defgroup PerfectReconstructionFilterBank Perfect Reconstruction Filter Bank
*/
/*@{*/


// ----- definition for class `PerfectReconstructionFilterBank' -----
//
class PerfectReconstructionFilterBank : public BaseFilterBank {
public:
  ~PerfectReconstructionFilterBank();

  virtual void reset();

protected:
  PerfectReconstructionFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false);

  double polyphase(unsigned m, unsigned n) const {
    return gsl_vector_get(prototype_, m + Mx2_ * n);
  }

  const unsigned				N_;
  const unsigned				processing_delay_;

  const gsl_vector*				prototype_;
  RealBuffer_					buffer_;
  gsl_vector*					convert_;
  gsl_vector_complex*				w_;
  RealBuffer_					gsi_;
};

/*@}*/

/**
* \addtogroup OversampledFilterBank
*/
/*@{*/

// ----- definition for class `OverSampledDFTAnalysisBank' -----
//
class OverSampledDFTAnalysisBank
: protected OverSampledDFTFilterBank, public VectorComplexFeatureStream {
 public:
  OverSampledDFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			     gsl_vector* prototype, unsigned M, unsigned m, unsigned r, unsigned delayCompensationType =0,
			     const String& nm = "OverSampledDFTAnalysisBank");
  ~OverSampledDFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset();

  unsigned fftLen()	 const { return M_; }
  unsigned nBlocks()	 const { return 4; }
  unsigned subsamplerate() const { return 2; }
#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() const { return subsamplerate(); }
#endif
  bool is_end(){return is_end_;}

  using OverSampledDFTFilterBank::polyphase;

 private:
  void update_buf_();
  bool update_buffer_(int frame_no);

#ifdef HAVE_LIBFFTW3
  fftw_plan					fftwPlan_;
#endif
  const VectorFloatFeatureStreamPtr		samp_;
  double*					polyphase_output_;
  unsigned					framesPadded_;
};

typedef Inherit<OverSampledDFTAnalysisBank, VectorComplexFeatureStreamPtr> OverSampledDFTAnalysisBankPtr;


// ----- definition for class `OverSampledDFTSynthesisBank' -----
//
class OverSampledDFTSynthesisBank
: private OverSampledDFTFilterBank, public VectorFloatFeatureStream {
 public:
  OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
			      gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
			      unsigned delayCompensationType = 0, int gainFactor=1,
			      const String& nm = "OverSampledDFTSynthesisBank");

  OverSampledDFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
			      unsigned delayCompensationType = 0, int gainFactor=1,
			      const String& nm = "OverSampledDFTSynthesisBank");

  ~OverSampledDFTSynthesisBank();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  using OverSampledDFTFilterBank::polyphase;

  void input_source_vector(const gsl_vector_complex* block){ update_buf_(block); }
  void no_stream_feature(bool flag=true){ no_stream_feature_ = flag; }
#ifdef ENABLE_LEGACY_BTK_API
  void inputSourceVector(const gsl_vector_complex* block){ input_source_vector(block); }
  void doNotUseStreamFeature(bool flag=true){ no_stream_feature(flag); }
#endif

 private:
  bool update_buffer_(int frame_no);
  void update_buf_(const gsl_vector_complex* block);

  const VectorComplexFeatureStreamPtr		samp_;
  bool                                          no_stream_feature_;
#ifdef HAVE_LIBFFTW3
  fftw_plan					fftwPlan_;
#endif
  double*					polyphase_input_;
};

typedef Inherit<OverSampledDFTSynthesisBank, VectorFloatFeatureStreamPtr> OverSampledDFTSynthesisBankPtr;

/*@}*/

/**
* \addtogroup PerfectReconstructionFilterBank
*/
/*@{*/

// ----- definition for class `PerfectReconstructionFFTAnalysisBank' -----
//
class PerfectReconstructionFFTAnalysisBank
: protected PerfectReconstructionFilterBank, public VectorComplexFeatureStream {
 public:
  PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
				       gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
				       const String& nm = "PerfectReconstructionFFTAnalysisBank");
  ~PerfectReconstructionFFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset();

  unsigned fftLen()	 const { return Mx2_; }
  unsigned nBlocks()	 const { return 4; }
  unsigned subsamplerate() const { return 2; }
#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() const { return subsamplerate(); }
#endif

  using PerfectReconstructionFilterBank::polyphase;

 protected:
  void update_buf_();
  virtual void update_buffer_(int frame_no);

#ifdef HAVE_LIBFFTW3
  fftw_plan					fftwPlan_;
#endif
  double*					polyphase_output_;
  unsigned					framesPadded_;

  const VectorFloatFeatureStreamPtr		samp_;
};

typedef Inherit<PerfectReconstructionFFTAnalysisBank, VectorComplexFeatureStreamPtr> PerfectReconstructionFFTAnalysisBankPtr;


// ----- definition for class `PerfectReconstructionFFTSynthesisBank' -----
//
class PerfectReconstructionFFTSynthesisBank
: private PerfectReconstructionFilterBank, public VectorFloatFeatureStream {
 public:
  PerfectReconstructionFFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
					gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBank");

  PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBank");

  ~PerfectReconstructionFFTSynthesisBank();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  using PerfectReconstructionFilterBank::polyphase;

 private:
  void update_buffer_(int frame_no);
  void update_buffer_(const gsl_vector_complex* block);

  const VectorComplexFeatureStreamPtr		samp_;
#ifdef HAVE_LIBFFTW3
  fftw_plan					fftwPlan_;
#endif
  double*					polyphase_input_;
};

typedef Inherit<PerfectReconstructionFFTSynthesisBank, VectorFloatFeatureStreamPtr> PerfectReconstructionFFTSynthesisBankPtr;

// ----- definition for class `DelayFeature' -----
//
class DelayFeature : public VectorComplexFeatureStream {
 public:
  DelayFeature( const VectorComplexFeatureStreamPtr& samp, float time_delay=0.0, const String& nm = "DelayFeature");
  ~DelayFeature();

  void set_time_delay(float time_delay){ time_delay_ = time_delay; }
  virtual const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset();

private:
  const VectorComplexFeatureStreamPtr samp_;
  float                               time_delay_;
};

typedef Inherit<DelayFeature, VectorComplexFeatureStreamPtr> DelayFeaturePtr;

gsl_vector* get_window(unsigned winType, unsigned winLen);

void write_gsl_format(const String& fileName, const gsl_vector* prototype);

/*@}*/

/*@}*/


#endif

