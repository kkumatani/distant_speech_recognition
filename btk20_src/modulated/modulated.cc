/*
 * @file modulated.cc
 * @brief Cosine modulated analysis and synthesis filter banks.
 * @author John McDonough and Kenichi Kumatani
 */

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_complex.h>

#include "common/jpython_error.h"
#include "modulated/modulated.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif

// convert 'gsl_vector_complex' into '2 x double'
//
static void pack_complex_array_(const gsl_vector_complex* src, double* tgt)
{
  for (unsigned m = 0; m < src->size; m++) {
    tgt[2*m]   = GSL_REAL(gsl_vector_complex_get(src, m));
    tgt[2*m+1] = GSL_IMAG(gsl_vector_complex_get(src, m));
  }
}

// convert '2 x double' into 'gsl_vector_complex'
//
static void unpack_complex_array_(gsl_vector_complex* tgt, const double* src)
{
  for (unsigned m = 0; m < tgt->size; m++)
    gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[2*m], src[2*m+1]));
}

/**
   @brief calculate a window.

   @param unsigned winType[in] a flag which indicates the returned window
                               0 -> a rectangle window
                               1 -> Hamming window
                               2 -> Hanning window
   @param unsigned winLen[in] the length of a window
   @return a window
 */
gsl_vector* get_window( unsigned winType, unsigned winLen )
{
  gsl_vector* win = gsl_vector_calloc(winLen);

  switch( winType ){
  case 0:
    /* rectangle window */
    for (unsigned i = 0; i < winLen; i++)
      gsl_vector_set( win, i , 1.0 );
    break;
  case 2:
    /* Hanning window */
    for (unsigned i = 0; i < winLen; i++) {
      gsl_vector_set( win, i , 0.5 * ( 1 - cos( (2.0*M_PI*i)/(double)(winLen-1) ) ) );
    }
    break;
  default:// Hamming window
    double temp = 2. * M_PI / (double)(winLen - 1);
    for ( unsigned i = 0 ; i < winLen; i++ )
      gsl_vector_set( win, i , 0.54 - 0.46 * cos( temp * i ) );
      //gsl_vector_set( win, i , 0.53836 - 0.46164 * cos( temp * i ) );
    break;
  }

  return win;
}

// ----- methods for class `BaseFilterBank' -----
//
BaseFilterBank::
BaseFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis)
  : M_(M), Mx2_(2*M_), m_(m), mx2_(2*m_),
    r_(r), R_(1 << r_), Rx2_(2 *R_), D_(M_ / R_) { }

BaseFilterBank::~BaseFilterBank()
{
}

// ----- methods for class `OverSampledDFTAnalysisBank' -----
//

/**
   @brief construct an objects to transform samples by FFT.

   @param VectorFloatFeatureStreamPtr& samp[in/out]
   @param unsigned M[in] the length of FFT
   @param unsigned r[in] a decimation factor which decides a frame shift size.
   @param unsigned windowType[in]
*/
NormalFFTAnalysisBank::
NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
		      unsigned M, unsigned r, unsigned windowType,
		      const String& nm )
  : BaseFilterBank(NULL, M, 1, r, /*synthesis*/ false ),
    VectorComplexFeatureStream(M_, nm),
    samp_(samp),
    winType_(windowType),
    N_(M_),
    processing_delay_(mx2_ - 1),
    framesPadded_(0),
    buffer_(M_, /* m=1 */ 1 * R_),
    convert_(gsl_vector_calloc(M_)),
    gsi_( /* synthesis==false */ D_, R_)
{
  if (samp_->size() != D_)
    throw jdimension_error("Input block length (%d) != D_ (%d)\n", samp_->size(), D_);

  prototype_  = (const gsl_vector *)get_window( winType_, N_ );

#ifdef HAVE_LIBFFTW3
  output_ = static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * N_ * 2));
#else
  output_ = new double[2 * N_];
#endif

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(N_,
			       (double (*)[2])output_,
			       (double (*)[2])output_,
			       FFTW_FORWARD,
			       FFTW_MEASURE);
#endif
  reset();
}

NormalFFTAnalysisBank::~NormalFFTAnalysisBank()
{
  gsl_vector_free((gsl_vector*) prototype_);
  gsl_vector_free(convert_);

#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(output_);
#else
  delete [] output_;
#endif
}

void NormalFFTAnalysisBank::reset()
{
  buffer_.zero();  gsi_.zero();
  samp_->reset();  VectorComplexFeatureStream::reset();
  framesPadded_ = 0;
}

void NormalFFTAnalysisBank::update_buf_()
{
  for (unsigned sampX = 0; sampX < R_; sampX++)
    for (unsigned dimX = 0; dimX < D_; dimX++)
      gsl_vector_set(convert_, dimX + sampX * D_, gsi_.sample(R_ - sampX - 1, dimX));
  buffer_.nextSample(convert_, /* reverse= */ true);
}

void NormalFFTAnalysisBank::update_buffer_(int frame_no)
{
  if (framesPadded_ == 0) {				// normal processing

    try {
      /*
      if (frame_no_ == frame_reset_no_) {
	for (unsigned i = 0; i < Rx2_; i++) {
	  const gsl_vector_float* block = samp_->next(i);
	  gsi_.nextSample(block);
	}
      }
      */
      const gsl_vector_float* block = samp_->next(frame_no /* + Rx2_ */);
      gsi_.nextSample(block);
      update_buf_();
    } catch  (exception& e) {
      gsi_.nextSample();
      update_buf_();

      // printf("Padding frame %d.\n", framesPadded_);

      framesPadded_++;
    }

  } else if (framesPadded_ < processing_delay_) {	// pad with zeros

    gsi_.nextSample();
    update_buf_();

    // printf("Padding frame %d.\n", framesPadded_);

    framesPadded_++;

  } else {						// end of utterance

    throw jiterator_error("end of samples!");

  }
}

const gsl_vector_complex* NormalFFTAnalysisBank::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  update_buffer_(frame_no);

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < M_; m++) {
    double win_i = gsl_vector_get( prototype_, m );

    output_[2*m]   = win_i * buffer_.sample(0, M_ - m - 1 );
    output_[2*m+1] = 0.0;
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(fftwPlan_);
#else
  gsl_fft_complex_radix2_forward(output_, /* stride= */ 1, N_);
#endif

  unpack_complex_array_(vector_, output_);

  increment_();
  return vector_;
}


// ----- methods for class `OverSampledDFTFilterBank' -----
//
OverSampledDFTFilterBank::
OverSampledDFTFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis, unsigned delayCompensationType, int gainFactor )
  : BaseFilterBank(prototype, M, m, r, synthesis), N_(M_ * m),
    prototype_(gsl_vector_calloc(N_)), buffer_(M_, m * R_),
    convert_(gsl_vector_calloc(M_)), gsi_((synthesis ? M_ : D_), R_),
    gain_factor_(gainFactor)
{
  if (prototype->size != N_)
    throw jconsistency_error("Prototype sizes do not match (%d vs. %d).",
			     prototype->size, N_);

  gsl_vector* pr = (gsl_vector*) prototype_;
  gsl_vector_memcpy(pr, prototype);

  laN_ = 0; // indicates how many frames should be skipped.
  switch ( delayCompensationType ) {
    // de Haan's filter bank or Nyquist(M) filter bank
  case 1 : // compensate delays in the synthesis filter bank
    processing_delay_ = m * R_ - 1 ; // m * 2^r - 1 ;
    break;
  case 2 : // compensate delays in the analythesis and synthesis filter banks
    if( synthesis == true )
      processing_delay_ = m * R_ / 2 ;
    else{
      processing_delay_ = m * R_ - 1;
      laN_ = m_ * R_  / 2 - 1;
    }
    break;
    // undefined filter bank
  default :
    processing_delay_ = mx2_ - 1;
    break;
  }

  // set the buffers to zero
  reset();
}

OverSampledDFTFilterBank::~OverSampledDFTFilterBank()
{
  gsl_vector_free((gsl_vector*) prototype_);
  gsl_vector_free(convert_);
}

void OverSampledDFTFilterBank::reset()
{
  buffer_.zero();  gsi_.zero();
}


// ----- methods for class `PerfectReconstructionFilterBank' -----
//
PerfectReconstructionFilterBank::
PerfectReconstructionFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis)
  : BaseFilterBank(prototype, M, m, r, synthesis), N_(Mx2_ * m), processing_delay_(mx2_ - 1),
    prototype_(gsl_vector_calloc(N_)), buffer_(Mx2_, m * (r_ + 2)),
    convert_(gsl_vector_calloc(Mx2_)), w_(gsl_vector_complex_calloc(Mx2_)),
    gsi_((synthesis ? Mx2_ : D_), Rx2_)
{
  if (prototype->size != N_)
    throw jconsistency_error("Prototype sizes do not match (%d vs. %d).",
			     prototype->size, N_);

  gsl_vector* pr = (gsl_vector*) prototype_;
  gsl_vector_memcpy(pr, prototype);

  // set the buffers to zero
  reset();
}

PerfectReconstructionFilterBank::~PerfectReconstructionFilterBank()
{
  gsl_vector_free((gsl_vector*) prototype_);
  gsl_vector_free(convert_);
  gsl_vector_complex_free(w_);
}

void PerfectReconstructionFilterBank::reset()
{
  buffer_.zero();  gsi_.zero();
}


// ----- methods for class `OverSampledDFTAnalysisBank' -----
//
/*
  @brief construct an object to calculate subbands with analysis filter banks (FBs)
  @param VectorFloatFeatureStreamPtr& samp [in] an object to keep wave data
  @param gsl_vector* prototype [in] filter coefficients of a prototype of an analysis filter 
  @param unsigned M [in] the number of subbands
  @param unsigned m [in] fliter length factor ( the filter length == m * M )
  @param unsigned r [in] decimation factor
  @param unsigned delayCompensationType [in] 1 : delays are compensated in the synthesis FBs only. 2 : delays are compensated in the both FBs.
*/
OverSampledDFTAnalysisBank::
OverSampledDFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			   gsl_vector* prototype, unsigned M, unsigned m, unsigned r, unsigned delayCompensationType, const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*sythesis=*/ false, delayCompensationType ),
    VectorComplexFeatureStream(M_, nm),
    samp_(samp),
#ifdef HAVE_LIBFFTW3
    polyphase_output_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * Mx2_))),
#else
    polyphase_output_(new double[2 * M_]),
#endif
    framesPadded_(0)
{
  if (samp_->size() != D_)
    throw jdimension_error("Input block length (%d) != D_ (%d)\n", samp_->size(), D_);

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(M_,
			       (double (*)[2])polyphase_output_,
			       (double (*)[2])polyphase_output_,
			       FFTW_BACKWARD,
			       FFTW_MEASURE);
#endif
}

OverSampledDFTAnalysisBank::~OverSampledDFTAnalysisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(polyphase_output_);
#else
  delete[] polyphase_output_;
#endif
}

void OverSampledDFTAnalysisBank::update_buf_()
{
  /* note
     gsi_ has samp_les[R][D], M samples.
     Then the data of gsi_ are given to convert_ which has _sample[mR][M].
  */
  for (unsigned sampX = 0; sampX < R_; sampX++)
    for (unsigned dimX = 0; dimX < D_; dimX++)
      gsl_vector_set(convert_, dimX + sampX * D_, gsi_.sample(R_ - sampX - 1, dimX));
  buffer_.nextSample(convert_, /* reverse= */ true);
}

const gsl_vector_complex* OverSampledDFTAnalysisBank::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if ( true == update_buffer_(frame_no) ) {
    throw jiterator_error("end of samples!");
  }

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < M_; m++) {
    double sum  = 0.0;
    for (unsigned k = 0; k < m_; k++)
      sum  += polyphase(m, k) * buffer_.sample(R_ * k, m);

    polyphase_output_[2*m]   = sum;
    polyphase_output_[2*m+1] = 0.0;
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(fftwPlan_);
#else
  gsl_fft_complex_radix2_backward(polyphase_output_, /* stride= */ 1, M_);
#endif

  unpack_complex_array_(vector_, polyphase_output_);

  if( gain_factor_ > 0 )
    for(unsigned m = 0; m < M_; m++) {
      gsl_vector_complex_set(vector_, m,
                             gsl_complex_mul_real( gsl_vector_complex_get(vector_, m ),  gain_factor_ ) );
}

  increment_();
  return vector_;
}

void OverSampledDFTAnalysisBank::reset()
{
  samp_->reset();  OverSampledDFTFilterBank::reset();  VectorComplexFeatureStream::reset();
  buffer_.zero();
  framesPadded_ = 0;
}

bool OverSampledDFTAnalysisBank::update_buffer_(int frame_no)
{
  const gsl_vector_float* block;

  if( true == is_end_ ){// reached the end of frame
    return is_end_;
  }
  if( laN_ >0 && frame_no_ == frame_reset_no_ ) {
    // skip samples for compensating the processing delays
    for (unsigned itnX = 0; itnX < laN_; itnX++){
      try {
        block = samp_->next(itnX/* + Rx2_ */);
      }
      catch( jiterator_error &e ) {
        is_end_ = true;
      }
      if( false == is_end_ ) {
        gsi_.nextSample(block);
        update_buf_();
      }
    }
  }
  if( framesPadded_ == 0 ) {// normal processing
    try {
      if( frame_no >= 0 )
        block = samp_->next(frame_no + laN_ );
      else // just take the next frame
        block = samp_->next(frame_no );
    }
    catch( jiterator_error &e ) {
      // it happens if the number of prcessing frames exceeds the data length.
      framesPadded_++;
    }
    if( framesPadded_ == 0 ) {
      gsi_.nextSample(block);
    }
    else {
      gsi_.nextSample();
    }
    update_buf_();
  }
  else if (framesPadded_ < processing_delay_) {// pad with zeros
      gsi_.nextSample();
      update_buf_();
      framesPadded_++;
  }
  else {// end of utterance
    is_end_ = true;
  }

  return is_end_;
}


// ----- methods for class `OverSampledDFTSynthesisBank' -----
//
OverSampledDFTSynthesisBank::
OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
			    gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
			    unsigned delayCompensationType, int gainFactor,
			    const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*synthesis=*/ true, delayCompensationType, gainFactor ),
    VectorFloatFeatureStream(D_, nm),
    samp_(samp),
    no_stream_feature_(false),
#ifdef HAVE_LIBOverSampledDFTW3
    polyphase_input_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * Mx2_)))
#else
    polyphase_input_(new double[2 * M_])
#endif
{
#ifdef HAVE_LIBOverSampledDFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(M_,
			       (double (*)[2])polyphase_input_,
			       (double (*)[2])polyphase_input_,
			       OverSampledDFTW_FORWARD,
			       OverSampledDFTW_MEASURE);
#endif
}

OverSampledDFTSynthesisBank::
OverSampledDFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, 
			    unsigned delayCompensationType, int gainFactor, 
			    const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*synthesis=*/ true, delayCompensationType, gainFactor ),
    VectorFloatFeatureStream(D_, nm),no_stream_feature_(true),
#ifdef HAVE_LIBOverSampledDFTW3
    polyphase_input_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * M_)))
#else
    polyphase_input_(new double[2 * M_])
#endif
{
#ifdef HAVE_LIBOverSampledDFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(M_,
			       (double (*)[2])polyphase_input_,
			       (double (*)[2])polyphase_input_,
			       OverSampledDFTW_FORWARD,
			       OverSampledDFTW_MEASURE);
#endif
}

OverSampledDFTSynthesisBank::~OverSampledDFTSynthesisBank()
{
#ifdef HAVE_LIBOverSampledDFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(polyphase_input_);
#else
  delete[] polyphase_input_;
#endif
}

bool OverSampledDFTSynthesisBank::update_buffer_(int frame_no)
{
  const gsl_vector_complex* block;

  // get next frame and perform forward OverSampledDFT
  if( false == no_stream_feature_ ){
    try {
      block = samp_->next(frame_no);
    }
    catch( jiterator_error &e ) {
      is_end_ = true;
      return is_end_;
    }
    update_buf_(block);
  }
  return false;
}

void OverSampledDFTSynthesisBank::update_buf_(const gsl_vector_complex* block)
{
  // get next frame and perform forward OverSampledDFT
  pack_complex_array_(block, polyphase_input_);

#ifdef HAVE_LIBOverSampledDFTW3
  fftw_execute(fftwPlan_);
#else
  gsl_fft_complex_radix2_forward(polyphase_input_, /* stride= */ 1, M_);
#endif

  for (unsigned m = 0; m < M_; m++)
    gsl_vector_set(convert_, m, polyphase_input_[2*m]);

  // update buffer
  buffer_.nextSample(convert_);
}

const gsl_vector_float* OverSampledDFTSynthesisBank::next(int frame_no)
{
  if (frame_no == frame_no_ + processing_delay_) return vector_;

  // "prime" the buffer
  if (frame_no_ == frame_reset_no_) {
    for (unsigned itnX = 0; itnX < processing_delay_; itnX++)
      if ( true == update_buffer_(itnX) )
        throw jiterator_error("end of samples!");
  }

  if ( frame_no >= 0 && frame_no - 1 != frame_no_ )
    printf("The output might not be continuous %s: %d != %d\n",name().c_str(), frame_no - 1, frame_no_);

  if( frame_no >= 0 ) {
    if ( true == update_buffer_( frame_no + processing_delay_) )
      throw jiterator_error("end of samples!");
  }
  else {
    if( true == update_buffer_(frame_no_ + 1 + processing_delay_) )
      throw jiterator_error("end of samples!");
  }
  increment_();

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < M_; m++) {
    double sum  = 0.0;
    for (unsigned k = 0; k < m_; k++)
      sum  += polyphase(M_ - m - 1, k) * buffer_.sample(R_ * k, m);
    gsl_vector_set(convert_, m, sum);
  }
  gsi_.nextSample(convert_);

  // synthesize final output of filterbank
  gsl_vector_float_set_zero(vector_);
  for (unsigned sampX = 0; sampX < R_; sampX++)
    for (unsigned d = 0; d < D_; d++)
      gsl_vector_float_set(vector_, D_ - d - 1, gsl_vector_float_get(vector_, D_ - d - 1) + gsi_.sample(R_ - sampX - 1, d + sampX * D_) );

  if( gain_factor_ > 0 )
    gsl_vector_float_scale(vector_, (float)gain_factor_);

  return vector_;
}

void OverSampledDFTSynthesisBank::reset()
{
  if( false == no_stream_feature_ ){
    samp_->reset();
  }
  OverSampledDFTFilterBank::reset();
  VectorFloatFeatureStream::reset();
  buffer_.zero();
}

void write_gsl_format(const String& fileName, const gsl_vector* prototype)
{
  FILE* fp = btk_fopen(fileName, "w");
  gsl_vector_fwrite(fp, prototype);
  btk_fclose(fileName, fp);
}


// ----- methods for class `PerfectReconstructionFFTAnalysisBank' -----
//
PerfectReconstructionFFTAnalysisBank::
PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp, gsl_vector* prototype,
				     unsigned M, unsigned m, unsigned r, const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*sythesis=*/ false),
    VectorComplexFeatureStream(Mx2_, nm),
#ifdef HAVE_LIBFFTW3
    polyphase_output_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * Mx2_))),
#else
    polyphase_output_(new double[2 * Mx2_]),
#endif
    framesPadded_(0), samp_(samp)
{
  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ - M_PI / (2.0 * M_));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);

  for (unsigned k = 0; k < Mx2_; k++) {
    gsl_vector_complex_set(w_, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(Mx2_,
			       (double (*)[2])polyphase_output_,
			       (double (*)[2])polyphase_output_,
			       FFTW_BACKWARD,
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTAnalysisBank::~PerfectReconstructionFFTAnalysisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(polyphase_output_);
#else
  delete[] polyphase_output_;
#endif
}

void PerfectReconstructionFFTAnalysisBank::update_buf_()
{
  for (unsigned sampX = 0; sampX < Rx2_; sampX++)
    for (unsigned dimX = 0; dimX < D_; dimX++)
      gsl_vector_set(convert_, dimX + sampX * D_, gsi_.sample(Rx2_ - sampX - 1, dimX));
  buffer_.nextSample(convert_, /* reverse= */ true);
}

const gsl_vector_complex* PerfectReconstructionFFTAnalysisBank::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  update_buffer_(frame_no);

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < Mx2_; m++) {
    double sum  = 0.0;
    int    flip = 1;

    for (unsigned k = 0; k < m_; k++) {
      sum  += flip * polyphase(m, k) * buffer_.sample((r_ + 2) * k, m);
      flip *= -1;
    }

    gsl_complex output      = gsl_complex_mul_real(gsl_vector_complex_get(w_, m), sum);
    polyphase_output_[2*m]   = GSL_REAL(output);
    polyphase_output_[2*m+1] = GSL_IMAG(output);
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(fftwPlan_);
  // scale output vector
  for(int i=0; i<Mx2_*2; i++) {
    polyphase_output_[i] = polyphase_output_[i] / Mx2_;
  }
#else
  gsl_fft_complex_radix2_inverse(polyphase_output_, /* stride= */ 1, Mx2_);
#endif

  unpack_complex_array_(vector_, polyphase_output_);

  increment_();
  return vector_;
}

void PerfectReconstructionFFTAnalysisBank::reset()
{
  PerfectReconstructionFilterBank::reset();  VectorComplexFeatureStream::reset();
  buffer_.zero();    samp_->reset();
  framesPadded_ = 0;
}

void PerfectReconstructionFFTAnalysisBank::update_buffer_(int frame_no)
{
  if (framesPadded_ == 0) { // normal processing
    try {
      /*
      if (frame_no_ == frame_reset_no_) {
	for (unsigned i = 0; i < Rx2_; i++) {
	  const gsl_vector_float* block = samp_->next(i);
	  gsi_.nextSample(block);
	}
      }
      */
      const gsl_vector_float* block = samp_->next(frame_no /* + Rx2_ */);
      gsi_.nextSample(block);
      update_buf_();
    } catch  (exception& e) {
      gsi_.nextSample();
      update_buf_();
      // printf("Padding frame %d.\n", framesPadded_);
      framesPadded_++;
    }
  } else if (framesPadded_ < processing_delay_) { // pad with zeros
    gsi_.nextSample();
    update_buf_();
    // printf("Padding frame %d.\n", framesPadded_);
    framesPadded_++;
  } else { // end of utterance
    throw jiterator_error("end of samples!");
  }
}


// ----- methods for class `PerfectReconstructionFFTSynthesisBank' -----
//
PerfectReconstructionFFTSynthesisBank::
PerfectReconstructionFFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
				      gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
				      const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*synthesis=*/ true),
    VectorFloatFeatureStream(D_, nm),
    samp_(samp),
#ifdef HAVE_LIBFFTW3
    polyphase_input_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * Mx2_)))
#else
    polyphase_input_(new double[2 * Mx2_])
#endif
{
  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ M_PI / (2.0 * M_));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);

  for (unsigned k = 0; k < Mx2_; k++) {
    gsl_vector_complex_set(w_, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(Mx2_,
			       (double (*)[2])polyphase_input_,
			       (double (*)[2])polyphase_input_,
			       FFTW_FORWARD,
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTSynthesisBank::
PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
		 const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*synthesis=*/ true),
    VectorFloatFeatureStream(D_, nm),
#ifdef HAVE_LIBFFTW3
    polyphase_input_(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * Mx2_)))
#else
    polyphase_input_(new double[2 * Mx2_])
#endif
{
  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ M_PI / (2.0 * M_));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);

  for (unsigned k = 0; k < Mx2_; k++) {
    gsl_vector_complex_set(w_, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_1d(Mx2_,
			       (double (*)[2])polyphase_input_,
			       (double (*)[2])polyphase_input_,
			       FFTW_FORWARD,
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTSynthesisBank::~PerfectReconstructionFFTSynthesisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(polyphase_input_);
#else
  delete[] polyphase_input_;
#endif
}

void PerfectReconstructionFFTSynthesisBank::update_buffer_(int frame_no)
{
  // get next frame and perform forward FFT
  const gsl_vector_complex* block = samp_->next(frame_no);
  update_buffer_(block);
}

void PerfectReconstructionFFTSynthesisBank::update_buffer_(const gsl_vector_complex* block)
{
  // get next frame and perform forward FFT
  pack_complex_array_(block, polyphase_input_);

#ifdef HAVE_LIBFFTW3
  fftw_execute(fftwPlan_);
#else
  gsl_fft_complex_radix2_forward(polyphase_input_, /* stride= */ 1, Mx2_);
#endif

  // apply 'w' factors
  for (unsigned m = 0; m < Mx2_; m++) {
    gsl_complex val = gsl_complex_rect(polyphase_input_[2*m], polyphase_input_[2*m+1]);
    gsl_vector_set(convert_, m, GSL_REAL(gsl_complex_mul(val, gsl_vector_complex_get(w_, m))));
  }

  // update buffer
  buffer_.nextSample(convert_);
}

const gsl_vector_float* PerfectReconstructionFFTSynthesisBank::next(int frame_no)
{
  if (frame_no == frame_no_ + processing_delay_) return vector_;

  // "prime" the buffer
  if (frame_no_ == frame_reset_no_) {
    for (unsigned itnX = 0; itnX < processing_delay_; itnX++)
      update_buffer_(itnX);
  }

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frame_no - 1, frame_no_);

  update_buffer_(frame_no_ + 1 + processing_delay_);
  increment_();

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < Mx2_; m++) {
    double sum  = 0.0;
    int    flip = (m_ % 2 == 1) ? 1 : -1;

    for (unsigned k = 0; k < m_; k++) {
      sum  += flip * polyphase(m, m_ - k - 1) * buffer_.sample((r_ + 2) * k, m);
      flip *= -1;
    }
    gsl_vector_set(convert_, m, sum);
  }
  gsi_.nextSample(convert_);

  // synthesize final output of filterbank
  gsl_vector_float_set_zero(vector_);
  for (unsigned sampX = 0; sampX < Rx2_; sampX++)
    for (unsigned d = 0; d < D_; d++)
      gsl_vector_float_set(vector_, D_ - d - 1, gsl_vector_float_get(vector_, D_ - d - 1) + gsi_.sample(Rx2_ - sampX - 1, d + sampX * D_) / R_);

  return vector_;
}

void PerfectReconstructionFFTSynthesisBank::reset()
{
  samp_->reset();  PerfectReconstructionFilterBank::reset();  VectorFloatFeatureStream::reset();
  buffer_.zero();
}

// ----- definition for class `DelayFeature' -----
//
DelayFeature::DelayFeature(const VectorComplexFeatureStreamPtr& samp, float time_delay, const String& nm )
  : VectorComplexFeatureStream(samp->size(), nm),
    samp_(samp),
    time_delay_(time_delay)
{}

DelayFeature::~DelayFeature()
{}

void DelayFeature::reset()
{
  samp_->reset();
  VectorComplexFeatureStream::reset();
}

const gsl_vector_complex* DelayFeature::next(int frame_no)
{
  if ( frame_no == frame_no_ ) return vector_;

  const gsl_vector_complex* samp = samp_->next(frame_no);
  const gsl_complex alpha = gsl_complex_polar(1.0, time_delay_);

  gsl_vector_complex_memcpy((gsl_vector_complex*)vector_, samp);
  gsl_blas_zscal(alpha, (gsl_vector_complex*)vector_);

  increment_();
  return vector_;
}
