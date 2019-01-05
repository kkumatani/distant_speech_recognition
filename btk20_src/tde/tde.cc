/*
 * @file tde.cc
 * @brief Time delay estimation
 * @author Kenichi Kumatani
 */

#include "tde.h"
#include <gsl/gsl_blas.h>
#include <matrix/blas1_c.h>
#include <matrix/linpack_c.h>


static unsigned get_fft_len( unsigned tmpi )
{/* calculate the FFT  length */
  unsigned int pp = 1; // 2^i
  for(unsigned i=0; pp < tmpi;i++){
    pp = pp * 2;
  }

  return pp;
}

/*
  @brief calculate the time delay of arrival (TDOA) between two sounds.
  @param const String& fn1[in] file name of the first sound
  @param const String& fn2[in] file name of the second sound
  @param unsigned nHeldMaxCC[in] the number of the max cross-correlation values held during the process
*/
CCTDE::CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, int fftLen, unsigned nHeldMaxCC, int freqLowerLimit, int freqUpperLimit, const String& nm )
:VectorFeatureStream(nHeldMaxCC, nm),
 nHeldMaxCC_(nHeldMaxCC),
 freq_lower_limit_(freqLowerLimit),
 freq_upper_limit_(freqUpperLimit)
{
  sample_delays_ = new unsigned[nHeldMaxCC_];
  cc_values_     = new double[nHeldMaxCC_];

  if( samp1->getSampleRate() != samp2->getSampleRate() ){
    throw jdimension_error("The sampling rates must be the same but %d != %d\n", samp1->getSampleRate(), samp2->getSampleRate() );
  }
  samplerate_ = samp1->getSampleRate();

  if( samp1->size() != samp2->size() ){
    throw jdimension_error("Block sizes must be the same but %d != %d \n",samp1->size(),samp2->size() );
  }
  unsigned tmpi = (samp1->size()>samp2->size())? samp1->size():samp2->size();
  fftLen_ = get_fft_len( tmpi );

  if( nHeldMaxCC_ >= fftLen_ ){
    throw jdimension_error("The number of the held cross-correlation coefficients should be less than the FFT length but %d > %d \n", nHeldMaxCC_, fftLen_ );
  }
  window_ = get_window( 2, fftLen_ ); // Hanning window
  channelL_.push_back( samp1 );
  channelL_.push_back( samp2 );
  _frameCounter.push_back( 0 );
  _frameCounter.push_back( 0 );
}

CCTDE::~CCTDE()
{
  if(  (int)channelL_.size() > 0 )
    channelL_.erase( channelL_.begin(), channelL_.end() );

  delete [] sample_delays_;
  delete [] cc_values_;

  gsl_vector_free( window_ );
}

void  CCTDE::allsamples( int fftLen )
{
  size_t stride = 1;
  size_t chanN = channelL_.size();

  if( fftLen < 0 ){
    ChannelIterator_ itr = channelL_.begin();
    unsigned samplesMax = (*itr)->samplesN();
    itr++;
    while( itr != channelL_.end() ) {// loop for 2 sources
      if( (*itr)->samplesN() > samplesMax )
        samplesMax = (*itr)->samplesN();
      itr++;
    }
    fftLen_ = get_fft_len( samplesMax );
  }
  else
    fftLen_ = (unsigned)fftLen;

  if( NULL != window_ )
    delete [] window_;
  window_ = get_window( 2, fftLen_ );

  double **samples = (double **)malloc( chanN * sizeof(double *));
  if( NULL == samples ){
    throw j_error("cannot allocate memory\n");
  }

  ChannelIterator_ itr = channelL_.begin();
  for(unsigned i=0; itr != channelL_.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;
    unsigned blockLen;

    samples[i] = (double *)calloc(fftLen_,sizeof(double));
    if( samples[i] == NULL ){
      throw j_error("cannot allocate memory\n");
    }
    block = (*itr)->data();
    blockLen = (*itr)->samplesN();
    for(unsigned j =0;j<fftLen_;j++){
      if( j >= blockLen )
        break;
      samples[i][j] = gsl_vector_get(window_,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, fftLen_ );// FFT for real data
  }

  this->detect_cc_peaks_( samples, stride );

  for (unsigned i=0;i<2;i++)
    free( samples[i] );
  free( samples );
}

/*
  @note re-write sample_delays_, cc_values_ and _vector.
*/
const gsl_vector* CCTDE::detect_cc_peaks_( double **samples, size_t stride )
{

  double *ccA = new double[2*fftLen_];

  { // calculate the CC function in the frequency domain
#define myREAL(z,i) ((z)[2*(i)])
#define myIMAG(z,i) ((z)[2*(i)+1])
    double hc_r[2], hc_i[2];
    double val;

    hc_r[0] = samples[0][0];
    hc_i[0] = 0.0;
    hc_r[1] = samples[1][0];
    hc_i[1] = 0.0;
    val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
    myREAL( ccA, 0 ) = cos(val);
    myIMAG( ccA, 0 ) = sin(val);
    for(unsigned j =1;j<fftLen_/2;j++){
      hc_r[0] = samples[0][j*stride];
      hc_i[0] = samples[0][(fftLen_-j)*stride];
      hc_r[1] = samples[1][j*stride];
      hc_i[1] = samples[1][(fftLen_-j)*stride];

      val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
      myREAL( ccA, j ) = cos(val);
      myIMAG( ccA, j ) = sin(val);

      val = atan2( -hc_i[1], hc_r[1] ) - atan2( -hc_i[0], hc_r[0] );
      myREAL( ccA, (fftLen_ - j)*stride ) = cos(val);
      myIMAG( ccA, (fftLen_ - j)*stride ) = sin(val);
    }
    hc_r[0] = samples[0][(fftLen_/2)*stride];
    hc_i[0] = 0.0;
    hc_r[1] = samples[1][(fftLen_/2)*stride];
    hc_i[1] = 0.0;
    val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
    myREAL( ccA, fftLen_/2 ) = cos(val);
    myIMAG( ccA, fftLen_/2 ) = sin(val);

    {// discard a band
      if( freq_upper_limit_ <= 0)
        freq_upper_limit_ = samplerate_ / 2;
      if( freq_lower_limit_ >= 0 && freq_upper_limit_ <= 0 ){

        int s1 = (int)(freq_lower_limit_ * fftLen_ / (float)samplerate_ );
        int e1 = (int)(freq_upper_limit_ * fftLen_ / (float)samplerate_ );
        for(int i=1;i<=s1;i++){
          myREAL( ccA, i ) = 0.0;
          myIMAG( ccA, i ) = 0.0;
          myREAL( ccA, fftLen_ - 1 - i ) = 0.0;
          myIMAG( ccA, fftLen_ - 1 - i ) = 0.0;
        }
        for(int i=e1;i<(int)fftLen_/2;i++){
          myREAL( ccA, i ) = 0.0;
          myIMAG( ccA, i ) = 0.0;
          myREAL( ccA, fftLen_ - 1 - i ) = 0.0;
          myIMAG( ccA, fftLen_ - 1 - i ) = 0.0;
        }
      }
    }
    gsl_fft_complex_radix2_inverse( ccA, stride, fftLen_ );// with scaling

  }
  {/* detect nHeldMaxCC_ peaks */
    unsigned *maxArgs = sample_delays_;
    double   *maxVals = cc_values_; /* maxVals[0] > maxVals[1] > maxVals[2] ... */

    maxArgs[0] = 0;
    maxVals[0] = myREAL( ccA, 0 );
    for(unsigned i1=1;i1<nHeldMaxCC_;i1++){
      maxArgs[i1]  = -1;
      maxVals[i1] = -10e10;
    }
    for(unsigned i=1;i<fftLen_;i++){
      double cc = myREAL( ccA, i );

      if( cc > maxVals[nHeldMaxCC_-1] ){
	for(unsigned i1=0;i1<nHeldMaxCC_;i1++){
	  if( cc >= maxVals[i1] ){
	    for(unsigned j=nHeldMaxCC_-1;j>i1;j--){
	      maxVals[j] = maxVals[j-1];
	      maxArgs[j] = maxArgs[j-1];
	    }
	    maxVals[i1] = cc;
	    maxArgs[i1] = i;
	    break;
	  }
	}
      }
    }

    //set time delays to _vector
    printf("# Nth candidate : delay (sample) : delay (msec) : CC\n"); fflush(stdout);
    for(unsigned i=0;i<nHeldMaxCC_;i++){
      int sampleDelay;
      float timeDelay;

      if( maxArgs[i] < fftLen_/2 ){
        timeDelay   = maxArgs[i] * 1.0 / samplerate_;
        sampleDelay = maxArgs[i];
      }
      else{
        timeDelay   = - ( (float)fftLen_ - maxArgs[i] ) * 1.0 / samplerate_;
        sampleDelay = - ( fftLen_ - maxArgs[i] );
      }
      gsl_vector_set(vector_, i, timeDelay );
      printf("%d : %d : %f : %f\n",i, sampleDelay, timeDelay * 1000, maxVals[i] ); fflush(stdout);
    }
#undef myREAL
#undef myIMAG
  }
  
  delete [] ccA;
  return vector_;
}

/**
   @brief
   @param
   @return TDOAs btw. two signals (sec).
*/
const gsl_vector* CCTDE::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  size_t stride = 1;

  double **samples = (double **)malloc( channelL_.size() * sizeof(double *));
  if( NULL == samples ){
    throw j_error("cannot allocate memory\n");
  }

  ChannelIterator_ itr = channelL_.begin();
  for(unsigned i=0; itr != channelL_.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;

    block = (*itr)->next(frame_no);
    samples[i] = (double *)calloc(fftLen_,sizeof(double));
    unsigned blockLen = (*itr)->size();
    for(unsigned j =0;j<fftLen_;j++){
      if( j >= blockLen )
        break;
      samples[i][j] = gsl_vector_get(window_,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, fftLen_ );// FFT for real data
  }

  this->detect_cc_peaks_( samples, stride );
  for (unsigned i=0;i<channelL_.size();i++)
    free( samples[i] );
  free( samples );

  increment_();
  return vector_;
}

const gsl_vector* CCTDE::nextX( unsigned chanX, int frame_no )
{
  size_t stride = 1;

  double **samples = (double **)malloc( channelL_.size() * sizeof(double *));
  if( NULL == samples ){
    throw j_error("cannot allocate memory\n");
  }

  ChannelIterator_ itr = channelL_.begin();
  for(unsigned i=0; itr != channelL_.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;

    if( i != chanX )
      block = (*itr)->current();
    else
      block = (*itr)->next(frame_no);
    samples[i] = (double *)calloc(fftLen_,sizeof(double));
    unsigned blockLen = (*itr)->size();
    for(unsigned j =0;j<fftLen_;j++){
      if( j >= blockLen )
        break;
      samples[i][j] = gsl_vector_get(window_,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, fftLen_ );// FFT for real data
  }

  this->detect_cc_peaks_( samples, stride );
  for (unsigned i=0;i<channelL_.size();i++)
    free( samples[i] );
  free( samples );

  if( chanX == 0 )
    increment_();
  _frameCounter[chanX]++;

  return vector_;
}

void CCTDE::reset()
{
  for (ChannelIterator_ itr = channelL_.begin(); itr != channelL_.end(); itr++) {
    (*itr)->reset();
  }
  VectorFeatureStream::reset();
}
