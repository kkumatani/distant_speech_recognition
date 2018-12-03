#include "spectralsubtraction.h"
#include <gsl/gsl_blas.h>
#include <matrix/blas1_c.H>
#include <matrix/linpack_c.H>

PSDEstimator::PSDEstimator(unsigned fftLen2)
{
  estimates_ = gsl_vector_alloc( fftLen2+1 );
  gsl_vector_set_zero( estimates_ );
}

PSDEstimator::~PSDEstimator()
{
  gsl_vector_free( estimates_ );
}

bool PSDEstimator::read_estimates( const String& fn )
{
  FILE *fp = fopen( fn.c_str(), "r" );
  if( NULL == fp ){
    fprintf(stderr,"could not read %s\n", fn.c_str() );
    return false;
  }
  
  for(size_t i=0;i<estimates_->size;i++){
    double val;
    
    fscanf( fp, "%lf\n", &val );
    gsl_vector_set( estimates_, i, val );
  }
  fclose(fp);
  return true;
}

bool PSDEstimator::write_estimates( const String& fn )
{

  FILE *fp = fopen( fn.c_str(), "w" );
  if( NULL == fp ){
    fprintf(stderr,"could not write %s\n", fn.c_str() );
    return false;
  }

  for(size_t i=0;i<estimates_->size;i++){
    fprintf( fp, "%lf\n", gsl_vector_get(estimates_, i ) );
  }
  fclose(fp);
  return true;
}


AveragePSDEstimator::AveragePSDEstimator(unsigned fftLen2, float alpha ):
  PSDEstimator(fftLen2), alpha_(alpha), sample_added_(false)
{
  sampleL_.clear();
}

AveragePSDEstimator::~AveragePSDEstimator()
{
  this->clear_samples();
}

void AveragePSDEstimator::clear()
{
  //fprintf(stderr,"AveragePSDEstimator::clear()\n");
  sample_added_ = false;
  this->clear_samples();
}

const gsl_vector* AveragePSDEstimator::average()
{
  if( alpha_ < 0 ){
    list<gsl_vector *>::iterator itr = sampleL_.begin();
    unsigned sampleN = 0;

    gsl_vector_set_zero( estimates_ );
    while( itr != sampleL_.end() ){
      //gsl_vector_add ( estimates_, *itr );
      gsl_vector_add ( estimates_, (gsl_vector*)*itr );
      sampleN++;
      itr++;
    }

    gsl_vector_scale( estimates_, 1.0/(float)sampleN );
  }
  return (const gsl_vector*)estimates_;
}

bool AveragePSDEstimator::add_sample( const gsl_vector_complex *sample )
{

  size_t fftLen2 = estimates_->size - 1;
  gsl_vector *tmp = gsl_vector_alloc( fftLen2+1 );

  for(size_t i=0;i<=fftLen2;i++){
    gsl_complex val = gsl_vector_complex_get( sample, i );
    gsl_vector_set( tmp, i, gsl_complex_abs2 ( val ) );
  }

  if( alpha_ < 0 ){/* to calc. the average */
    sampleL_.push_back( tmp );
  }
  else{/* for recursive averaging */
    if( sample_added_==false ){
      gsl_vector_memcpy( estimates_, tmp );
      sample_added_ = true;
    }
    else{
      gsl_vector_scale( estimates_, alpha_ );
      gsl_vector_scale( tmp,       (1.0-alpha_) );
      gsl_vector_add( estimates_, tmp );
    }
    gsl_vector_free(tmp);
  }

  //sample_added_ = true;

  return true;
}

void AveragePSDEstimator::clear_samples()
{
  list<gsl_vector *>::iterator itr = sampleL_.begin();
  while( itr != sampleL_.end() ){
    gsl_vector_free( *itr );
    itr++;
  }
  sampleL_.clear();
}

/*
@brief

@param unsigned fftLen
@param bool halfBandShift
@param float ft[in] subtraction coeffient
@param float flooringV
@param const String& nm

*/
SpectralSubtractor::SpectralSubtractor(unsigned fftLen, bool halfBandShift, float ft, float flooringV, const String& nm)
: VectorComplexFeatureStream(fftLen, nm)
{
  fftLen_ = fftLen;
  fftLen2_ = fftLen/2;
  halfBandShift_ = halfBandShift;
  training_started_ = true;
  totalTrainingSampleN_ = 0;
  ft_ = ft;
  flooringV_ = flooringV;
  start_noise_subtraction_ = false;
}

SpectralSubtractor::~SpectralSubtractor()
{
  if(  channelList_.size() > 0 )
    channelList_.erase( channelList_.begin(), channelList_.end() );

  if(  noisePSDList_.size() > 0 )
    noisePSDList_.erase( noisePSDList_.begin(), noisePSDList_.end() );
}

void SpectralSubtractor::reset()
{
  //fprintf(stderr,"SpectralSubtractor::reset\n");

  totalTrainingSampleN_  = 0;

  ChannelIterator_  itr1 = channelList_.begin();
  //NoisePSDIterator_ itr2 = noisePSDList_.begin();
  for(; itr1 != channelList_.end(); itr1++){
    (*itr1)->reset();
    //(*itr2)->reset();
    //itr2++;
  }
  
  VectorComplexFeatureStream::reset();
  //_endOfSample = false;
}

/*
@brief
@param
*/
void SpectralSubtractor::set_channel( VectorComplexFeatureStreamPtr& chan, double alpha )
{
  channelList_.push_back(chan);
  noisePSDList_.push_back( new AveragePSDEstimator(fftLen2_, alpha) );
}

void SpectralSubtractor::stop_training()
{
  training_started_ = false;
  for(NoisePSDIterator_ itr = noisePSDList_.begin(); itr != noisePSDList_.end(); itr++)
    (*itr)->average();
}

void SpectralSubtractor::clear_noise_samples()
{
  for(NoisePSDIterator_ itr = noisePSDList_.begin(); itr != noisePSDList_.end(); itr++)
    (*itr)->clear_samples();
}

bool SpectralSubtractor::read_noise_file(const String& fn, unsigned idx){
  training_started_ = false;
  return noisePSDList_.at(idx)->read_estimates( fn );
}

/*
@brief If the noise PSD model has been trained, which means training_started_ == false, 
       this method performs spectral subtraction on the audio data set by setChannel().  
       Otherwise, this updates the noise PSD model.
@param 
@return 
*/
const gsl_vector_complex* SpectralSubtractor::next(int frame_no)
{
  size_t chanN = channelList_.size();
  ChannelIterator_  itr1 = channelList_.begin();
  NoisePSDIterator_ itr2 = noisePSDList_.begin();
  gsl_complex c1 = gsl_complex_rect(1.0, 0.0);

  gsl_vector_complex_set_zero(vector_ );
  for(size_t i=0;i<chanN;i++,itr1++,itr2++){
	const gsl_vector_complex* samp_i = (*itr1)->next(frame_no);
	
	if( training_started_ == true ){
	  (*itr2)->add_sample( samp_i ); // for computating the variance of the noise PSD
	  totalTrainingSampleN_++;
	}
	if( start_noise_subtraction_ == false ){
	  gsl_blas_zaxpy( c1, samp_i, vector_); // _vector = samp_i + _vector
	}
	else{
	  const gsl_vector* noisePSD = (*itr2)->estimate();
	  gsl_complex Xt, tmp;

#ifdef IGNORE_DC
	  //for fbinX = 0
	  Xt  = gsl_vector_complex_get( samp_i, 0 );
	  tmp = gsl_vector_complex_get(vector_, 0 );
	  gsl_vector_complex_set(vector_, 0, gsl_complex_add( Xt, tmp) );
	  for(unsigned fbinX=1;fbinX<=fftLen2_;fbinX++){
	    Xt  = gsl_vector_complex_get( samp_i, fbinX );
	    tmp = gsl_vector_complex_get(vector_, fbinX );
	    double th = gsl_complex_arg( Xt );
	    double X2 = gsl_complex_abs2( Xt );
	    double N2 = gsl_vector_get( noisePSD, fbinX );
	    double S2 = X2 - ft_ * N2;
	    if( S2 <= flooringV_ ){ // flooring
	      S2 = flooringV_;
	    }
	    gsl_complex St  = gsl_complex_polar( sqrt(S2), th );
	    gsl_complex Stp = gsl_complex_add( St, tmp );
	    gsl_vector_complex_set(vector_, fbinX, Stp );
	    if( fbinX < fftLen2_ )
	      gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(Stp) );
	  }
	  //for fbinX = fftLen2
#else
	  for(unsigned fbinX=0;fbinX<=fftLen2_;fbinX++){
	    Xt  = gsl_vector_complex_get( samp_i, fbinX );
	    tmp = gsl_vector_complex_get(vector_, fbinX );
	    double th = gsl_complex_arg( Xt );
	    double X2 = gsl_complex_abs2( Xt );
	    double N2 = gsl_vector_get( noisePSD, fbinX );
	    double S2 = X2 - ft_ * N2;
	    if( S2 <= flooringV_ ){ // flooring
	      S2 = flooringV_;
	    }
	    gsl_complex St  = gsl_complex_polar( sqrt(S2), th );
	    gsl_complex Stp = gsl_complex_add( St, tmp );
	    gsl_vector_complex_set(vector_, fbinX, Stp );
	    if( fbinX > 0 && fbinX < fftLen2_ )
	      gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(Stp) );
	  }
#endif /* IGNORE_DC */
	  
	}
  }

  gsl_blas_zdscal( 1.0/(double)chanN, vector_);
  increment_();
  return vector_;
}

WienerFilter::WienerFilter( VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift, float alpha, float flooringV, double beta, const String& nm )
  : VectorComplexFeatureStream(targetSignal->size(), nm),
    target_signal_(targetSignal),
    noise_signal_(noiseSignal),
    halfBandShift_(halfBandShift),
    alpha_(alpha),
    flooringV_(flooringV),
    beta_(beta),
    update_noise_PSD_(true)
{
  fftLen_  = target_signal_->size();
  fftLen2_ = fftLen_ / 2 ;

  if( fftLen_ != noise_signal_->size() ){
    throw jdimension_error("Input block length (%d) != fftLen (%d)\n", fftLen_, noise_signal_->size() );
  }

  prev_PSDs_ = gsl_vector_calloc( fftLen2_ + 1 );
  prev_PSDn_ = gsl_vector_calloc( fftLen2_ + 1 );
}

WienerFilter::~WienerFilter()
{
  gsl_vector_free( prev_PSDs_ );
  gsl_vector_free( prev_PSDn_ );
}

const gsl_vector_complex *WienerFilter::next( int frame_no )
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex *St = target_signal_->next( frame_no );
  const gsl_vector_complex *Nt = noise_signal_->next( frame_no );
  double alpha, H, PSDs, PSDn, prevPSDs, prevPSDn, currPSDs, currPSDn;
  gsl_complex val;

  if( frame_no_ > 0 )
    alpha =  alpha_;
  else
    alpha = 0.0;

  if( false == halfBandShift_ ){
    gsl_vector_complex_set(vector_, 0, gsl_vector_complex_get( St, 0) );
    for (unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++) {
      prevPSDs = gsl_vector_get( prev_PSDs_, fbinX );
      currPSDs = gsl_complex_abs2( gsl_vector_complex_get( St, fbinX ) );
      PSDs = alpha * prevPSDs + (1-alpha) * currPSDs;

      prevPSDn = gsl_vector_get( prev_PSDn_, fbinX );
      if( update_noise_PSD_ ){
        currPSDn = gsl_complex_abs2( gsl_vector_complex_get( Nt, fbinX ) );
        if( currPSDn < flooringV_ )
          currPSDn = flooringV_;
        PSDn = alpha * prevPSDn + (1-alpha) * currPSDn;
      }
      else{
        PSDn = prevPSDn;
      }

      H = PSDs / ( PSDs + beta_ * PSDn );
      val = gsl_complex_mul_real( gsl_vector_complex_get( St, fbinX ), H );
      gsl_vector_complex_set(vector_, fbinX, val );
      gsl_vector_set( prev_PSDs_, fbinX, PSDs );
      if( update_noise_PSD_ )
        gsl_vector_set( prev_PSDn_, fbinX, PSDn );
      if( fbinX == fftLen2_ )
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
    }
  }
  else{
    throw  j_error("WienerFilter::next() for the half band shift is not implemented\n");
  }

  increment_();
  return vector_;
}

void WienerFilter::reset()
{
  target_signal_->reset();
  noise_signal_->reset();
}

