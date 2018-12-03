/*
 * @file binauralprocessing.cc
 * @brief Binaural processing
 * @author Kenichi Kumatani
 */
#include <math.h>
#include "postfilter/binauralprocessing.h"

/*
  @brief calculate the interaural time delay (ITD) at each frequency bin
  @param fbinX[in] the index of the frequency component
  @param fftLen[in]
  @param X_L_f[in]
  @param X_R_f[in]
  @return ITD
*/
static double calcITDf( unsigned fbinX, unsigned fftLen, gsl_complex X_L_f, gsl_complex X_R_f )
{
  double ITD; /* => |d_{s^*[m,k][m,k]}| Eq. (4) */
  double ad_X_LAngle_f = gsl_complex_arg( X_L_f );
  double ad_X_RAngle_f = gsl_complex_arg( X_R_f );
  double adPhaseDiff1  = fabs( ad_X_LAngle_f - ad_X_RAngle_f );
  double adPhaseDiff2  = fabs( ad_X_LAngle_f - ad_X_RAngle_f - 2 * M_PI );
  double adPhaseDiff3  = fabs( ad_X_LAngle_f - ad_X_RAngle_f + 2 * M_PI );
  double adPhaseDiff; /* => |w_k| * |d_{s^*[m,k][m,k]}| Eq.  */
  
  if( adPhaseDiff1 < adPhaseDiff2 ){
    adPhaseDiff = adPhaseDiff1;
  }
  else{
    adPhaseDiff = adPhaseDiff2;
  }
  if( adPhaseDiff3 < adPhaseDiff )
    adPhaseDiff = adPhaseDiff3;

  ITD = adPhaseDiff / ( 2 * M_PI * fbinX / fftLen );
  return ITD;
}

// ----- definition for class 'BinaryMaskFilter' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
*/
BinaryMaskFilter::BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
				    float threshold, float alpha, float dEta , const String& nm ):
  VectorComplexFeatureStream(M, nm),
  srcL_(srcL),
  srcR_(srcR),
  chanX_(chanX),
  alpha_(alpha),
  dEta_(dEta),
  threshold_(threshold),
  threshold_per_freq_(NULL)
{
  if( srcL->size() != M ){
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M);
  }

  if( srcR->size() != M ){
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M);
  }

  prevMu_ = gsl_vector_float_alloc(M/2+1);
  gsl_vector_float_set_all(prevMu_, 1.0);
}

BinaryMaskFilter::~BinaryMaskFilter()
{
  gsl_vector_float_free( prevMu_ );
  if( NULL != threshold_per_freq_ )
    gsl_vector_free( threshold_per_freq_ );
}

void BinaryMaskFilter::set_thresholds( const gsl_vector *thresholds )
{
  unsigned fftLen2 = (unsigned)srcL_->size()/2;

  if( NULL != threshold_per_freq_ ){
    for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
      gsl_vector_set( threshold_per_freq_, fbinX, gsl_vector_get( thresholds, fbinX ) );
    }
  }
  else{
    threshold_per_freq_ = gsl_vector_alloc( fftLen2+1 );
  }

}

const gsl_vector_complex* BinaryMaskFilter::next(int frame_no)
{
  increment_();
  return vector_;
}

void BinaryMaskFilter::reset()
{
  srcL_->reset();
  srcR_->reset();
  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- definition for class 'KimBinaryMaskFilter' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
KimBinaryMaskFilter::KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
					 float threshold, float alpha, float dEta , float dPowerCoeff, const String& nm ):
  BinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm ),
  dpower_coeff_(dPowerCoeff)
{
}

KimBinaryMaskFilter::~KimBinaryMaskFilter()
{
}

/**
   @brief perform binary masking which picks up the left channel when the ITD <= threshold
*/
const gsl_vector_complex* KimBinaryMaskFilter::masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold )
{
  unsigned fftLen  = (unsigned)ad_X_L->size;
  unsigned fftLen2 = fftLen / 2;
  gsl_complex val;

  // Direct component : fbinX = 0
  val = gsl_vector_complex_get( ad_X_L, 0 );
  gsl_vector_complex_set(vector_, 0, val );

  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    float  mu;
    double ITD = calcITDf( fbinX, fftLen,
			   gsl_vector_complex_get( ad_X_L, fbinX ),
			   gsl_vector_complex_get( ad_X_R, fbinX ) );
    if( chanX_ == 0 ){
      if( ITD <= threshold_ )
	mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ );
      else
	mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
      val = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_L, fbinX ), mu );
    }
    else{
      if( ITD <= threshold_ )
	mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
      else
	mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ );
      val = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_R, fbinX ), mu );
    }

    if( fbinX < fftLen2 ){
      gsl_vector_complex_set(vector_, fbinX, val);
      gsl_vector_complex_set(vector_, fftLen - fbinX, gsl_complex_conjugate(val) );
    }
    else
      gsl_vector_complex_set(vector_, fftLen2, val);
    
    gsl_vector_float_set( prevMu_, fbinX, mu );
  }

  return vector_;
}

const gsl_vector_complex* KimBinaryMaskFilter::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;

  if( frame_no >= 0 ){
    ad_X_L = srcL_->next(frame_no);
    ad_X_R = srcR_->next(frame_no);
  }
  else{
    if( frame_no_ == frame_reset_no_ ){
      ad_X_L = srcL_->next(0);
      ad_X_R = srcR_->next(0);
    }
    else{
      ad_X_L = srcL_->next(frame_no_+1);
      ad_X_R = srcR_->next(frame_no_+1);
    }
  }
  masking1( ad_X_L, ad_X_R, threshold_ );
  
  increment_();
  return vector_;
}

void KimBinaryMaskFilter::reset()
{
  srcL_->reset();
  srcR_->reset();
  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- definition for class 'KimITDThresholdEstimator' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float minThreshold
   @param float maxThreshold
   @param float width
   @param float minFreq
   @param float maxFreq
   @param int sampleRate
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
KimITDThresholdEstimator::KimITDThresholdEstimator(VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
						   float minThreshold, float maxThreshold, float width,
						   float minFreq, float maxFreq, int sampleRate, 
						   float dEta, float dPowerCoeff, const String& nm ):
  KimBinaryMaskFilter( 0, srcL, srcR, M, 0.0, 0.0 /* alha must be zero */, dEta, dPowerCoeff ),
  width_(width),
  nCand_(0),
  cost_func_computed_(false)
{
  if( srcL->size() != M ){
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M);
  }

  if( srcR->size() != M ){
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M);
  }

  if( minThreshold == maxThreshold ){
    min_threshold_ = - 0.2 * 16000 / 340;
    max_threshold_ =   0.2 * 16000 / 340;
  }
  else{
    min_threshold_ = minThreshold;
    max_threshold_ = maxThreshold;
  }

  if( minFreq < 0 || maxFreq < 0 || sampleRate < 0 ){
    min_fbinX_ = 1;
    max_fbinX_ = M/2 + 1;
  }
  else{
    min_fbinX_ = (unsigned) ( M * minFreq / (float)sampleRate );
    max_fbinX_ = (unsigned) ( M * maxFreq / (float)sampleRate );
  }

  int nCand = (int)( (max_threshold_-min_threshold_)/width + 1.5 );
  nCand_ = (unsigned int)nCand;

  cost_func_values_ = (double *)calloc(nCand,sizeof(double));
  sigma_T_ = (double *)calloc(nCand,sizeof(double));
  sigma_I_ = (double *)calloc(nCand,sizeof(double));
  mean_T_  = (double *)calloc(nCand,sizeof(double));
  mean_I_  = (double *)calloc(nCand,sizeof(double));
  if( NULL==cost_func_values_ ||
      NULL==sigma_T_ || NULL==sigma_I_ ||
      NULL==mean_T_  || NULL==mean_I_  ){
    throw jallocation_error("KimITDThresholdEstimator:cannot allocate memory\n");
  }

  buffer_ = gsl_vector_alloc( nCand_ );

  nSamples_ = 0;
}

KimITDThresholdEstimator::~KimITDThresholdEstimator()
{
  free( cost_func_values_ );
  free( sigma_T_ );
  free( sigma_I_ );
  free( mean_T_ );
  free( mean_I_ );
  if( NULL != buffer_ ){
    gsl_vector_free( buffer_ );
    buffer_ = NULL;
  }
}

const gsl_vector* KimITDThresholdEstimator::cost_function()
{
  if( cost_func_computed_ == false ){
    fprintf(stderr,"KimITDThresholdEstimator:call calc_threshold() first\n");
  }
  for(unsigned int iSearchX=0;iSearchX<nCand_;iSearchX++){
    gsl_vector_set( buffer_, iSearchX, cost_func_values_[iSearchX] );
  }

  return (const gsl_vector*)buffer_;
}

void KimITDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
  unsigned fftLen = (unsigned)ad_X_L->size;
  gsl_complex X_T, X_I;
  double      mu_T, mu_I;
  unsigned    iSearchX=0;

  for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
    double P_T = 0.0, R_T;
    double P_I = 0.0, R_I;

    for(unsigned fbinX=min_fbinX_;fbinX<max_fbinX_;fbinX++){
      double ITD = calcITDf(fbinX, fftLen,
			    gsl_vector_complex_get( ad_X_L, fbinX ),
			    gsl_vector_complex_get( ad_X_R, fbinX ));
      if( ITD <= threshold ){
	mu_T = 1.0;
	mu_I = dEta_;
      }
      else{
	mu_T = dEta_;
	mu_I = 1.0;
      }
      X_T = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_L, fbinX ), mu_T );
      X_I = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_R, fbinX ), mu_I );
      P_T += gsl_complex_abs2( X_T );
      P_I += gsl_complex_abs2( X_I );
    }
    R_T = (double)pow( (double)P_T, (double)dpower_coeff_ );
    R_I = (double)pow( (double)P_I, (double)dpower_coeff_ );
    cost_func_values_[iSearchX] += R_T * R_I;
    mean_T_[iSearchX]  += R_T;
    mean_I_[iSearchX]  += R_I;
    sigma_T_[iSearchX] += R_T * R_T;
    sigma_I_[iSearchX] += R_I * R_I;
  }
  
  nSamples_++;
  return;
}

const gsl_vector_complex* KimITDThresholdEstimator::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;

  if( frame_no >= 0 ){
    ad_X_L = srcL_->next(frame_no);
    ad_X_R = srcR_->next(frame_no);
  }
  else{
    if( frame_no_ == frame_reset_no_ ){
      ad_X_L = srcL_->next(0);
      ad_X_R = srcR_->next(0);
    }
    else{
      ad_X_L = srcL_->next(frame_no_+1);
      ad_X_R = srcR_->next(frame_no_+1);
    }
  }
  accumStats1( ad_X_L, ad_X_R );
  
  increment_();
  return vector_;
}

double KimITDThresholdEstimator::calc_threshold()
{
  float argMin = min_threshold_;
  double min_rho = 1000000;
  unsigned iSearchX=0;

  for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
    double rho;

    mean_T_[iSearchX]  /= nSamples_;
    mean_I_[iSearchX]  /= nSamples_;
    sigma_T_[iSearchX] = ( sigma_T_[iSearchX] / nSamples_ ) - mean_T_[iSearchX] * mean_T_[iSearchX];
    sigma_I_[iSearchX] = ( sigma_I_[iSearchX] / nSamples_ ) - mean_I_[iSearchX] * mean_I_[iSearchX];
    cost_func_values_[iSearchX] /= nSamples_;
    rho = fabs( ( cost_func_values_[iSearchX] - mean_T_[iSearchX] *mean_I_[iSearchX] ) / ( sqrt( sigma_T_[iSearchX] ) * sqrt( sigma_I_[iSearchX] ) ) );
    //fprintf(stderr,"%f %f\n",threshold,rho);
    if( rho < min_rho ){
      argMin = threshold;
      min_rho = rho;
    }
  }

  threshold_ = argMin;
  cost_func_computed_ = true;
  return argMin;
}

void KimITDThresholdEstimator::reset()
{
  srcL_->reset();
  srcR_->reset();
  VectorComplexFeatureStream::reset();
  is_end_ = false;
  
  unsigned iSearchX=0;
  for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
    cost_func_values_[iSearchX] = 0.0;
    mean_T_[iSearchX]  = 0.0;
    mean_I_[iSearchX]  = 0.0;
    sigma_T_[iSearchX] = 0.0;
    sigma_I_[iSearchX] = 0.0;
  }
  nSamples_ = 0;
  cost_func_computed_ = false;
}

// ----- definition for class 'IIDBinaryMaskFilter' -----
// 

IIDBinaryMaskFilter::IIDBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
										 float threshold, float alpha, float dEta, const String& nm ):
BinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm )
{
}

IIDBinaryMaskFilter::~IIDBinaryMaskFilter()
{
}

const gsl_vector_complex* IIDBinaryMaskFilter::masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold )
{
  unsigned fftLen  = (unsigned)ad_X_L->size;
  unsigned fftLen2 = fftLen / 2;
  gsl_complex val;
  
  // Direct component : fbinX = 0
  val = gsl_vector_complex_get( ad_X_L, 0 );
  gsl_vector_complex_set(vector_, 0, val );
  
  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    gsl_complex X_T, X_I;
    double P_T, P_I;
    float  mu;
    
    if( NULL != threshold_per_freq_ )
      threshold_ = gsl_vector_get( threshold_per_freq_, fbinX );
    
    if( chanX_ == 0 ){ /* the left channel contains the stronger target signal */
      X_T = gsl_vector_complex_get( ad_X_L, fbinX );
      X_I = gsl_vector_complex_get( ad_X_R, fbinX );
    }
    else{
      X_T = gsl_vector_complex_get( ad_X_R, fbinX );
      X_I = gsl_vector_complex_get( ad_X_L, fbinX );
    }
    P_T = gsl_complex_abs( X_T );
    P_I = gsl_complex_abs( X_I );
    
    if( P_T <= ( P_I + threshold_ ) )
      mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
    else
      mu = alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) ;
    val = gsl_complex_mul_real ( X_T, mu );
    if( fbinX < fftLen2 ){
      gsl_vector_complex_set(vector_, fbinX, val);
      gsl_vector_complex_set(vector_, fftLen - fbinX, gsl_complex_conjugate(val) );
    }
    else
      gsl_vector_complex_set(vector_, fftLen2, val);
    gsl_vector_float_set( prevMu_, fbinX, mu );
  }
  
  return vector_;
}

const gsl_vector_complex* IIDBinaryMaskFilter::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  
  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;
  
  if( frame_no >= 0 ){
    ad_X_L = srcL_->next(frame_no);
    ad_X_R = srcR_->next(frame_no);
  }
  else{
    if( frame_no_ == frame_reset_no_ ){
      ad_X_L = srcL_->next(0);
      ad_X_R = srcR_->next(0);
    }
    else{
      ad_X_L = srcL_->next(frame_no_+1);
      ad_X_R = srcR_->next(frame_no_+1);
    }
  }
  masking1( ad_X_L, ad_X_R, threshold_ );
	
  increment_();
  return vector_;
}

void IIDBinaryMaskFilter::reset()
{
  srcL_->reset();
  srcR_->reset();
  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- definition for class 'IIDThresholdEstimator' -----
// 

IIDThresholdEstimator::IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
											  float minThreshold, float maxThreshold, float width,
											 float minFreq, float maxFreq, int sampleRate, float Eta, float dPowerCoeff, const String& nm ):
  KimITDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, Eta, dPowerCoeff, nm ),
  _beta(3.0)
{
  _Y4_T = (double *)calloc(nCand_,sizeof(double));
  _Y4_I = (double *)calloc(nCand_,sizeof(double));

  if( NULL==_Y4_T || NULL==_Y4_I  ){
    throw jallocation_error("IIDThresholdEstimator:cannot allocate memory\n");
  }
}

IIDThresholdEstimator::~IIDThresholdEstimator()
{
  free(_Y4_T);
  free(_Y4_I);
}

/*
  @brief calculate kurtosis of beamformer's outputs
*/
void IIDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
	unsigned    iSearchX=0;
	
	//fprintf(stderr,"IIDThresholdEstimator::accumStats1 %d\n",frame_no_);
	for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
        	double Y1_T = 0.0, Y2_T = 0.0, Y4_T = 0.0;
		double Y1_I = 0.0, Y2_I = 0.0, Y4_I = 0.0;
		
		for(unsigned fbinX=min_fbinX_;fbinX<max_fbinX_;fbinX++){
			gsl_complex X_T_f,  X_I_f;
			double      P_T_f,  P_I_f;
			double      mu_T,   mu_I;
			double      Y1_T_f, Y1_I_f; /* the magnitude of the binary masked value */
			double      Y2_T_f, Y2_I_f; /* the power of the binary masked value */

			X_T_f = gsl_vector_complex_get( ad_X_L, fbinX );
			X_I_f = gsl_vector_complex_get( ad_X_R, fbinX );
			P_T_f = gsl_complex_abs( X_T_f );
			P_I_f = gsl_complex_abs( X_I_f );
			
			if( P_T_f <= ( P_I_f + threshold ) )
				mu_T = dEta_; //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
			else
				mu_T = 1.0;   //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) ;
			
			if( P_I_f <= ( P_T_f + threshold ) )
				mu_I = dEta_; //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
			else
				mu_I = 1.0;   //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) ;
			
			Y1_T_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) ), (double)2.0*dpower_coeff_ );
			Y1_I_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) ), (double)2.0*dpower_coeff_ );
			//Y1_T_f = gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) );
			//Y1_I_f = gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) );
			Y2_T_f = Y1_T_f * Y1_T_f;
			Y2_I_f = Y1_I_f * Y1_I_f;
			Y1_T += Y1_T_f;
			Y1_I += Y1_I_f;
			Y2_T += Y2_T_f;
			Y2_I += Y2_I_f;
			Y4_T += Y2_T_f * Y2_T_f;
			Y4_I += Y2_I_f * Y2_I_f;
		}
		//cost_func_values_[iSearchX] += Y4_T + Y4_I;
		_Y4_T[iSearchX]    += Y4_T;
		_Y4_I[iSearchX]    += Y4_I;
		mean_T_[iSearchX]  += Y1_T;
		mean_I_[iSearchX]  += Y1_I;
		sigma_T_[iSearchX] += Y2_T;
		sigma_I_[iSearchX] += Y2_I;
	}
	
	nSamples_++;
	return;
}

const gsl_vector_complex* IIDThresholdEstimator::next(int frame_no)
{
	if (frame_no == frame_no_) return vector_;
	
	const gsl_vector_complex* ad_X_L;
	const gsl_vector_complex* ad_X_R;
	
	if( frame_no >= 0 ){
		ad_X_L = srcL_->next(frame_no);
		ad_X_R = srcR_->next(frame_no);
	}
	else{
		if( frame_no_ == frame_reset_no_ ){
			ad_X_L = srcL_->next(0);
			ad_X_R = srcR_->next(0);
		}
		else{
			ad_X_L = srcL_->next(frame_no_+1);
			ad_X_R = srcR_->next(frame_no_+1);
		}
	}
	accumStats1( ad_X_L, ad_X_R );
	
	increment_();
	return vector_;
}

double IIDThresholdEstimator::calc_threshold()
{
  float argMin = min_threshold_;
  double min_rho = 1000000;
  unsigned iSearchX=0;

  for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
    double rho, sig2;

    mean_T_[iSearchX]  /= nSamples_;
    mean_I_[iSearchX]  /= nSamples_;
    sigma_T_[iSearchX] /= nSamples_;
    sigma_I_[iSearchX] /= nSamples_;
    _Y4_T[iSearchX]    /= nSamples_;
    _Y4_I[iSearchX]    /= nSamples_;
    sig2 = sigma_T_[iSearchX] + sigma_I_[iSearchX];
    cost_func_values_[iSearchX] = ( _Y4_T[iSearchX] + _Y4_I[iSearchX] ) - _beta * sig2 * sig2;
    rho = - cost_func_values_[iSearchX]; /* negative kurtosis */
    //fprintf(stderr,"%f %e\n",threshold,rho);
    if( rho < min_rho ){
      argMin = threshold;
      min_rho = rho;
    }
  }

  threshold_ = argMin;
  cost_func_computed_ = true;
  return argMin;
}

void IIDThresholdEstimator::reset()
{
	srcL_->reset();
	srcR_->reset();
	VectorComplexFeatureStream::reset();
	is_end_ = false;
	
	unsigned iSearchX=0;
	for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
		cost_func_values_[iSearchX] = 0.0;
		_Y4_T[iSearchX] = 0.0;
		_Y4_I[iSearchX] = 0.0;
		mean_T_[iSearchX]  = 0.0;
		mean_I_[iSearchX]  = 0.0;
		sigma_T_[iSearchX] = 0.0;
		sigma_I_[iSearchX] = 0.0;
	}
	nSamples_ = 0;
	cost_func_computed_ = false;
}


// ----- definition for class 'FDIIDThresholdEstimator' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float minThreshold
   @param float maxThreshold
   @param float width
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
FDIIDThresholdEstimator::FDIIDThresholdEstimator(VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
						 float minThreshold, float maxThreshold, float width,
						 float dEta, float dPowerCoeff, const String& nm ):
  BinaryMaskFilter( 0, srcL, srcR, M, 0.0, 0.0 /* alha must be zero */, dEta ),
  width_(width),
  dpower_coeff_(dPowerCoeff),
  nCand_(0),
  cost_func_computed_(false)
{
  if( srcL->size() != M ){
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M );
  }

  if( srcR->size() != M ){
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M );
  }

  if( minThreshold == maxThreshold ){
    min_threshold_ = - 100000;
    max_threshold_ =   100000;
  }
  else{
    min_threshold_ = minThreshold;
    max_threshold_ = maxThreshold;
  }

  unsigned fftLen2 = M/2;
  int nCand = (int)( (max_threshold_-min_threshold_)/width + 1.5 );
  nCand_ = (unsigned int)nCand;

  cost_func_values_ = (double **)malloc((fftLen2+1)*sizeof(double *));
  _Y4    = (double **)malloc((fftLen2+1)*sizeof(double *));
  _sigma = (double **)malloc((fftLen2+1)*sizeof(double *));
  _mean  = (double **)malloc((fftLen2+1)*sizeof(double *));
  if( NULL==cost_func_values_ || NULL==_Y4 || NULL==_sigma || NULL==_mean ){
    throw jallocation_error("FDIIDThresholdEstimator:cannot allocate memory\n");
  }

  cost_func_values_[0] = (double *)calloc((fftLen2+1)*nCand_,sizeof(double));
  _Y4[0]    = (double *)calloc((fftLen2+1)*nCand_,sizeof(double));
  _sigma[0] = (double *)calloc((fftLen2+1)*nCand_,sizeof(double));
  _mean[0]  = (double *)calloc((fftLen2+1)*nCand_,sizeof(double));
  if( NULL==cost_func_values_[0] || NULL==_Y4[0] || NULL==_sigma[0] || NULL==_mean[0] ){
    throw jallocation_error("FDIIDThresholdEstimator:cannot allocate memory\n");
  }
  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
    cost_func_values_[fbinX] = &cost_func_values_[0][fbinX*nCand_];
    _Y4[fbinX]    = &(_Y4[0][fbinX*nCand_]);
    _sigma[fbinX] = &(_sigma[0][fbinX*nCand_]);
    _mean[fbinX]  = &(_mean[0][fbinX*nCand_]);
  }

  buffer_ = gsl_vector_alloc( nCand_ );

  threshold_per_freq_ = gsl_vector_alloc( fftLen2+1 );

  nSamples_ = 0;
}

FDIIDThresholdEstimator::~FDIIDThresholdEstimator()
{
  free( cost_func_values_[0] );
  free( _Y4[0] );
  free( _sigma[0] );
  free( _mean[0] );
  free( cost_func_values_ );
  free( _Y4 );
  free( _sigma );
  free( _mean );
  if( NULL != buffer_ ){
    gsl_vector_free( buffer_ );
    buffer_ = NULL;
  }
  if( NULL != threshold_per_freq_ ){
    gsl_vector_free( threshold_per_freq_ );
    threshold_per_freq_ = NULL;
  }
}

const gsl_vector* FDIIDThresholdEstimator::cost_function( unsigned freqX )
{
  if( cost_func_computed_ == false ){
    calc_threshold();
  }
  for(unsigned int iSearchX=0;iSearchX<nCand_;iSearchX++){
    gsl_vector_set( buffer_, iSearchX, cost_func_values_[freqX][iSearchX] );
  }

  return buffer_;
}

/*
  @brief calculate kurtosis of beamformer's outputs
*/
void FDIIDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
  unsigned fftLen2  = (unsigned)ad_X_L->size / 2;
	
  //fprintf(stderr,"IIDThresholdEstimator::accumStats1 %d\n",frame_no_);
  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    unsigned iSearchX=0;
    for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
      gsl_complex X_T_f,  X_I_f;
      double      P_T_f,  P_I_f;
      double      mu_T,   mu_I;
      double      Y1_T_f, Y1_I_f; /* the magnitude of the binary masked value */
      double      Y2_T_f, Y2_I_f; /* the power of the binary masked value */
      
      X_T_f = gsl_vector_complex_get( ad_X_L, fbinX );
      X_I_f = gsl_vector_complex_get( ad_X_R, fbinX );
      P_T_f = gsl_complex_abs( X_T_f );
      P_I_f = gsl_complex_abs( X_I_f );
      
      if( P_T_f <= ( P_I_f + threshold ) )
	mu_T = dEta_; //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
      else
	mu_T = 1.0;   //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) ;
      
      if( P_I_f <= ( P_T_f + threshold ) )
	mu_I = dEta_; //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) * dEta_;
      else
	mu_I = 1.0;   //alpha_ * gsl_vector_float_get( prevMu_, fbinX ) + ( 1 - alpha_ ) ;
      
      Y1_T_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) ), (double)2.0*dpower_coeff_ );
      Y1_I_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) ), (double)2.0*dpower_coeff_ );
      //Y1_T_f = gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) );
      //Y1_I_f = gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) );
      Y2_T_f = Y1_T_f * Y1_T_f;
      Y2_I_f = Y1_I_f * Y1_I_f;
      _Y4[fbinX][iSearchX]    += Y2_T_f * Y2_T_f + Y2_I_f *Y2_I_f;
      _mean[fbinX][iSearchX]  += Y1_T_f          + Y1_I_f;
      _sigma[fbinX][iSearchX] += Y2_T_f          + Y2_I_f;
    }
    //cost_func_values_[iSearchX] += Y4_T + Y4_I;
  }

  nSamples_++;
  return;
}

const gsl_vector_complex* FDIIDThresholdEstimator::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
	
  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;
	
  if( frame_no >= 0 ){
    ad_X_L = srcL_->next(frame_no);
    ad_X_R = srcR_->next(frame_no);
  }
  else{
    if( frame_no_ == frame_reset_no_ ){
      ad_X_L = srcL_->next(0);
      ad_X_R = srcR_->next(0);
    }
    else{
      ad_X_L = srcL_->next(frame_no_+1);
      ad_X_R = srcR_->next(frame_no_+1);
    }
  }
  accumStats1( ad_X_L, ad_X_R );

  increment_();
  return vector_;
}

double FDIIDThresholdEstimator::calc_threshold()
{
  unsigned fftLen2 = (unsigned)srcL_->size() / 2;
  double min_rho   = 1000000;

  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
    unsigned iSearchX=0;
    double local_min_rho = 1000000;

    for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
      double rho;

      _mean[fbinX][iSearchX]  /= nSamples_;
      _sigma[fbinX][iSearchX] /= nSamples_;
      _Y4[fbinX][iSearchX]    /= nSamples_;
      cost_func_values_[fbinX][iSearchX] = _Y4[fbinX][iSearchX] - _beta * _sigma[fbinX][iSearchX] * _sigma[fbinX][iSearchX];
      rho = - cost_func_values_[fbinX][iSearchX]; /* negative kurtosis */
      //fprintf(stderr,"%f %e\n",threshold,rho);
      if( rho <= min_rho ){
	threshold_ = threshold;
	min_rho = rho;
      }
      if( rho <= local_min_rho ){
	gsl_vector_set( threshold_per_freq_, fbinX, threshold );
	local_min_rho = rho;
      }
    }
  }

  cost_func_computed_ = true;
  return threshold_;
}

void FDIIDThresholdEstimator::reset()
{
  unsigned fftLen2 = (unsigned)srcL_->size()/2;

  srcL_->reset();
  srcR_->reset();
  VectorComplexFeatureStream::reset();
  is_end_ = false;

  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
    unsigned iSearchX=0;

    for(float threshold=min_threshold_;threshold<=max_threshold_;threshold+=width_,iSearchX++){
      cost_func_values_[fbinX][iSearchX] = 0.0;
      _Y4[fbinX][iSearchX]    = 0.0;
      _mean[fbinX][iSearchX]  = 0.0;
      _sigma[fbinX][iSearchX] = 0.0;
    }
  }

  nSamples_ = 0;
  cost_func_computed_ = false;
}
