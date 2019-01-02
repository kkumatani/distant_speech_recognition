/**
 * @file objective_measure.cc
 * @brief objective measure for speech quality or distance
 * @author Kenichi Kumatani
 */

#include"objective_measure.h"
#include <sndfile.h>

/**
   @brief calculate the SNR.
   suppose that 
   s(t) is an original speech signal, and
   s'(t) is the enhanced one.
                           E{ s^2(t) }
   SNR = 10 log10 ( ---------------------- )
                    E{ ( s(t) - s'(t) )^2 }
   @param gsl_vector_float *original (original speech signal) [i/o]
   @param gsl_vector_float *enhanced (enhanced speech signal) [i/o]
   @param int normalizationOption [i]
   @notice *orignal and *enhanced will be replaced with mean-subtracted values
   @return SNR
 */
#define MEAN_NORMALIZATION 0x01 // 1
#define MAXPEAK_SCALING    0x02 // 2
#define STDV_SCALING       0x04 // 4
#define CC_SCALING         0x08 // 4
float calcSNR( gsl_vector_float *original, gsl_vector_float *enhanced, 
	       int normalizationOption = 0, bool ignoreSizeDifference=false );

/**
   @brief calculate the Itakura-Saito (IS) distance with power spectrums.
          The definition of the discrete version of the IS distance is 
          based on A. E-Jaroudi et al., "Discrete All-Pole Modeling", IEEE Trans. Signal Processing, vol. 39, 1991.
   @param gsl_vector_float *original (power spectrums of original speech signal)
   @param gsl_vector_float *enhanced (power spectrums of enhanced speech signal)
*/
float calcISDistance( NormalFFTAnalysisBankPtr& original,
		      NormalFFTAnalysisBankPtr& enhanced,
		      int bframe = 0, int eframe = -1 );

float calcSNR( gsl_vector_float *samp1, gsl_vector_float *samp2, 
	       int normalizationOption, bool ignoreSizeDifference )
{
  float err,sum,dif;
  float val1;
  float val2;
  size_t minLen;
  bool isOriginalSpeehLong;
  float snr;
  float scalingFactor1 = 1.0;
  float scalingFactor2 = 1.0;
  
  if( samp1->size > samp2->size ){
     minLen = samp2->size;
     isOriginalSpeehLong = true; 
  }
  else{
    minLen = samp1->size;
    isOriginalSpeehLong = false;
  }
  
  if( normalizationOption & MEAN_NORMALIZATION ){
    float mean1 = 0.0;
    float mean2 = 0.0;
      
    printf("subtraction with the mean\n");
    if( false==ignoreSizeDifference ){
      for(size_t i=0;i<samp1->size;i++){
	mean1 += gsl_vector_float_get( samp1, i );
      }
      mean1 = mean1 / samp1->size;
      for(size_t i=0;i<samp2->size;i++){
	mean2 += gsl_vector_float_get( samp2, i );
      }
      mean2 = mean2 / samp2->size;
    }
    else{
       for(size_t i=0;i<minLen;i++){
	 mean1 += gsl_vector_float_get( samp1, i );
	 mean2 += gsl_vector_float_get( samp2, i );
       }
       mean1 = mean1 / minLen;
       mean2 = mean2 / minLen;
    }
    gsl_vector_float_add_constant( samp1,  -mean1 );
    gsl_vector_float_add_constant( samp2,  -mean2 );
  }
  if( normalizationOption & MAXPEAK_SCALING ){
    float max1 = -10000;
    float max2 = -10000;

    printf("scaling samples with ther maximum values\n");
    for(size_t i=0;i<samp1->size;i++){
      if( gsl_vector_float_get( samp1, i ) > max1 )
	max1 = gsl_vector_float_get( samp1, i );
    }
    for(size_t i=0;i<samp2->size;i++){
      if( gsl_vector_float_get( samp2, i ) > max2 )
	max2 = gsl_vector_float_get( samp2, i );
    }
    scalingFactor1 = 1.0 / max1;
    scalingFactor2 = 1.0 / max2;
  }
  else if( normalizationOption & STDV_SCALING ){
    float val1, val2;
    float sigma21 = 0.0;
    float sigma22 = 0.0;

    printf("scaling samples with ther standard deviation\n");
    if( false==ignoreSizeDifference ){
      for(size_t i=0;i<samp1->size;i++){
	val1  = gsl_vector_float_get( samp1, i );
	sigma21 += val1 * val1;
      }
      sigma21 = sigma21 / samp1->size;
      for(size_t i=0;i<samp2->size;i++){
	val2 = gsl_vector_float_get( samp2, i );
	sigma22 += val2 * val2;
      }
      sigma22 = sigma22 / samp2->size;
    }
    else{
       for(size_t i=0;i<minLen;i++){
	 val1 = gsl_vector_float_get( samp1, i );
	 sigma21 += val1 * val1;
	 val2 = gsl_vector_float_get( samp2, i );
	 sigma22 += val2 * val2;
       }
       sigma21 = sigma21 / minLen;
       sigma22 = sigma22 / minLen;
    }
    //scalingFactor1 = 1.0;
    scalingFactor2 = sqrt( sigma21 / sigma22 );
  }
  else if( normalizationOption & CC_SCALING ){
    float sigma12 = 0.0;
    float sigma22 = 0.0;

    scalingFactor1 = 1.0;
    for(size_t i=0;i<samp2->size;i++){
      val1 = gsl_vector_float_get( samp1, i );
      val2 = gsl_vector_float_get( samp2, i );
      sigma12 += val1 * val2;
      sigma22 += val2 * val2;
    }
    scalingFactor2 = sigma12 / sigma22;
    fprintf(stderr,"The normalization factor %e\n",scalingFactor2);
  }

  sum = err = 0;
  for(size_t i=0;i<minLen;i++){
    val1 = gsl_vector_float_get( samp1, i ) * scalingFactor1;
    val2 = gsl_vector_float_get( samp2, i ) * scalingFactor2;
    sum += ( val1 * val1 );
    dif  =   val1 - val2;
    err += ( dif  * dif );
  }
  if( false == ignoreSizeDifference ){
    if( true==isOriginalSpeehLong ){
      for(size_t i=minLen;i<samp1->size;i++){
	val1 =  gsl_vector_float_get( samp1, i ) * scalingFactor1;
	sum += val1 * val1;
	err += val1 * val1;
      }
      printf("use %lu samples for the comparison\n",samp1->size);
    }
    else{
      for(size_t i=minLen;i<samp2->size;i++){
	val2 =  gsl_vector_float_get( samp2, i ) * scalingFactor2;
	sum += val2 * val2;
	err += val2 * val2;
      }
      printf("use %lu samples for the comparison\n",samp2->size);
    }
  }
  else{
    printf("use %lu samples for the comparison\n",minLen);
  }
  if( err > 0.0 )
    snr = 10 * log10f( sum / err );
  else
    snr = FLT_MAX;
  return snr;
}

float SNR::getSNR( const String& fn1, const String& fn2, int normalizationOption, int chX, int samplerate, int cfrom, int to)
{
  using namespace sndfile;
  SNDFILE* sndfile1, *sndfile2;
  SF_INFO sfinfo1, sfinfo2;
  gsl_vector_float *original,*enhanced;
  int nsamples1, nsamples2;
  float snr;

  sfinfo1.samplerate = samplerate;
  sfinfo2.samplerate = samplerate;

  sndfile1 = sf_open( fn1.c_str(), SFM_READ, &sfinfo1 );
  if (!sndfile1){
    throw jio_error("Could not open file %s.", fn1.c_str());
  }

  if (sf_error(sndfile1)) {
    sf_close(sndfile1);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile1));
  }

  sndfile2 = sf_open( fn2.c_str(), SFM_READ, &sfinfo2 );
  if (!sndfile2){
    throw jio_error("Could not open file %s.", fn2.c_str());
  }

  if (sf_error(sndfile2)) {
    sf_close(sndfile2);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile2));
  }

  if( chX > sfinfo1.channels || chX > sfinfo2.channels || chX < 1 ) {
    if(chX == 0 )
      throw jconsistency_error("Multi-channel read is not yet supported.");

    // for now just allow one channel to be loaded
    throw jconsistency_error("Selected channel out of range of available channels. %d", chX);
    //    chX = 1;
  }
  chX--;

  if( sfinfo1.samplerate != sfinfo2.samplerate ){
    throw jio_error("Sampling rates must be the same but %d!=%d\n",sfinfo1.samplerate,sfinfo2.samplerate);
  }

  if (cfrom < 0)
    cfrom = 0;
  if ( (to < 0) || (to >= sfinfo1.frames) || (to >= sfinfo2.frames) ){
    nsamples1 =  sfinfo1.frames - cfrom;
    nsamples2 =  sfinfo2.frames - cfrom;
  }
  else{
    nsamples1 = to - cfrom + 1;
    nsamples2 = to - cfrom + 1;
  }
  if ( (cfrom > sfinfo1.frames) || (cfrom > sfinfo2.frames) ) {
    sf_close(sndfile1);
    sf_close(sndfile2);
    throw jio_error("Cannot load samples from %d to %d.", cfrom, to);
  }

  original = gsl_vector_float_alloc(nsamples1);
  enhanced = gsl_vector_float_alloc(nsamples2);
  {
    unsigned maxChanN = (sfinfo1.channels>sfinfo2.channels)? sfinfo1.channels:sfinfo2.channels;
    unsigned maxLen = (nsamples1>nsamples2)? nsamples1:nsamples2;
    float *tmpsamples = new float[maxLen*maxChanN];
    
    if (sf_seek(sndfile1, cfrom, SEEK_SET) == -1)
      throw jio_error("Error seeking to %d", cfrom);
    sf_readf_float(sndfile1, tmpsamples, nsamples1);
    for (unsigned int i=0; i < nsamples1; i++)
      gsl_vector_float_set( original, i, tmpsamples[i*sfinfo1.channels + chX] );

    if (sf_seek(sndfile2, cfrom, SEEK_SET) == -1)
      throw jio_error("Error seeking to %d", cfrom);
    sf_readf_float(sndfile2, tmpsamples, nsamples2);
    for (unsigned int i=0; i < nsamples2; i++)
      gsl_vector_float_set( enhanced, i, tmpsamples[i*sfinfo2.channels + chX] );

    delete [] tmpsamples;
  }

  snr = calcSNR( original, enhanced, normalizationOption, false );
  sf_close(sndfile1);
  sf_close(sndfile2);
  gsl_vector_float_free( original );
  gsl_vector_float_free( enhanced );
  return snr;
}

float SNR::getSNR2( gsl_vector_float *original, gsl_vector_float *enhanced, int normalizationOption )
{
  return calcSNR( original, enhanced, normalizationOption );
}

float calcISDistance( NormalFFTAnalysisBankPtr& original,
		      NormalFFTAnalysisBankPtr& enhanced,
		      int bframe, int eframe ){
  float sumDist = 0.0;
  const gsl_vector_complex *vec1, *vec2;
  unsigned fftLen1 = original->fftLen();
  unsigned fftLen2 = enhanced->fftLen();
  unsigned frameN=0;

  if( fftLen1 != fftLen2 ){
    throw jio_error("The numbers of FFT points must be the same but %d != %d\n", fftLen1, fftLen2);
  }
  for(int frameX =0;;frameX++){
    try{
      vec1 = original->next();
      vec2 = enhanced->next();
      if( frameX >= bframe ){// calculate the discrete case of the Itakura-saito distance
	if( frameX <= eframe || eframe < 0 ){
	  float P1, P2;
	  float ratio, Eis = 0.0;
	  
	  for(unsigned fbinX=1;fbinX<=fftLen1/2;fbinX++){
	    P1 = gsl_complex_abs2( gsl_vector_complex_get(vec1,fbinX) );
	    P2 = gsl_complex_abs2( gsl_vector_complex_get(vec2,fbinX) );
	    if( P1 > 0.0 && P2 > 0.0 ){
	      ratio = P1 / P2;
	      Eis += ratio - log(ratio) - 1;
	    }
	    else{
	      if( P1 > 0.0 && P2 == 0.0 )
		printf("The processed spectrum is zero: fr %d freq %d\n",frameN,fbinX);
	    }
	  }
	  Eis = Eis / ( fftLen1/2 );
	  sumDist += Eis;
	  frameN++;
	}
      }
    }
    catch(  jiterator_error& e) {
      break;
    }
    if( original->is_end()==true || enhanced->is_end()==true )
      break;
  }

  return sumDist/frameN;
}

float ItakuraSaitoMeasurePS::getDistance( const String& fn1, const String& fn2, int chX, int samplerate, int bframe, int eframe )
{
  
  SampleFeaturePtr sampP1 = new SampleFeature( "", _D, _D, false );
  SampleFeaturePtr sampP2 = new SampleFeature( "", _D, _D, false );
  NormalFFTAnalysisBankPtr fftP1 = new NormalFFTAnalysisBank((VectorFloatFeatureStreamPtr&)sampP1, _fftLen, _r, _windowType );
  NormalFFTAnalysisBankPtr fftP2 = new NormalFFTAnalysisBank((VectorFloatFeatureStreamPtr&)sampP2, _fftLen, _r, _windowType );

  sampP1->read( fn1.c_str(), 0, samplerate, chX );
  sampP2->read( fn2.c_str(), 0, samplerate, chX );

  float ISDist = calcISDistance( fftP1, fftP2, bframe, eframe );
  return ISDist;
}
