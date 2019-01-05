/*
 * @file localization.h
 * @brief source localization
 * @author Ulrich Klee
 */

#include <math.h>
#include "localization.h"
#include "square_root/square_root.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "feature/feature.h"

#define SOUNDSPEED 343740.0

//  Implementation of InnerLoop for SrpPhat-SourceLocalization

gsl_vector* getSrpPhat(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeX, gsl_vector *searchRangeY, gsl_matrix *arrgeom, int zPos) {

  int searchStartX    = (int)gsl_vector_get(searchRangeX, 0);
  int searchStopX     = (int)gsl_vector_get(searchRangeX, 1);
  int iterStepsX      = (int)gsl_vector_get(searchRangeX, 2);
  int searchStartY    = (int)gsl_vector_get(searchRangeY, 0);
  int searchStopY     = (int)gsl_vector_get(searchRangeY, 1);
  int iterStepsY      = (int)gsl_vector_get(searchRangeY, 2);
  const float PI     = 3.14159265358979323846f;
  int nChan          = mFramePerChannel->size1;
  int fftLen         = mFramePerChannel->size2;
  int fftLen2 = fftLen / 2;
  double fftSum, sumK, sumL;
  gsl_complex one_j = gsl_complex_rect(0.0, 1.0);

  double argMax = 0.0;
  int bestPosX  = -10000;
  int bestPosY = -10000;
  
  gsl_vector *delays = gsl_vector_alloc(nChan);
  gsl_vector *result = gsl_vector_alloc(3);
  
  for (int iterX = searchStartX; iterX<searchStopX; iterX +=iterStepsX) {
    for (int iterY = searchStartY; iterY<searchStopY; iterY +=iterStepsY) {
      calcDelays(iterX, iterY, zPos, arrgeom, delays);
      
      sumL = 0.0;
      for (int l=0; l<nChan; l++) {
	sumK = 0.0;
	for (int k=l; k<nChan; k++) {
	  fftSum = 0.0;
	  if (k!=l) {
	    for (int j=0; j<=fftLen2; j++) {
	      gsl_complex jXl             = gsl_matrix_complex_get(mFramePerChannel, l, j);
	      gsl_complex jXk             = gsl_complex_conjugate(gsl_matrix_complex_get(mFramePerChannel, k, j));
	      
	      gsl_complex vectorProduct   = gsl_complex_mul(jXl, jXk);
	      double      phat            = gsl_complex_abs(vectorProduct);
	      
	      gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
	      
	      if (phat>0.0) {
		weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
	      }
	      
	      double help1                = 2*PI*j*delta_f*(gsl_vector_get(delays, k)-gsl_vector_get(delays, l));
	      
	      gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
	      
	      gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);
	      fftSum += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
	    } 
	  }
	  sumK += fftSum;
	}
	sumL +=sumK;
    }
      if (sumL > argMax) {
	argMax  = sumL;
	bestPosY = iterY;
	bestPosX = iterX;
      } 
    }
  }
  
  gsl_vector_free(delays);

  gsl_vector_set(result, 0, bestPosX);
  gsl_vector_set(result, 1, bestPosY);
  return result;
}

//Method to calculate Delays between MicrophoneArray for a given ArrayPosition, taken from subbandBeamforming.py

void calcDelays(int x, int y, int z, const gsl_matrix* mpos, gsl_vector* delays) {
  const double sspeed = SOUNDSPEED;
  const int channel   = mpos->size1;

  for (int i=0; i<channel; i++) {
    double delayi = sqrt(pow((x-gsl_matrix_get(mpos,i,0)),2)+pow((y-gsl_matrix_get(mpos,i,1)),2)
			 +pow((z-gsl_matrix_get(mpos,i,2)),2))/sspeed; 
    gsl_vector_set(delays,i, delayi);
  }
}

// same as above for plane-wave-assuption

void calcDelaysOfLinearMicrophoneArray( float azimuth, const gsl_matrix* mpos, gsl_vector* delays) {
  const double sspeed = SOUNDSPEED;
  size_t channel   = mpos->size1;

  gsl_vector_set(delays, 0, (double)0.0);
  for (size_t i=1; i<channel; i++) {
    float dist = fabs( gsl_matrix_get(mpos,i,1) - gsl_matrix_get(mpos,0,1) );
    double delayi = - dist * sin((double)azimuth) /sspeed;
    gsl_vector_set(delays, i, delayi);
    //printf("%e %e %e %e \n",delayi,dist,sin((double)azimuth),sspeed);
  }
}

/**
   @brief calculate time delays of the circular microphone array under the far-field assumption.
   @param float azimuth[in] radian, 0 to 2pi
   @param float polarAngle[in] radian, 0 to pi
   @param const gsl_matrix* mpos[in] the positions of microphones: mpos[i][0], mpos[i][1] and mpos[i][2] contain x, y and z-values of the i-th microphone, respectively.
   @param gsl_vector* delays[out]
 */
void calcDelaysOfCircularMicrophoneArray(float azimuth, float polarAngle, const gsl_matrix* mpos, gsl_vector* delays)
{
   const float sspeed = SOUNDSPEED;
   const int channel   = mpos->size1;

   for (int i=0; i<channel; i++) {
     float cx = - sin(polarAngle) * cos(azimuth);
     float cy = - sin(polarAngle) * sin(azimuth);
     float cz = - cos(polarAngle);
     float delay = ( cx * gsl_matrix_get(mpos,i,0) + cy * gsl_matrix_get(mpos,i,1) + cz * gsl_matrix_get(mpos,i,2) ) / sspeed;
     gsl_vector_set(delays, i, (double)delay);
   }
}

//Implementation to calculate Delays for PDA
gsl_vector* getDelays(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *delays)
{
  const float PI = 3.14159265358979323846f;
  int fftLen = mFramePerChannel->size2;
  int fftLen2 = fftLen / 2;

  gsl_complex one_j = gsl_complex_rect(0.0, 1.0);

  gsl_vector* result = gsl_vector_alloc(delays->size);

  int counter = 0;

  for (unsigned delayX = 0; delayX < delays->size; delayX++) {
    double sum = 0.0;

    for (int j = 0; j <= fftLen2; j++) {
      gsl_complex jXl             = gsl_matrix_complex_get(mFramePerChannel, /* l= */ 0, j);
      gsl_complex jXk             = gsl_complex_conjugate(gsl_matrix_complex_get(mFramePerChannel, /* k= */ 1, j));

      gsl_complex vectorProduct   = gsl_complex_mul(jXl, jXk);
      double      phat            = gsl_complex_abs(vectorProduct);

      gsl_complex weightedProduct =  gsl_complex_rect(0.0,0.0);
      if (phat > 0.0)
	weightedProduct           = gsl_complex_div_real(vectorProduct, phat);

      double help1                = 2*PI*j*delta_f * gsl_vector_get(delays, delayX);
      gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
      gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);

      sum += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
    }
    gsl_vector_set(result, counter, sum / fftLen);
    ++counter;
  }
  return result;
}

// Expecting a plain soundwave, this method calculates the azimuth from the array to the speaker (Benesty et al)

double getAzimuth(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_matrix *arrgeom, gsl_vector *delays) {
  const float  PI         = 3.14159265358979323846f;
  const int    nChan      = mFramePerChannel->size1;
  const int    fftLen     = mFramePerChannel->size2;
  double microDist        = gsl_matrix_get(arrgeom, 0, 1)-gsl_matrix_get(arrgeom, 1, 1);
  const double sspeed     = SOUNDSPEED;
  const int    fftLen2    = fftLen / 2;
  const gsl_complex one_j = gsl_complex_rect(0.0, 1.0);
  double bestDelay, minDeterminant=0.0;
  gsl_matrix *R_m = gsl_matrix_alloc(nChan, nChan);

  for (unsigned delayX = 0; delayX < delays->size; delayX++) {
    for (int k = 0; k<nChan; k++) {
      for (int l = 0; l<nChan; l++) {
	long double crossCorr = 0;
	if (k==l) crossCorr = 1.0;
	else {
	  for (int j = 0; j <= fftLen2; j++) {
	    gsl_complex jXk             = gsl_matrix_complex_get(mFramePerChannel, k, j);
	    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(mFramePerChannel, l, j));
	    
	    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
	    double phat                 = gsl_complex_abs(vectorProduct);
	    
	    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
	    if (phat > 0.0)
	      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
	    
	    double help1                = 2*PI*j*delta_f * gsl_vector_get(delays, delayX)*microDist*(l-k);
	    gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
	    gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);
	    
	    crossCorr += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
	  } 
	}
	gsl_matrix_set(R_m, k, l, crossCorr);
      }
    } 
    gsl_permutation *p = gsl_permutation_alloc(nChan);
    int signum;
    gsl_linalg_LU_decomp(R_m, p, &signum);
    long double determinant = gsl_linalg_LU_det(R_m, signum);
    if (minDeterminant == 0.0) {
      minDeterminant = determinant;
      bestDelay = gsl_vector_get(delays, delayX);
    }
    else if (determinant < minDeterminant) {
      minDeterminant = determinant;
      bestDelay = gsl_vector_get(delays, delayX);
    }
    gsl_permutation_free(p);
  }
  gsl_matrix_free(R_m);
  double azimuth = acos(sspeed*bestDelay/microDist);
 
  return azimuth;
}

// get Correlation between the first and second channel of the array
gsl_vector* getCurrentCorrelation(gsl_matrix_complex *mFramePerChannel, gsl_matrix *arrgeom ,double delta_f, gsl_vector *delays){
  const float  PI         = 3.14159265358979323846f;
  const int    fftLen     = mFramePerChannel->size2;
  const double microDist  = (gsl_matrix_get(arrgeom, 0, 1)-gsl_matrix_get(arrgeom, 1, 1));
  const int    fftLen2    = fftLen / 2;
  const gsl_complex one_j = gsl_complex_rect(0.0, 1.0);

  gsl_vector *crossResult = gsl_vector_alloc(delays->size);
  
  

  for (unsigned delayX = 0; delayX < delays->size; delayX++) {
    int k = 0;
    int l = 1;
    double crossCorr = 0.0;
    for (int j = 0; j <= fftLen2; j++) {
      gsl_complex jXk             = gsl_matrix_complex_get(mFramePerChannel, k, j);
      gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(mFramePerChannel, l, j));
      
      gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
      double      phat            = gsl_complex_abs2(vectorProduct);
      
      gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
      if (phat > 0.0)
	weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
      
      double help1                = 2*PI*j*delta_f * gsl_vector_get(delays, delayX)*microDist*(l-k);
      gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
      gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);
      
      crossCorr += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
    }
    gsl_vector_set(crossResult, delayX, crossCorr);
  }
  return crossResult;
}   


//Using the same plane wave assumption for srp_phat_algorithm

double getPlaneWaveSrp(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeY, gsl_matrix *arrgeom) {
  int searchStartY    = (int)gsl_vector_get(searchRangeY, 0);
  int searchStopY     = (int)gsl_vector_get(searchRangeY, 1);
  int iterStepsY      = (int)gsl_vector_get(searchRangeY, 2);
  const float PI     = 3.14159265358979323846f;
  int nChan          = mFramePerChannel->size1;
  int fftLen         = mFramePerChannel->size2;
  int fftLen2 = fftLen / 2;
  double fftSum, sumK, sumL;
  gsl_complex one_j = gsl_complex_rect(0.0, 1.0);

  double argMax = 0.0;
  int bestPosY = -10000;
  
  gsl_vector *delays = gsl_vector_alloc(nChan);
  
  for (int iterY = searchStartY; iterY<searchStopY; iterY +=iterStepsY) {
    calcDelaysOfLinearMicrophoneArray(iterY, arrgeom, delays);
    
    sumL = 0.0;
    for (int l=0; l<nChan; l++) {
      sumK = 0.0;
      for (int k=l; k<nChan; k++) {
	fftSum = 0.0;
	if (k!=l) {
	  for (int j=0; j<=fftLen2; j++) {
	    gsl_complex jXl             = gsl_matrix_complex_get(mFramePerChannel, l, j);
	    gsl_complex jXk             = gsl_complex_conjugate(gsl_matrix_complex_get(mFramePerChannel, k, j));
	    
	    gsl_complex vectorProduct   = gsl_complex_mul(jXl, jXk);
	    double      phat            = gsl_complex_abs(vectorProduct);
	    
	    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
	      
	    if (phat>0.0) {
	      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
	    }
	      
	    double help1                = 2*PI*j*delta_f*(gsl_vector_get(delays, k)-gsl_vector_get(delays, l));
	    
	    gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
	    
	    gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);
	    fftSum += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
	  } 
	}
	sumK += fftSum;
      }
	sumL +=sumK;
    }
    if (sumL > argMax) {
      argMax  = sumL;
      bestPosY = iterY;
    } 
  }
  return bestPosY;
}


// Implementation of BenestyApproach, using Interpolation to speedup search

double getCrossCorrDeterminant(gsl_matrix_complex *spectralSamplesPerChan, double delta_f) {

  const int    nChan      = spectralSamplesPerChan->size1;
  const int    fftLen     = spectralSamplesPerChan->size2;
  double determinant      = 10000;

  gsl_matrix *R_m = gsl_matrix_alloc(nChan, nChan);

  for (int k = 0; k<nChan; k++) {
    for (int l = 0; l<nChan; l++) {
      double crossCorr = 0;
      for (int j = 0; j < fftLen; j++) {
	gsl_complex jXk             = gsl_matrix_complex_get(spectralSamplesPerChan, k, j);
	gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSamplesPerChan, l, j));
	
	gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
	double phat                 = gsl_complex_abs(vectorProduct);
	
	gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
	if (phat > 0.0)
	  weightedProduct           = gsl_complex_div_real(vectorProduct, phat);

	crossCorr += GSL_REAL(weightedProduct);
      } 
      gsl_matrix_set(R_m, k, l, crossCorr);
    }
   
    gsl_permutation *p = gsl_permutation_alloc(nChan);
    int signum;
    gsl_linalg_LU_decomp(R_m, p, &signum);
    determinant = gsl_linalg_LU_det(R_m, signum);
    
    gsl_permutation_free(p);
  }
  gsl_matrix_free(R_m);
  
  return determinant;
}

// returns time-delay and corresponding cross-correlation for a given sample
void getGCCRaw(const gsl_matrix_complex* spectralSample, double sampleRate, gsl_vector* gcc)
{
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  if (gcc->size != fftLen)
    throw jdimension_error("GCC vector length (%d) should be same as FFT length (%d)",
			   gcc->size, fftLen);

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  for (unsigned i = 0; i < fftLen; i++)
    gsl_vector_set(gcc, i, crossCorrelation[i]);

  delete[] crossCorrelation;
  gsl_vector_complex_free(crossSpectrum);
}

// returns time-delay and corresponding cross-correlation for a given sample
const gsl_vector* getGCC(gsl_matrix_complex *spectralSample, double sampleRate)
{ 
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  gsl_matrix *interpolValues = gsl_matrix_alloc(fftLen, 2);
  
  double maxCorr = -HUGE_VAL;
  int delayPos   = 0;
  for (int j = 0; j < fftLen; j++) {
    int idx;
    double del = 0.0;
    if (j<fftLen2) {
      idx = j + fftLen2;
      del = j / sampleRate;
    } else {
      idx = j - fftLen2;
      del = ((j - fftLen)/sampleRate);
    }

    double corr = crossCorrelation[j];
    gsl_matrix_set(interpolValues, idx, 0, del);
    gsl_matrix_set(interpolValues, idx, 1, corr);
    
    if (corr > maxCorr) {
      maxCorr  = corr;
      delayPos = idx;
    }
  }
  
  gsl_vector_complex_free(crossSpectrum);
  delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(2);

  gsl_vector_set(finalDelay, 0, getInterpolation(interpolValues, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);

  gsl_matrix_free(interpolValues);

  return finalDelay;
}

// returns time-delay and corresponding cross-correlation for a given sample
const gsl_vector* getWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay)
{ 
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  gsl_matrix *interpolValues = gsl_matrix_alloc(fftLen, 2);
  
  double maxCorr = -HUGE_VAL;
  int delayPos   = 0;
  for (int j = 0; j < fftLen; j++) {
    int idx;
    double del = 0.0;
    if (j<fftLen2) {
      idx = j + fftLen2;
      del = j / sampleRate;
    } else {
      idx = j - fftLen2;
      del = ((j - fftLen)/sampleRate);
    }

    double corr = crossCorrelation[j];
    gsl_matrix_set(interpolValues, idx, 0, del);
    gsl_matrix_set(interpolValues, idx, 1, corr);
    
    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr  = corr;
      delayPos = idx;
    }
  }
  
  gsl_vector_complex_free(crossSpectrum);
  delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(2);

  gsl_vector_set(finalDelay, 0, getInterpolation(interpolValues, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);

  gsl_matrix_free(interpolValues);

  return finalDelay;
}

// returns time-delay and corresponding cross-correlation for a given sample and the ratio between the 1st and 2nd largest peak
const gsl_vector* getWindowedGCCratio(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay)
{
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  gsl_matrix *interpolValues = gsl_matrix_alloc(fftLen, 2);
  
  double maxCorr = -HUGE_VAL;
  double maxCorr2 = -HUGE_VAL;
  int delayPos   = 0;
  for (int j = 0; j < fftLen; j++) {
    int idx;
    double del = 0.0;
    if (j<fftLen2) {
      idx = j + fftLen2;
      del = j / sampleRate;
    } else {
      idx = j - fftLen2;
      del = ((j - fftLen)/sampleRate);
    }

    double corr = crossCorrelation[j];
    gsl_matrix_set(interpolValues, idx, 0, del);
    gsl_matrix_set(interpolValues, idx, 1, corr);
    
    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr2 = maxCorr;
      maxCorr  = corr;
      delayPos = idx;
    }
    else if((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr2)) {
      maxCorr2 = corr;
    } 
  }
  
  gsl_vector_complex_free(crossSpectrum);
  delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(3);

  gsl_vector_set(finalDelay, 0, getInterpolation(interpolValues, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);
  gsl_vector_set(finalDelay, 2, maxCorr/maxCorr2);

  gsl_matrix_free(interpolValues);

  return finalDelay;
}

// returns time-delay and corresponding cross-correlation for a given sample
const gsl_vector* getWindowedGCCdirect(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay)
{
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  //gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  //double*             crossCorrelation = new double[fftLen];
  static gsl_vector_complex* crossSpectrum = NULL;
  if (crossSpectrum == NULL) crossSpectrum = gsl_vector_complex_alloc(len);
  static double* crossCorrelation = NULL;
  if (crossCorrelation == NULL) crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  double maxCorr = -HUGE_VAL;
  double delay   = 0;
  for (int j = 0; j < fftLen; j++) {
    double del = 0.0;
    if (j<fftLen2) {
      del = j / sampleRate;
    } else {
      del = ((j - fftLen)/sampleRate);
    }

    double corr = crossCorrelation[j];
    
    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr  = corr;
      delay = del;
    }
  }
  
  //gsl_vector_complex_free(crossSpectrum);
  //delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(2);

  gsl_vector_set(finalDelay, 0, delay);
  gsl_vector_set(finalDelay, 1, maxCorr);

  return finalDelay;
}

// returns time-delay and corresponding cross-correlation for a given sample
const gsl_vector* getWindowedGCCabs(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay)
{
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  gsl_matrix *interpolValues = gsl_matrix_alloc(fftLen, 2);
  
  double maxCorr = -HUGE_VAL;
  int delayPos   = 0;
  for (int j = 0; j < fftLen; j++) {
    int idx;
    double del = 0.0;
    if (j<fftLen2) {
      idx = j + fftLen2;
      del = j / sampleRate;
    } else {
      idx = j - fftLen2;
      del = ((j - fftLen)/sampleRate);
    }

    double corr = fabs(crossCorrelation[j]);
    gsl_matrix_set(interpolValues, idx, 0, del);
    gsl_matrix_set(interpolValues, idx, 1, corr);
    
    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr  = corr;
      delayPos = idx;
    }
  }
  
  gsl_vector_complex_free(crossSpectrum);
  delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(2);

  gsl_vector_set(finalDelay, 0, getInterpolation(interpolValues, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);

  gsl_matrix_free(interpolValues);

  return finalDelay;
}

// returns time-delay and corresponding cross-correlation for a given sample
const gsl_vector* getDynWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay, double wMinDelay, double wMaxDelay, double threshold)
{
  const int fftLen  = spectralSample->size2;
  const int fftLen2 = (fftLen+1)/ 2;

  int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;

  gsl_vector_complex* crossSpectrum    = gsl_vector_complex_alloc(len);
  double*             crossCorrelation = new double[fftLen];

  unsigned k = 0;
  unsigned l = 1;

  for (unsigned j = 0; j < len; j++) {
    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
    
    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
    double phat                 = gsl_complex_abs(vectorProduct);
    
    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
    if (phat > 0.0)
      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
    
    gsl_vector_complex_set(crossSpectrum, j, weightedProduct);
  }
    
  pack_half_complex(crossCorrelation, crossSpectrum, fftLen);
  gsl_fft_halfcomplex_radix2_inverse (crossCorrelation, /* stride=*/ 1, fftLen);

  gsl_matrix *interpolValues = gsl_matrix_alloc(fftLen, 2);
  
  double maxCorr = -HUGE_VAL;
  double wMaxCorr = -HUGE_VAL;
  int delayPos   = 0;
  int wDelayPos   = 0;
  for (int j = 0; j < fftLen; j++) {
    int idx;
    double del = 0.0;
    if (j<fftLen2) {
      idx = j + fftLen2;
      del = j / sampleRate;
    } else {
      idx = j - fftLen2;
      del = ((j - fftLen)/sampleRate);
    }

    double corr = crossCorrelation[j];
    gsl_matrix_set(interpolValues, idx, 0, del);
    gsl_matrix_set(interpolValues, idx, 1, corr);
    
    if ((del >= wMinDelay)&&(del <= wMaxDelay)&&(corr > wMaxCorr)) {
      wMaxCorr  = corr;
      wDelayPos = idx;
    }
    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr  = corr;
      delayPos = idx;
    }
  }
  
  gsl_vector_complex_free(crossSpectrum);
  delete[] crossCorrelation;
  
  static gsl_vector* finalDelay = NULL;
  if(finalDelay == NULL) finalDelay = gsl_vector_alloc(2);

  if (wMaxCorr > threshold) {
    maxCorr  = wMaxCorr;
    delayPos = wDelayPos;
  }

  gsl_vector_set(finalDelay, 0, getInterpolation(interpolValues, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);

  gsl_matrix_free(interpolValues);

  return finalDelay;
}

double getInterpolation(gsl_matrix *crossResult, int delayPos) {
  const int vectorLength = crossResult->size1;
  double delay;

  if ((delayPos>0) && (delayPos<(vectorLength-1))) {
    double x0 = gsl_matrix_get(crossResult, delayPos-1, 0);
    double y0 = gsl_matrix_get(crossResult, delayPos-1, 1);
    double x1 = gsl_matrix_get(crossResult, delayPos, 0);
    double y1 = gsl_matrix_get(crossResult, delayPos, 1);
    double x2 = gsl_matrix_get(crossResult, delayPos+1, 0);
    double y2 = gsl_matrix_get(crossResult, delayPos+1, 1);
    
    
    //Calculate best Delay
    delay = 0.5*((x0+x1)-(y1-y0)/(x1-x0)*(x2-x0)/((y2-y1)/(x2-x1)-(y1-y0)/(x1-x0)));
    
  }
  else if (delayPos==0) delay  = getInterpolation(crossResult, ++delayPos);
  else if (delayPos==(vectorLength-1))  delay = getInterpolation(crossResult, --delayPos);
  else delay = 0.0;

  return delay;
}


gsl_vector* get3DPosition(gsl_vector* yCoord, gsl_vector* azimuth1, gsl_vector* azimuth2, double xPos, double zPos) {
  const float  Pi       = 3.14159265358979323846f;
  const int lenAzimuth1 = azimuth1->size;
  const int lenAzimuth2 = azimuth2->size;
  const int lenYCoord   = yCoord  ->size;

  int counter = 0;
  double Xres = 0.0;
  double Yres = 0.0;
  double yMin, xMin = 0.0;
  double yMax, xMax = 0.0;

  for (int i=0; i<lenAzimuth1; i++){
    double angle1 = gsl_vector_get(azimuth1, i);
    for (int j=0; j<lenAzimuth2; j++){
      double angle2 = gsl_vector_get(azimuth2, j);
      double X1 = xPos;
      double X2 = xPos;
      double Y1 = gsl_vector_get(yCoord, i);
      double Y2 = gsl_vector_get(yCoord, j+lenYCoord/2);

      double depth = 1.0;
      double k     = 0.0;

      double Ya = depth;
      double Xa = tan(angle1)*Ya;
      double Yb = depth;
      double Xb = tan(angle2)*Yb;

      if ((angle1-angle2)>0.005||(angle1-angle2)<-0.005){
	if (((angle1<Pi/2.0)&&(angle2>Pi/2.0)) || ((angle1>=Pi/2.0)&&(angle2>angle1)) || ((angle2<=Pi/2.0)&(angle1<angle2))) {
	  if ((Xb!=0.0) && (Yb!=0.0)) {
	    k = ((X1-X2)/Xb-(Y1-Y2)/Yb)/(Ya/Yb-Xa/Xb);
	  }
	}
      }

      if (k!=0) {
	double x = X1+k*Xa;
	double y = Y1+k*Ya;
	Xres += x;
	Yres += y;
	if (x<xMin) {
	  xMin = x;
	  yMin = y;
	}
	else if (x>xMax) {
	  xMax = x;
	  yMax = y;
	}

	++counter;
      }
    }
  }

  gsl_vector *position = gsl_vector_alloc(2);
  if (counter!=0){
    //Try little "cheating" with xPos
    gsl_vector_set(position, 0, ((Xres-xMin-xMax)/(counter-2)));
    gsl_vector_set(position, 1, (Yres-yMin-yMax)/(counter-2));
  }
  else {
    gsl_vector_set(position, 0, 0);
    gsl_vector_set(position, 1, 0);
  }
   
  return position;
}


// Implementation for T-Shape
gsl_vector* get3DPosition_T_shape(gsl_matrix *arrgeom1, int arrayNr1, gsl_matrix *arrgeom2, int arrayNr2, gsl_matrix *arrgeom3, double azimuth1, double azimuth2, double azimuth3) {

  double depth = 10.0;
  double k     = 0.0;
  
  double X1    = (gsl_matrix_get(arrgeom1, 0, 0)+gsl_matrix_get(arrgeom1, 1, 0))/2.0;
  double Y1    = (gsl_matrix_get(arrgeom1, 0, 1)+gsl_matrix_get(arrgeom1, 1, 1))/2.0;
  double X2    = (gsl_matrix_get(arrgeom2, 0, 0)+gsl_matrix_get(arrgeom2, 1, 0))/2.0;
  double Y2    = (gsl_matrix_get(arrgeom2, 0, 1)+gsl_matrix_get(arrgeom2, 1, 1))/2.0;

  double Xa, Xb, Ya, Yb, x, y, z;

  if (arrayNr1 == 0 || arrayNr1 == 3) {
    Ya = depth;
    Xa = tan(azimuth1)*Ya;
  }
  else {
    Xa = depth;
    Ya = tan(azimuth1)*Xa;
  }

  if (arrayNr2 == 0 || arrayNr2 == 3) {
    Yb = depth;
    Xb = tan(azimuth2)*Yb;
  }
  else {
    Xb = depth;
    Yb = tan(azimuth2)*Xb;
  }

  if ((Xb!=0.0) && (Yb!=0.0)) {
    k = ((X1-X2)/Xb-(Y1-Y2)/Yb)/(Ya/Yb-Xa/Xb);
  }

  gsl_vector *position = gsl_vector_alloc(3);
  
  if (k!=0.0) {
    x = X1+k*Xa;
    y = Y1+k*Yb;

    double posZX = gsl_matrix_get(arrgeom3, 0, 0); 
    double posZY = gsl_matrix_get(arrgeom3, 0, 1);
    double posZZ = gsl_matrix_get(arrgeom3, 0, 2);
    
    //z = sqrt(height_tdoa*height_tdoa*sspeed*sspeed-(x-posZX)*(x-posZX)-(y-posZY)*(y-posZY))+posZZ;
   
    if (arrayNr1 == 0 || arrayNr1 ==3) 
      z = sqrt((posZX-x)*(posZX-x))/cos(azimuth3)+posZZ;
    else 
      z = sqrt((posZY-y)*(posZY-y))/cos(azimuth3)+posZZ;

    gsl_vector_set(position, 0, x);
    gsl_vector_set(position, 1, y);
    gsl_vector_set(position, 2, z);
  }
  
  else {
    gsl_vector_set(position, 0, 0.0);
    gsl_vector_set(position, 1, 0.0);
    gsl_vector_set(position, 2, 0.0);
  }
   
  return position;
}


//old implementation of getGCC
gsl_vector* getGCC_old(gsl_matrix_complex *spectralSample, double delta_f, gsl_vector *delays) {
  const float  PI         = 3.14159265358979323846f;
  const int    nChan      = 2;
  const int    fftLen     = spectralSample->size2;
  const int    fftLen2    = fftLen / 2;
  const gsl_complex one_j = gsl_complex_rect(0.0, 1.0);
  const int nDelays       = delays->size;
  double maxCorr          = 0.0;
  int delayPos            = 0;

  gsl_matrix *crossResults = gsl_matrix_alloc(nDelays, 2);

  
  for (unsigned delayX = 0; delayX < nDelays; delayX++) {
    double sumK = 0;
    for (int k = 0; k<nChan; k++) {
      double sumL = 0;
      for (int l = 0; l<nChan; l++) {
        double crossCorr = 0;
	if (k==l) crossCorr = 512.0;
	else {
	  for (int j = 0; j <= fftLen2; j++) {
	    gsl_complex jXk             = gsl_matrix_complex_get(spectralSample, k, j);
	    gsl_complex jXl             = gsl_complex_conjugate(gsl_matrix_complex_get(spectralSample, l, j));
	    
	    gsl_complex vectorProduct   = gsl_complex_mul(jXk, jXl);
	    double phat                 = gsl_complex_abs(vectorProduct);
	    
	    gsl_complex weightedProduct = gsl_complex_rect(0.0,0.0);
	    if (phat > 0.0)
	      weightedProduct           = gsl_complex_div_real(vectorProduct, phat);
	    
	    double help1                = 2*PI*j*delta_f * gsl_vector_get(delays, delayX)*(l-k);
	    gsl_complex Wj              = gsl_complex_exp(gsl_complex_mul_real(one_j, help1));
	    gsl_complex finalProduct    = gsl_complex_mul(weightedProduct, Wj);
	    
	    crossCorr += (j == 0 || j == fftLen2) ? GSL_REAL(finalProduct) : 2.0 * GSL_REAL(finalProduct);
	  } 
	}
	sumL += crossCorr;
      }
      sumK += sumL;
    }
    gsl_matrix_set(crossResults, delayX, 0, gsl_vector_get(delays, delayX));
    gsl_matrix_set(crossResults, delayX, 1, sumK);

    if (sumK >= maxCorr) {
      maxCorr  = sumK;
      delayPos = delayX; 
    }
  }
 
  gsl_vector *finalDelay = gsl_vector_alloc(2);
  gsl_vector_set(finalDelay, 0, getInterpolation(crossResults, delayPos));
  gsl_vector_set(finalDelay, 1, maxCorr);

  gsl_matrix_free(crossResults);
  
  return finalDelay;
}

// Calculate the lower triangular Matrix of a cholesky decomposition
gsl_matrix* getLowerTriangMatrix(gsl_matrix* fullMatrix) {
  gsl_linalg_cholesky_decomp(fullMatrix);

  return fullMatrix;
}

// Calculate new Xi for iterated RLS_PosEst

gsl_vector* getXi(gsl_matrix* D1_2){
  const int dimD = D1_2->size1;
  gsl_vector* xi = gsl_vector_alloc(dimD);

  cholesky_backsub(D1_2, xi);

  return xi;
}

#if 0
PhaseTransform::PhaseTransform(unsigned sz)
  : _fftLen(),
#ifdef HAVE_LIBFFTW3
    _crossSpectrum(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _fftLen))),
    _crossSpectrum(static_cast<double*>(fftw_malloc(sizeof(fftw_double) * _fftLen)))
#else
  _crossSpectrum(gsl_vector_complex_alloc(len)),
  _crossCorrelation(double[fftLen]),
  _interpolValues(gsl_matrix_alloc(fftLen, 2))
#endif
{
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_r2c_1d(_Mx2, 
			       (double (*)[2])_samples,
			       (double (*)[2])_output,
			       FFTW_MEASURE);
#endif
}

PhaseTransform::~PhaseTransform()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan);
  fftw_free(_samples);
  fftw_free(_ouput)
#else
  _crossSpectrum(gsl_vector_complex_alloc(len)),
  _crossCorrelation(double[fftLen]),
  _interpolValues(gsl_matrix_alloc(fftLen, 2))
#endif
}

#endif


NoisePowerSpectrum::NoisePowerSpectrum(double alpha) {
  powerSpectrum = NULL;
  this->alpha = alpha;
  alpha1 = 1 - alpha;
  timestamp = 0.0;
}

void NoisePowerSpectrum::add(const gsl_vector_complex *noiseSpectrum, double timestamp)
{
  if (this->timestamp != timestamp) {
    if (powerSpectrum != NULL) {
      for (unsigned i = 0; i < powerSpectrum->size; i++) {
	double v = gsl_complex_abs(gsl_vector_complex_get(noiseSpectrum, i));
	gsl_vector_set(powerSpectrum, i, alpha*gsl_vector_get(powerSpectrum, i) + alpha1*v*v);
      }
    } else {
      unsigned fftLen = noiseSpectrum->size;
      unsigned fftLen2 = (fftLen+1)/2;
      unsigned len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;
      powerSpectrum = gsl_vector_alloc(len);
      for (unsigned i = 0; i < len; i++) {
	double v = gsl_complex_abs(gsl_vector_complex_get(noiseSpectrum, i));
	gsl_vector_set(powerSpectrum, i, alpha1*v*v);
      }
    }
    this->timestamp = timestamp;
  }
}

NoiseCrossSpectrum::NoiseCrossSpectrum(double alpha) {
  crossSpectrum = NULL;
  this->alpha = alpha;
  alpha1 = 1 - alpha;
}

void NoiseCrossSpectrum::add(const gsl_vector_complex *noiseSpectrum1, const gsl_vector_complex *noiseSpectrum2)
{
  if (noiseSpectrum1->size != noiseSpectrum2->size)
    throw jdimension_error("FFT length of noiseSpectra does not match (%d and %d).", noiseSpectrum1->size, noiseSpectrum2->size);
  if (crossSpectrum != NULL) {
    for (unsigned i = 0; i < crossSpectrum->size; i++) {
      gsl_complex n1 = gsl_vector_complex_get(noiseSpectrum1, i);
      gsl_complex n2 = gsl_complex_conjugate(gsl_vector_complex_get(noiseSpectrum2, i));
      gsl_complex v = gsl_complex_mul(n1, n2);
      
      gsl_vector_complex_set(crossSpectrum, i, gsl_complex_add(gsl_complex_mul_real(gsl_vector_complex_get(crossSpectrum, i), alpha),
							       gsl_complex_mul_real(v, alpha1)));
    }
  } else {
    int fftLen = noiseSpectrum1->size;
    int fftLen2 = (fftLen+1)/2;
    int len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;
    crossSpectrum = gsl_vector_complex_alloc(len);

    for (unsigned i = 0; i < len; i++) {
      gsl_complex n1 = gsl_vector_complex_get(noiseSpectrum1, i);
      gsl_complex n2 = gsl_complex_conjugate(gsl_vector_complex_get(noiseSpectrum2, i));
      gsl_complex v = gsl_complex_mul(n1, n2);
      
      gsl_vector_complex_set(crossSpectrum, i, gsl_complex_mul_real(v, alpha1));
    }
  }
}

GCC::GCC(double sampleRate, unsigned fftLen, unsigned nChan, unsigned pairs, double alpha, double beta, double q, bool interpolate, bool noisereduction)
{
  this->sampleRate = sampleRate;
  this->fftLen = fftLen;
  this->nChan = nChan;
  this->pairs = pairs;
  fftLen2 = (fftLen+1)/2;
  len = ((fftLen & 1) == 0)?(fftLen2+1):fftLen2;
  crossSpectrum = gsl_vector_complex_alloc(len);
  crossCorrelation = gsl_vector_alloc(fftLen);
  noisePowerSpectrum = new NoisePowerSpectrum[nChan];
  for (unsigned i=0; i<nChan; i++)
    noisePowerSpectrum[i] = NoisePowerSpectrum(alpha);
  noiseCrossSpectrum = new NoiseCrossSpectrum[pairs];
  for (unsigned i=0; i<pairs; i++)
    noiseCrossSpectrum[i] = NoiseCrossSpectrum(alpha);
  sad = true;
  this->beta = beta;
  beta1 = 1-beta;
  this->q = q;
  q1 = 1 - q;
  q2 = 2*q;
  this->interpolate = interpolate;
  if (interpolate)
    interpolValues = gsl_matrix_alloc(fftLen, 2);
  this->noisereduction = noisereduction;
  retValues = gsl_vector_alloc(3);
  gsl_vector_set_zero(retValues);
}
   
GCC::~GCC()
{
  gsl_vector_complex_free(crossSpectrum);
  gsl_vector_free(crossCorrelation);
  delete[] noisePowerSpectrum;
  delete[] noiseCrossSpectrum;
  if (interpolate)
    gsl_matrix_free(interpolValues);
  gsl_vector_free(retValues);
}


// calculates GCC
void GCC::calculate(const gsl_vector_complex *spectralSample1, unsigned chan1, const gsl_vector_complex *spectralSample2, unsigned chan2, unsigned pair, double timestamp, bool sad, bool smooth)
{
  if (sad) {
    if (spectralSample1->size != fftLen)
      throw jdimension_error("FFT length of spectralSample1 (%d) does not match %d.", spectralSample1->size, fftLen);
    if (spectralSample2->size != fftLen)
      throw jdimension_error("FFT length of spectralSample2 (%d) does not match %d.", spectralSample2->size, fftLen);

    const gsl_vector_complex *Gn1n2 = noiseCrossSpectrum[pair].get();
    const gsl_vector *N1 = noisePowerSpectrum[chan1].get();
    const gsl_vector *N2 = noisePowerSpectrum[chan2].get();

    for (unsigned i = 0; i < len; i++) {
      // get frequency-bin of first signal
      x1 = gsl_vector_complex_get(spectralSample1, i);
      // get frequency-bin of second signal
      x2 = gsl_vector_complex_get(spectralSample2, i);

      G = calcCrossSpectrumValue(x1, x2, Gn1n2, N1, N2, i);
      if (smooth) {
	gsl_vector_complex_set(crossSpectrum, i, gsl_complex_add(gsl_complex_mul_real(gsl_vector_complex_get(crossSpectrum, i), beta), gsl_complex_mul_real(G, beta1)));
      } else {
	gsl_vector_complex_set(crossSpectrum, i, G);
      }
    }
    pack_half_complex(gsl_vector_ptr(crossCorrelation, 0), crossSpectrum, fftLen);
    gsl_fft_halfcomplex_radix2_inverse(gsl_vector_ptr(crossCorrelation, 0), /* stride=*/ 1, fftLen);
  } else {
    noisePowerSpectrum[chan1].add(spectralSample1, timestamp);
    noisePowerSpectrum[chan2].add(spectralSample2, timestamp);
    noiseCrossSpectrum[pair].add(spectralSample1, spectralSample2);
  }
}

const gsl_vector* GCC::findMaximum(double minDelay, double maxDelay)
{
  maxCorr = -HUGE_VAL;
  maxCorr2 = -HUGE_VAL;
  delayPos = 0;
  delay = 0.0;
  for (unsigned i = 0; i < fftLen; i++) {
    unsigned idx;
    double del = 0.0;
    if (i<fftLen2) {
      idx = i + fftLen2;
      del = i / sampleRate;
    } else {
      idx = i - fftLen2;
      del = -((fftLen - i)/sampleRate);
    }

    double corr = gsl_vector_get(crossCorrelation, i);
    if (interpolate) {
      gsl_matrix_set(interpolValues, idx, 0, del);
      gsl_matrix_set(interpolValues, idx, 1, corr);
    }

    if ((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr)) {
      maxCorr2 = maxCorr;
      maxCorr  = corr;
      delayPos = idx;
      delay = del;
    }
    else if((del >= minDelay)&&(del <= maxDelay)&&(corr > maxCorr2)) {
      maxCorr2 = corr;
    } 
  }
  
  ratio = maxCorr / maxCorr2;

  if (interpolate)
    delay = getInterpolation(interpolValues, delayPos);

  gsl_vector_set(retValues, 0, delay);
  gsl_vector_set(retValues, 1, maxCorr);
  gsl_vector_set(retValues, 2, ratio);
  return retValues;
}

gsl_complex GCCRaw::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)
  return gsl_complex_mul(x1, gsl_complex_conjugate(x2));
}

gsl_complex GCCGnnSub::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)-Gn1n2[i]
  return gsl_complex_sub(gsl_complex_mul(x1, gsl_complex_conjugate(x2)), gsl_vector_complex_get(Gn1n2, i));
}

gsl_complex GCCPhat::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)/|x1*(x2*)|
  // calculate cross-spectrum coeffizient
  Gx1x2 = gsl_complex_mul(x1, gsl_complex_conjugate(x2));
  double w = gsl_complex_abs(Gx1x2);
  if (w == 0.0) {
    return gsl_complex_rect(0.0,0.0);
  } else
    return gsl_complex_div_real(Gx1x2, w);
}

gsl_complex GCCGnnSubPhat::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)-Gn1n2[i]/|x1*(x2*)-Gn1n2[i]|
  // calculate cross-spectrum coeffizient
  Gx1x2 = gsl_complex_mul(x1, gsl_complex_conjugate(x2));
  if (Gn1n2 != NULL)
    return gsl_complex_div_real(gsl_complex_sub(Gx1x2, gsl_vector_complex_get(Gn1n2, i)), gsl_complex_abs(gsl_complex_sub(Gx1x2, gsl_vector_complex_get(Gn1n2, i))));
  else
    return gsl_complex_div_real(Gx1x2, gsl_complex_abs(Gx1x2));
}

gsl_complex GCCMLRRaw::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)-Gn1n2[i]/|x1*(x2*)-Gn1n2[i]|
  // calculate cross-spectrum coeffizient
  Gx1x2 = gsl_complex_mul(x1, gsl_complex_conjugate(x2));
  // calculate power of first signal
  X1 = gsl_complex_abs(x1);
  // square of X1
  X12 = X1*X1;
  // calculate power of second signal
  X2 = gsl_complex_abs(x2);
  // square of X2
  X22 = X2*X2;
  if ((N1 != NULL)&&(N2 != NULL))
    return gsl_complex_mul_real(Gx1x2, X1*X2/(q2*X12*X22+q1*(gsl_vector_get(N2, i)*X12 + gsl_vector_get(N1, i)*X22)));
  else
    return gsl_complex_mul_real(Gx1x2, X1*X2/(q2*X12*X22));
}

gsl_complex GCCMLRGnnSub::calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i)
{
  // G = x1*(x2*)-Gn1n2[i]/|x1*(x2*)-Gn1n2[i]|
  // calculate cross-spectrum coeffizient
  Gx1x2 = gsl_complex_mul(x1, gsl_complex_conjugate(x2));
  // calculate power of first signal
  X1 = gsl_complex_abs(x1);
  // square of X1
  X12 = X1*X1;
  // calculate power of second signal
  X2 = gsl_complex_abs(x2);
  // square of X2
  X22 = X2*X2;
  if ((Gn1n2 != NULL)&&(N1 != NULL)&&(N2 != NULL))
    return gsl_complex_mul_real(gsl_complex_sub(Gx1x2, gsl_vector_complex_get(Gn1n2, i)), X1*X2/(q2*X12*X22+q1*(gsl_vector_get(N2, i)*X12 + gsl_vector_get(N1, i)*X22)));
  else
    return gsl_complex_mul_real(Gx1x2, X1*X2/(q2*X12*X22));
}
