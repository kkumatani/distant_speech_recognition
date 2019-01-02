/*
 * @file spectralestimator.cc
 * @brief Speech recognition front end.
 * @author John McDonough, Matthias Woelfel, Kenichi Kumatani
 */

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <float.h>
#include <math.h>
#include "spectralestimator.h"
#include "feature.h"

void calcAutoCorrelationVector( const gsl_vector_float *samples, gsl_vector *autoCorrelationVector )
{
  int nSamples = (int)samples->size;
  int maxLag = (int)autoCorrelationVector->size;

  for(int lag=0;lag<maxLag;lag++){
    float r_xx = 0.0;

    for(int i=lag;i<nSamples;i++){
      r_xx += ( gsl_vector_float_get( samples, i ) * gsl_vector_float_get( samples, i-lag ) );
    }
    gsl_vector_set( autoCorrelationVector, lag, r_xx / (float)nSamples );
  }

  return ;
}

float calcLPCswithLevinsonDurbin( gsl_vector *autoCorrelationVector, gsl_vector_float *lpcs )
{
  int i, j;  
  int p = (int)autoCorrelationVector->size; /* order + 1 */
  float  * ac = new float[p];    /* autocorrelation values */
  float  * ref = new float[p-1]; /* reflection coef's	 */
  float	 * lpc = new float[p-1]; /* LPCs                 */
  
  for(i=0;i<p;i++)
    ac[i] = gsl_vector_get( autoCorrelationVector, i );
  /* initialize the LP values */
  for(i=0;i<p-1;i++)
    lpc[i] = (float)0.0;

  float r;
  float error = ac[0]; /* the prediction error */

  if (ac[0] == 0) {
    for (i=0;i<p-1;i++) ref[i] = 0; 
    return 0; 
  }

  for(i=0;i<p-1;i++){ /* Sum up this iteration's reflection coefficient. */
    r = -ac[i+1];
    for(j=0;j<i;j++) 
      r -= lpc[j] * ac[i-j];
    ref[i] = r /= error;

    /* Update LPC coefficients and total error. */
    lpc[i] = r;
    for(j=0;j<i/2;j++) {
      double tmp  = lpc[j];
      lpc[j]     += r * lpc[i-1-j];
      lpc[i-1-j] += r * tmp;
    }
    if (i % 2) lpc[j] += lpc[j] * r;

    error *= 1.0 - r * r;
  }
		
  for (i=0;i<p-1;i++){
    gsl_vector_float_set( lpcs,i,lpc[i]);
  }

  delete [] ac;
  delete [] ref;
  delete [] lpc;

  return error;
}

const double LPCSpectrumEstimator::_epsilon = 10e-8;

LPCSpectrumEstimator::LPCSpectrumEstimator( const VectorFloatFeatureStreamPtr& samp, unsigned order, unsigned fftLen, const String& nm)
  : VectorFloatFeatureStream(fftLen, nm), _samp(samp), _order(order), _fftLen(fftLen), _fftLen2(fftLen/2),
    _r(gsl_vector_float_calloc(order)),
    _autoCorrelation(gsl_vector_calloc(order+1)),
    _lpcs(gsl_vector_float_alloc(order)),
    _wslpcs(new double[_fftLen]),
    _ftlpcs(gsl_vector_complex_calloc(fftLen))
{
}

LPCSpectrumEstimator::~LPCSpectrumEstimator()
{
  gsl_vector_float_free(_r);
  gsl_vector_free(_autoCorrelation);
  gsl_vector_float_free(_lpcs);
  delete [] _wslpcs;
  gsl_vector_complex_free(_ftlpcs);

}

void LPCSpectrumEstimator::reset() 
{
  VectorFloatFeatureStream::reset();
  gsl_vector_float_set_zero(_r);
  gsl_vector_set_zero(_autoCorrelation);
}

const gsl_vector_float* LPCSpectrumEstimator::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* samples = _samp->next(frame_no_+1);
  calcLPCs( samples );
  calcLPEnvelope();

  /*
  static char fileName[256];
  sprintf(fileName, "spectra/frame%04d.txt", frame_no);
  FILE* fp = fileOpen(fileName, "w");
  for (unsigned i = 0; i <= _fftLen / 2; i++)
    fprintf(fp, "%12.4f   %12.4f\n", 16000.0 * i / _fftLen, 10*log10(gsl_vector_float_get(vector_, i)));
  fileClose(fileName, fp);
  */

  increment_();
  return vector_;
}

/**
   @brief compute the LPCs. 
   @param const gsl_vector_float* samples[in] time-discrete signal, x[n].
   @return LPCs. The estimated signal is \hat{x}(n) = sum_{k=1}^{order} _vector[k-1] * x[n-k]
 */
const gsl_vector_float* LPCSpectrumEstimator::calcLPCs( const gsl_vector_float* samples )
{
  calcAutoCorrelationVector( samples, _autoCorrelation );
  for(int i=0;i<(int)_order;i++){
    gsl_vector_float_set( _r, i, gsl_vector_get( _autoCorrelation, i+1 ) );
  }
  _e = calcLPCswithLevinsonDurbin( _autoCorrelation, _lpcs );

  for(unsigned i = 0; i < _order; i++)	// Notice that this definition is different from the MATLAB
    gsl_vector_float_set( _lpcs, i, - gsl_vector_float_get( _lpcs, i ) );

  return (const gsl_vector_float* )_lpcs;
}

// @brief calculate the LP envelope from the LPCs (computation of Eq. (2))
// @return the LP envelope
void LPCSpectrumEstimator::calcLPEnvelope()
{
  gsl_vector_float*  lpEnvelope = vector_;

  // keep the LP coefficients and prediction error.
  _wslpcs[0] = -1;
  for (unsigned k = 1; k <= _order; k++)
    _wslpcs[k] = gsl_vector_float_get(_lpcs, k - 1 );
  for (unsigned k = _order+1; k < _fftLen; k++)
    _wslpcs[k] = 0.0;

  gsl_fft_real_radix2_transform(_wslpcs,  /* stride= */ 1, _fftLen);
  complexPackedArrayUnpack(_wslpcs, _ftlpcs);
  
  for (unsigned m = 0; m <= _fftLen2; m++) {
    double h2 = gsl_complex_abs2(gsl_vector_complex_get(_ftlpcs, m));
    double s2 = _e / (h2 + _epsilon); 

    gsl_vector_float_set( lpEnvelope, m, s2);
    if(m > 0 && m < _fftLen2)
      gsl_vector_float_set( lpEnvelope, _fftLen - m, s2);
  }
}

void LPCSpectrumEstimator::complexPackedArrayUnpack(const double* halfcomplex_coefficient, gsl_vector_complex* complex_coefficient) const
{
  gsl_vector_complex_set(complex_coefficient, 0, gsl_complex_rect(halfcomplex_coefficient[0], 0.0));

  for (unsigned i = 1; i < _fftLen2; i++) {
    double hc_real = halfcomplex_coefficient[i];
    double hc_imag = halfcomplex_coefficient[_fftLen - i];
    gsl_vector_complex_set(complex_coefficient,             i, gsl_complex_rect( hc_real,  hc_imag));
    gsl_vector_complex_set(complex_coefficient, (_fftLen - i), gsl_complex_rect( hc_real, -hc_imag));
  }

  gsl_vector_complex_set(complex_coefficient, _fftLen2, gsl_complex_rect(halfcomplex_coefficient[_fftLen2], 0.0));
}


// ----- methods for class `CepstralSpectrumEstimator' -----
//
CepstralSpectrumEstimator::
CepstralSpectrumEstimator(const VectorComplexFeatureStreamPtr& source, unsigned order, unsigned fftLen,
			  double logPadding, const String& nm)
  : VectorFloatFeatureStream(fftLen, nm), _source(source), _order(order), _fftLen(fftLen), _logPadding(logPadding),
    _samples(new double[_fftLen]), _spectrum(gsl_vector_complex_calloc(_fftLen)), _cepstrum(gsl_vector_complex_calloc(_fftLen)) { }

CepstralSpectrumEstimator::~CepstralSpectrumEstimator()
{
  delete[] _samples;
  gsl_vector_complex_free(_spectrum);
  gsl_vector_complex_free(_cepstrum);
}

const gsl_vector_float* CepstralSpectrumEstimator::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  // get next block of data and zero-pad if necessary
  const gsl_vector_complex* block = _source->next(frame_no_ + 1);
  increment_();

  // take log of the square-magnitude and repack
  for (unsigned i = 0; i < _fftLen; i++) {
    gsl_complex logSpectralSample = gsl_complex_rect(log(_logPadding + gsl_complex_abs2(gsl_vector_complex_get(block, i))), 0.0);
    gsl_vector_complex_set(_cepstrum, i, logSpectralSample);
  }
  pack_half_complex(_samples, _cepstrum, _fftLen);

  // inverse transform into the cepstral domain and truncate sequence
  gsl_fft_halfcomplex_radix2_inverse(_samples, /*stride=*/ 1, _fftLen);
  for (unsigned i = _order + 1; i < _fftLen - _order; i++)
    _samples[i] = 0.0;

  // one more forward transform to obtain the cepstral log-spectral envelope and unpack
  gsl_fft_real_radix2_transform(_samples, /*stride=*/ 1, _fftLen);
  unpack_half_complex(_spectrum, _samples);

  // exponentiate log-spectrum to obtain the cepstral spectral envelope
  for (unsigned i = 0; i < _fftLen; i++)
    gsl_vector_float_set(vector_, i, exp(gsl_complex_abs(gsl_vector_complex_get(_spectrum, i))));

  return vector_;
}


// ----- methods for class `SEMNB' -----
//
SEMNB::SEMNB( unsigned order, unsigned fftLen, const String& nm)
  : _order(order), _fftLen(fftLen), _fftLen2(fftLen/2),
    _derivative(gsl_vector_calloc(_fftLen2 + 1)),
    _R(gsl_matrix_calloc(_order, _order)),
    _Rcopy(gsl_matrix_calloc(_order, _order)),
    _workSpace(gsl_eigen_symmv_alloc(_order)),
    _evalues(gsl_vector_alloc(_order)), _evectors(gsl_matrix_calloc(_order, _order)),

    _r(gsl_vector_alloc(_order)),
    _dr_dPm(gsl_vector_alloc(_order)),
    _dak_dPm(gsl_vector_alloc(_order)),
    _Lambda_bar(gsl_matrix_calloc(_order, _order)),
    _dLambdamatrix(gsl_matrix_calloc(_order, _order)),

    _C(gsl_matrix_calloc(_order, _order)),
    _D(gsl_matrix_calloc(_order, _order)) { }

SEMNB::~SEMNB()
{
  gsl_vector_free(_derivative);
  gsl_matrix_free(_R);
  gsl_matrix_free(_Rcopy);
  gsl_eigen_symmv_free(_workSpace);
  gsl_vector_free(_evalues);
  gsl_matrix_free(_evectors);

  gsl_vector_free(_r);
  gsl_vector_free(_dr_dPm);
  gsl_vector_free(_dak_dPm);
  gsl_matrix_free(_Lambda_bar);
  gsl_matrix_free(_dLambdamatrix);

  gsl_matrix_free(_C);
  gsl_matrix_free(_D);
}

/** 
    @brief calculate the derivative of the deviation w.r.t the subband component over the entire frequency; See eqn. (28)
    @param LPCSpectrumEstimatorPtr &lpcSEPtr[in]
    @return the gradient vector
*/
const gsl_vector* SEMNB::calcDerivativeOfDeviation( LPCSpectrumEstimatorPtr &lpcSEPtr )
{
  const gsl_vector_float* lpcs = lpcSEPtr->getLPCs(); // lpcs LPC vector
  double epsilonp = (double)lpcSEPtr->getPredictionError(); // epsilonp variance of LPC residual  
  _ftlpcs = lpcSEPtr->getFTLPCs();
  _lpEnvelope = (gsl_vector*)lpcSEPtr->current();

  { // build the auto-correlation vector, _r, and matrix, _R.
    const gsl_vector_float *ac = lpcSEPtr->getAutoCorrelationVector();
    for (unsigned k = 0; k <_order; k++)
      gsl_vector_set( _r, k, gsl_vector_float_get( ac, k+1 ) );
    for (int rowX = 0; rowX < _order; rowX++)
      for (int colX = 0; colX < _order; colX++)
	gsl_matrix_set(_R, rowX, colX, gsl_vector_float_get(ac, abs(rowX - colX)));
  }

  lpcSEPtr->calcLPEnvelope();  _calcEigen();  _calcLambda_bar();

  for (unsigned m = 0; m <= _fftLen2; m++) {
    _da_dPm(m);
    gsl_vector_set(_derivative, m, _dSigma_dP(lpcs, epsilonp, m));
  }

  return _derivative;
}

// Calculate eigenvectors and eigenvalues of the autcorrelation matrix
void SEMNB::_calcEigen()
{
  gsl_matrix_memcpy(_Rcopy, _R);

  gsl_eigen_symmv(_Rcopy, _evalues, _evectors, _workSpace);

  gsl_eigen_symmv_sort(_evalues, _evectors, GSL_EIGEN_SORT_VAL_DESC);
}

// Eqn.(24)
double SEMNB::_dLambdak_dPm(unsigned k, unsigned m)
{
  double result = 0.0;
  int p 	= _order;

  for (int i = -(p-1); i <= p-1; i++) {
    double innerR = 0.0;
    for (int l = max(0, -i); l <= min(p-1, p-1-i); l++)
      innerR += gsl_matrix_get(_evectors, i+l, k) * gsl_matrix_get(_evectors, l, k);

    double der = (2.0 / _fftLen) * cos(2.0 * M_PI * m * i / _fftLen);

    result += innerR * der;
  }

  return result;
}

// Matrix with derived eigenvectors
void SEMNB::_dLambdabar_dPm(unsigned m)
{
  for (unsigned i = 0; i < _order; i++) {
    double lambda_i = gsl_vector_get(_evalues, i);
    double value = _dLambdak_dPm(i, m) / (lambda_i * lambda_i);
    gsl_matrix_set(_dLambdamatrix, i, i, -value);
  }
}

void SEMNB::_calcLambda_bar()
{
  for (unsigned n = 0; n < _order; n++)
    gsl_matrix_set(_Lambda_bar, n, n, 1.0 / gsl_vector_get(_evalues, n));
}

void SEMNB::_calc_dr_dPm(unsigned m)
{
  for (unsigned n = 0; n < _order; n++)
    gsl_vector_set(_dr_dPm, n, (2.0 / _fftLen) * cos(2 * M_PI * m * n / _fftLen));
}

// Eqn. (13)
double SEMNB::_dSp_dan(const gsl_vector_float* lpcs, unsigned m, unsigned n)
{
#define STUPID_CALC
#ifdef STUPID_CALC
  double sum2 = - cos(2.0 * M_PI * m * n / _fftLen);
  for (unsigned k = 0; k < _order; k++)
    sum2 += gsl_vector_float_get(lpcs, k) * cos(2.0 * M_PI * m * (n - k - 1) / _fftLen);
  sum2 *= 2;
#endif
  gsl_complex val = gsl_vector_complex_get(_ftlpcs, m );
  double sum = cos(2.0 * M_PI * m * n / _fftLen) * GSL_REAL(val) - sin(2.0 * M_PI * m * n / _fftLen) * GSL_IMAG(val);
  sum *= 2.0;

  //printf("%e,%e\n",sum2, sum);
  return sum;
}

// Eqn. (9)
double SEMNB::_dSp_dPm(const gsl_vector_float* lpcs, unsigned m)
{
  double sum = 0.0;
  for (unsigned n = 0; n < _order; n++){
    //sum += _dSp_dan(lpcs, m, n) * gsl_vector_get(_dak_dPm, n);
    double dSp = _dSp_dan(lpcs, m, n);
    sum += dSp * gsl_vector_get(_dak_dPm, n);
  }
  return sum;
}

// Eqn. (12)
double SEMNB::_depsilonp_dPm(const gsl_vector_float* lpcs, unsigned m)
{
  double sum = 0.0;
  for (unsigned n = 0; n < _order; n++)
    sum += gsl_vector_get(_r, n) * gsl_vector_get(_dak_dPm, n) + gsl_vector_float_get(lpcs, n) * gsl_vector_get(_dr_dPm, n);
  sum *= - 1.0;
  sum += (2.0 / _fftLen);
  return sum;
}

// Eqn. (26)
void SEMNB::_da_dPm(unsigned m)
{
  _dLambdabar_dPm(m);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _evectors, _dLambdamatrix, 0.0, _C);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, _C, _evectors, 0.0, _D);

  gsl_blas_dgemv(CblasNoTrans, 1.0, _D, _r, 0.0, _dak_dPm);
  _calc_dr_dPm(m);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _evectors, _Lambda_bar, 0.0, _C);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans,   1.0, _C, _evectors, 0.0, _D);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _D, _dr_dPm, 1.0, _dak_dPm);

}

// @brief calculate the derivative of the LP envelope w.r.t the subband component; Eqn. (8)
// @param const gsl_vector_float* lpcs linear prediction coefficients
// @param double epsilonp variance of LPC residual
// @param int m subband index
double SEMNB::_dPhat_dP(const gsl_vector_float* lpcs, double epsilonp, unsigned m)
{
  double Spm = gsl_vector_get(_lpEnvelope, m);
  double depsilonp = _depsilonp_dPm(lpcs, m);
  double dSp = _dSp_dPm(lpcs, m); 
  double result = ( depsilonp  - epsilonp * dSp / Spm ) / Spm;
  //double result = _depsilonp_dPm(lpcs, m) * Spm - epsilonp * _dSp_dPm(lpcs, m);
  //return result / (Spm * Spm);
  return result;
}

// @brief calculate the derivative of the deviation w.r.t the subband component; See eq. (28)
// @notice you must call _calcLPEnvelope before you use this method
// @param const gsl_vector* in LPC vector
// @param double epsilonp variance of LPC residual
// @param int fbinX the index of the frequency bin
float SEMNB::_dSigma_dP(const gsl_vector_float* lpcs, double epsilonp, unsigned m)
{
  double s2 = gsl_vector_get(_lpEnvelope, m);
  float result = ( 1.0 / (2.0 * sqrt(s2)) ) * _dPhat_dP(lpcs, epsilonp, m);

  return result;
}

void SEMNB::reset()
{
  gsl_matrix_set_zero(_R);
  gsl_vector_set_zero(_evalues);
  gsl_matrix_set_zero(_evectors);
}
