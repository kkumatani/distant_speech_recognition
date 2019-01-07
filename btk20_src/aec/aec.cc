/*
 * @file aec.cc
 * @brief Cancelation of a voice prompt based on either NLMS or Kalman filter algorithms
 * @author John McDonough, Wei Chu and Kenichi Kumatani
 */

#include <string.h>
#include <math.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_linalg.h>
#include "matrix/gslmatrix.h"

#include "common/jpython_error.h"
#include "aec/aec.h"

// ----- methods for class `NLMSAcousticEchoCancellationFeature' -----
//
NLMSAcousticEchoCancellationFeature::
NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played,
                                    const VectorComplexFeatureStreamPtr& recorded,
                                    double delta, double epsilon, double threshold, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     played_(played), recorded_(recorded), fftLen_(played->size()), fftLen2_(fftLen_ / 2), filterCoefficient_(gsl_vector_complex_alloc(fftLen_)),
     delta_(delta), epsilon_(epsilon), threshold_(threshold) { }


NLMSAcousticEchoCancellationFeature::~NLMSAcousticEchoCancellationFeature()
{
  gsl_vector_complex_free(filterCoefficient_);
}

bool NLMSAcousticEchoCancellationFeature::update_(const gsl_complex Vk)
{
  double energy = gsl_complex_abs2(Vk);

  return (energy > threshold_);
}

const gsl_vector_complex* NLMSAcousticEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock = played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock = recorded_->next(frame_no_ + 1);

  for (unsigned k = 0; k <= fftLen2_; k++) {
    gsl_complex Vk = gsl_vector_complex_get(playBlock, k);
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, k);
    gsl_complex Rk = gsl_vector_complex_get(filterCoefficient_, k);
    
    gsl_complex Ek = gsl_complex_sub(Ak, gsl_complex_mul(Rk, Vk));
    if (k > 0 && k < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - k, gsl_complex_conjugate(Ek));

    gsl_vector_complex_set(vector_, k, Ek);

    if (update_(Vk)) {
      gsl_complex Gkhat = gsl_complex_div(Ak, Vk);
      gsl_complex dC    = gsl_complex_sub(Rk, Gkhat);
      double Vk2        = gsl_complex_abs2(Vk);
      double Ak2        = gsl_complex_abs2(Ak);

      gsl_complex deltaC = gsl_complex_mul_real(dC, epsilon_ * Vk2/(delta_ + Ak2));

      gsl_complex nC = gsl_complex_sub(Rk, deltaC);
      gsl_vector_complex_set(filterCoefficient_, k, nC);
      if (k > 0 && k < fftLen2_)
        gsl_vector_complex_set(filterCoefficient_, fftLen_ - k, gsl_complex_conjugate(nC));
    }
  }

  increment_();
  return vector_;
}


// ----- methods for class `KalmanFilterEchoCancellationFeature' -----
//
KalmanFilterEchoCancellationFeature::
KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played,
                                    const VectorComplexFeatureStreamPtr& recorded,
                                    double beta, double sigma2, double threshold, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     played_(played), recorded_(recorded), fftLen_(played->size()), fftLen2_(fftLen_ / 2),
     filterCoefficient_(gsl_vector_complex_calloc(fftLen_)), sigma2_v_(gsl_vector_calloc(fftLen_)),
     K_k_(gsl_vector_calloc(fftLen_)), beta_(beta), threshold_(threshold), sigma2_u_(sigma2)
{
  // Initialize variances
  for (unsigned m = 0; m < fftLen_; m++) {
    gsl_vector_set(sigma2_v_, m, sigma2);
    gsl_vector_set(K_k_, m, sigma2);
  }
}


KalmanFilterEchoCancellationFeature::~KalmanFilterEchoCancellationFeature()
{
  gsl_vector_complex_free(filterCoefficient_);
  gsl_vector_free(sigma2_v_);
  gsl_vector_free(K_k_);
}


bool KalmanFilterEchoCancellationFeature::update_(const gsl_complex Vk)
{
  double energy = gsl_complex_abs2(Vk);

  return (energy > threshold_);
}


const gsl_vector_complex* KalmanFilterEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock	= played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock	= recorded_->next(frame_no_ + 1);

  for (unsigned m = 0; m <= fftLen2_; m++) {
          gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
          gsl_complex Rk = gsl_vector_complex_get(filterCoefficient_, m);
    const gsl_complex Vk = gsl_vector_complex_get(playBlock, m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex Ek = gsl_complex_sub(Ak, gsl_complex_mul(Rk, Vk));
    gsl_vector_complex_set(vector_, m, Ek);
    if (m > 0 && m < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - m, gsl_complex_conjugate(Ek));

    if (update_(Vk)) {

      // Estimate the observation noise variance
      double Ek2      = gsl_complex_abs2(Ek);
      double sigma2_v = beta_ * gsl_vector_get(sigma2_v_, m) + (1.0 - beta_) * Ek2;
      gsl_vector_set(sigma2_v_, m, sigma2_v);

      // Calculate the Kalman gain
      double      Vk2		= gsl_complex_abs2(Vk);
      double      K_k_k1	= gsl_vector_get(K_k_, m) + sigma2_u_;
      double      sigma2_s	= Vk2 * K_k_k1 + sigma2_v;
      gsl_complex Gk		= gsl_complex_mul_real(gsl_complex_conjugate(Vk), K_k_k1 / sigma2_s);

      // Update the filter weight
      Rk = gsl_complex_add(Rk, gsl_complex_mul(Gk, Ek));
      gsl_vector_complex_set(filterCoefficient_, m, Rk);

      // Store the state estimation error variance for next the iteration
      double K_k = (1.0 - K_k_k1 * Vk2 / sigma2_s) * K_k_k1;
      gsl_vector_set(K_k_, m, K_k);
    }
  }

  increment_();
  return vector_;
}

// ----- methods for class `BlockKalmanFilterEchoCancellationFeature' -----
//
gsl_complex BlockKalmanFilterEchoCancellationFeature::ComplexOne_  = gsl_complex_rect(1.0, 0.0);
gsl_complex BlockKalmanFilterEchoCancellationFeature::ComplexZero_ = gsl_complex_rect(0.0, 0.0);

BlockKalmanFilterEchoCancellationFeature::
BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played,
                                         const VectorComplexFeatureStreamPtr& recorded,
                                         unsigned sampleN, double beta, double sigmau2, double sigmak2, double threshold, double amp4play, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     played_(played), recorded_(recorded), fftLen_(played->size()), fftLen2_(fftLen_ / 2), sampleN_(sampleN),
     buffer_(fftLen_, sampleN_), filterCoefficient_(new gsl_vector_complex*[fftLen_]),
     sigma2_v_(gsl_vector_calloc(fftLen_)), K_k_(new gsl_matrix_complex*[fftLen_]),
     K_k_k1_(gsl_matrix_complex_calloc(sampleN_, sampleN_)),
     beta_(beta), threshold_(threshold), Sigma2_u_(new gsl_matrix_complex*[fftLen_]),
     Gk_(gsl_vector_complex_calloc(sampleN_)),
     scratch_(gsl_vector_complex_calloc(sampleN_)),
     scratch2_(gsl_vector_complex_calloc(sampleN_)),
     scratchMatrix_(gsl_matrix_complex_calloc(sampleN_, sampleN_)),
     scratchMatrix2_(gsl_matrix_complex_calloc(sampleN_, sampleN_)),
     amp4play_(amp4play),skippedN_(0),maxSkippedN_(30)
{
  // Initialize variances
  for (unsigned m = 0; m < fftLen_; m++)
    gsl_vector_set(sigma2_v_, m, sigmau2);

  // Initialize subband-dependent covariance matrices
  for (unsigned m = 0; m < fftLen_; m++) {
    filterCoefficient_[m] = gsl_vector_complex_calloc(sampleN_);
    K_k_[m]               = gsl_matrix_complex_calloc(sampleN_, sampleN_);
    Sigma2_u_[m]          = gsl_matrix_complex_calloc(sampleN_, sampleN_);

    for (unsigned n = 0; n < sampleN_; n++) {
      gsl_matrix_complex_set(K_k_[m], n, n, gsl_complex_rect(sigmak2, 0.0));
      gsl_matrix_complex_set(Sigma2_u_[m], n, n, gsl_complex_rect(sigmau2, 0.0));
    }
  }
}


BlockKalmanFilterEchoCancellationFeature::~BlockKalmanFilterEchoCancellationFeature()
{
  gsl_vector_free(sigma2_v_);
  gsl_vector_complex_free(Gk_);
  gsl_matrix_complex_free(K_k_k1_);
  gsl_vector_complex_free(scratch_);
  gsl_vector_complex_free(scratch2_);
  gsl_matrix_complex_free(scratchMatrix_);
  gsl_matrix_complex_free(scratchMatrix2_);
  for (unsigned m = 0; m < fftLen_; m++) {
    gsl_vector_complex_free(filterCoefficient_[m]);
    gsl_matrix_complex_free(K_k_[m]);
    gsl_matrix_complex_free(Sigma2_u_[m]);
  }
  delete[] filterCoefficient_;
  delete[] K_k_;
  delete[] Sigma2_u_;
}


bool BlockKalmanFilterEchoCancellationFeature::update_(const gsl_vector_complex* Vk)
{
  double energy = gsl_complex_abs2(gsl_vector_complex_get(Vk, /* sampleX= */ 0));

  return (energy > threshold_);
}


void BlockKalmanFilterEchoCancellationFeature::conjugate_(gsl_vector_complex* dest, const gsl_vector_complex* src) const
{
  if (src->size != dest->size)
    throw jdimension_error("BlockKalmanFilterEchoCancellationFeature::conjugate_:: Vector sizes (%d vs. %d) do not match\n", src->size, dest->size);
  for (unsigned n = 0; n < dest->size; n++)
    gsl_vector_complex_set(dest, n, gsl_complex_conjugate((gsl_vector_complex_get(src, n))));
}


const gsl_vector_complex* BlockKalmanFilterEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock	= played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock	= recorded_->next(frame_no_ + 1);
  buffer_.next_sample(playBlock,amp4play_);

  for (unsigned m = 0; m <= fftLen2_; m++) {
    gsl_complex         Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex* Rk = filterCoefficient_[m];
    const gsl_vector_complex* Vk = buffer_.get_samples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(vector_, m, Ek);
    if (m > 0 && m < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - m, gsl_complex_conjugate(Ek));

    double Ek2;
    if (update_(Vk)) {

      // Estimate the observation noise variance
      Ek2      = gsl_complex_abs2(Ek);
      double sigma2_v = beta_ * gsl_vector_get(sigma2_v_, m) + (1.0 - beta_) * Ek2;
      gsl_vector_set(sigma2_v_, m, sigma2_v);

      // Calculate the Kalman gain
      gsl_matrix_complex_memcpy(K_k_k1_, Sigma2_u_[m]);
      gsl_matrix_complex_add(K_k_k1_, K_k_[m]);

      conjugate_(scratch2_, Vk);
      gsl_blas_zgemv(CblasNoTrans, ComplexOne_, K_k_k1_, scratch2_, ComplexZero_, scratch_);
      gsl_blas_zdotu(Vk, scratch_, &iprod);

      double sigma2_s = GSL_REAL(iprod) + sigma2_v;
      gsl_vector_complex_set_zero(Gk_);
      gsl_blas_zaxpy(gsl_complex_rect(1.0 / sigma2_s, 0.0), scratch_, Gk_);

      // Update the filter weights
      gsl_blas_zaxpy(Ek, Gk_, Rk);

      // Store the state estimation error variance for next the iteration
      gsl_matrix_complex_set_zero(scratchMatrix_);
      for (unsigned rowX = 0; rowX < sampleN_; rowX++) {
        for (unsigned colX = 0; colX < sampleN_; colX++) {
          gsl_complex diagonal = ((rowX == colX) ? ComplexOne_ : ComplexZero_);
          gsl_complex value    =  gsl_complex_sub(diagonal, gsl_complex_mul(gsl_vector_complex_get(Gk_, rowX), gsl_vector_complex_get(Vk, colX)));
          gsl_matrix_complex_set(scratchMatrix_, rowX, colX, value);
        }
      }
      gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, ComplexOne_, scratchMatrix_, K_k_k1_, ComplexZero_, K_k_[m]);
    }
  }

  increment_();
  return vector_;
}


// ----- methods for class `InformationFilterEchoCancellationFeature' -----
//
InformationFilterEchoCancellationFeature::
InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played,
                                         const VectorComplexFeatureStreamPtr& recorded,
                                         unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth, double loading, double amp4play, const String& nm)
  : BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, amp4play, nm),
    smoothEk_(smooth), smoothSk_(smooth), engTh_(engTh),
    snr_(new double[fftLen_]), EkEnergy_(new double[fftLen_]), SkEnergy_(new double[fftLen_]), loading_(loading),
    inverse_(gsl_matrix_complex_calloc(sampleN_, sampleN_)), eigenWorkSpace_(gsl_eigen_hermv_alloc(sampleN_)), evalues_(gsl_vector_calloc(sampleN_)),
    scratchInverse_(gsl_vector_complex_calloc(sampleN_)),
    scratchMatrixInverse_(gsl_matrix_complex_calloc(sampleN_, sampleN_)),
    scratchMatrixInverse2_(gsl_matrix_complex_calloc(sampleN_, sampleN_)),
    matrixCopy_(gsl_matrix_complex_calloc(sampleN_, sampleN_))
{
  for (unsigned m = 0; m < fftLen_; m++) {
    snr_[m] = EkEnergy_[m] = SkEnergy_[m] = 0.0;

    // initialize filter coefficients
    gsl_vector_complex_set(filterCoefficient_[m], /* n= */ 0, gsl_complex_rect(1.0, 0.0));
    for (unsigned n = 1; n < sampleN_; n++)
      gsl_vector_complex_set(filterCoefficient_[m], n, gsl_complex_rect(/* 1.0e-04 */ 0.0, 0.0));
  }

  floorVal_ = 0.01;
}


InformationFilterEchoCancellationFeature::~InformationFilterEchoCancellationFeature()
{
  delete[] snr_;
  delete[] EkEnergy_;
  delete[] SkEnergy_;

  gsl_matrix_complex_free(inverse_);
  gsl_eigen_hermv_free(eigenWorkSpace_);
  gsl_vector_complex_free(scratchInverse_);
  gsl_matrix_complex_free(scratchMatrixInverse_);
  gsl_matrix_complex_free(scratchMatrixInverse2_);
  gsl_matrix_complex_free(matrixCopy_);
}

void InformationFilterEchoCancellationFeature::print_matrix_(const gsl_matrix_complex* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      gsl_complex value = gsl_matrix_complex_get(mat, m, n);
      printf("%8.4f %8.4f  ", GSL_REAL(value), GSL_IMAG(value));
    }
    printf("\n");
  }
}

void InformationFilterEchoCancellationFeature::print_vector_(const gsl_vector_complex* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    gsl_complex value = gsl_vector_complex_get(vec, n);
    printf("%8.4f %8.4f\n", GSL_REAL(value), GSL_IMAG(value));
  }
}

double InformationFilterEchoCancellationFeature::update_band_(const gsl_complex Ak, const gsl_complex Ek, int frame_no, unsigned m)
{
  double smthEk, smthSk;
  // if it is the first 100 frames
  if (frame_no < 100) {
    smthEk = 1.0 - (double) frame_no * (1.0 - smoothEk_) / 100.0;
    smthSk = 1.0 - (double) frame_no * (1.0 - smoothSk_) / 100.0;
  } else {
    smthEk = smoothEk_;
    smthSk = smoothSk_;
  }

  double sf;
  const gsl_complex Sk = gsl_complex_sub(Ak, Ek);
  double currEkEng = gsl_complex_abs2(Ek);
  double currSkEng = gsl_complex_abs2(Sk); // Ek to Ak
  EkEnergy_[m] = currEkEng * smthEk + EkEnergy_[m] * (1.0 - smthEk);
  SkEnergy_[m] = currSkEng * smthSk + SkEnergy_[m] * (1.0 - smthSk);
  double currSnr = currSkEng / (currEkEng + 1.0e-15);
  snr_[m] = currSnr * smthEk + snr_[m] * (1.0 - smthEk);
  if (frame_no < 100 || (snr_[m] > threshold_ && SkEnergy_[m] > engTh_))
    sf = 2.0 / (1.0 + exp(-snr_[m])) - 1.0;             // snr -> inf, sf -> 1; snr-> 0
  else
    sf = -1.0;

  return sf;
}

double InformationFilterEchoCancellationFeature::_EigenValueThreshold = 1.0e-06;

void InformationFilterEchoCancellationFeature::invert_(gsl_matrix_complex* matrix)
{
  // perform eigen decomposition
  gsl_matrix_complex_memcpy(matrixCopy_, matrix);
  gsl_eigen_hermv(matrixCopy_, evalues_, scratchMatrixInverse_, eigenWorkSpace_);

  // find maximum eigenvalue
  double maxEvalue = 0.0;
  for (unsigned n = 0; n < sampleN_; n++) {
    double value = gsl_vector_get(evalues_, n);
    if (value > maxEvalue)
      maxEvalue = value;
  }

  // scale columns by inverse of eigenvector
  for (unsigned n = 0; n < sampleN_; n++) {
    double value = gsl_vector_get(evalues_, n);
    // if ((value / maxEvalue) < _EigenValueThreshold) continue;
    double scale = 1.0 / value;
    for (unsigned m = 0; m < sampleN_; m++) {
      gsl_matrix_complex_set(scratchMatrixInverse2_, m, n, gsl_complex_mul_real(gsl_matrix_complex_get(scratchMatrixInverse_, m, n), scale));
    }
  }

  // final matrix-matrix multiply to get the psuedo-inverse
  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, ComplexOne_, scratchMatrixInverse2_, scratchMatrixInverse_, ComplexZero_, inverse_);

  /*
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, ComplexOne_, matrix, inverse_, ComplexZero_, scratchMatrixInverse_);
  gsl_matrix_complex_fprintf(stdout, scratchMatrixInverse_, "%8.4e");
  printf("Done\n");
  */
}

const gsl_vector_complex* InformationFilterEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock	= played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock	= recorded_->next(frame_no_ + 1);
  buffer_.next_sample(playBlock,amp4play_);

  for (unsigned m = 0; m <= fftLen2_; m++) {
    gsl_complex			Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex*		Rk = filterCoefficient_[m];
    const gsl_vector_complex*	Vk = buffer_.get_samples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    double absEk   = gsl_complex_abs(Ek);
    if ( absEk < floorVal_ )
      Ek = gsl_complex_rect( GSL_REAL(Ek)/absEk, GSL_IMAG(Ek)/absEk );

    gsl_vector_complex_set(vector_, m, Ek);
    if (m > 0 && m < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - m, gsl_complex_conjugate(Ek));

    if (update_(Vk) == false || update_band_(Ak, Ek, frame_no, m) < 0.0){
      if( skippedN_ >= maxSkippedN_ ){
	// initialize filter coefficients
	gsl_vector_complex_set(filterCoefficient_[m], 0, gsl_complex_rect(1.0, 0.0));
	for (unsigned n = 1; n < sampleN_; n++)
	  gsl_vector_complex_set(filterCoefficient_[m], n, gsl_complex_rect(/* 1.0e-04 */ 0.0, 0.0));
	skippedN_ = 0;
      }
      skippedN_++;
      continue;
    }

    // Estimate the observation noise variance
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= beta_ * gsl_vector_get(sigma2_v_, m) + (1.0 - beta_) * Ek2;
    gsl_vector_set(sigma2_v_, m, sigma2_v);

    // Perform the prediction step; scratch_ = y_{k|k-1}, and inverse_ = Y_{k|k-1}
    gsl_matrix_complex_memcpy(K_k_k1_, Sigma2_u_[m]);
    gsl_matrix_complex_add(K_k_k1_, K_k_[m]);
    invert_(K_k_k1_);
    gsl_blas_zgemv(CblasNoTrans, ComplexOne_, inverse_, Rk, ComplexZero_, scratch_);

    // form the matrix I_k = scratchMatrix_ and vector i_k = scratch2_
    double scale = 1.0 / sigma2_v;
    for (unsigned rowX = 0; rowX < sampleN_; rowX++) {
      gsl_complex value = gsl_complex_mul_real(gsl_complex_conjugate(gsl_vector_complex_get(Vk, rowX)), scale);
      gsl_vector_complex_set(scratch2_, rowX, gsl_complex_mul(value, Ak));

      for (unsigned colX = 0; colX < sampleN_; colX++) {
        gsl_complex colV = gsl_vector_complex_get(Vk, colX);
        gsl_matrix_complex_set(scratchMatrix_, rowX, colX, gsl_complex_mul(value, colV));
      }
    }

    // now perform the information correction/update step
    gsl_matrix_complex_add(scratchMatrix_, inverse_);
    gsl_vector_complex_add(scratch_, scratch2_);

    // extra diagonal loading to limit the size of the filter coefficients
    static const gsl_complex load = gsl_complex_rect(loading_, 0.0);
    for (unsigned diagX = 0; diagX < sampleN_; diagX++) {
      gsl_complex diagonal = gsl_complex_add(gsl_matrix_complex_get(scratchMatrix_, diagX, diagX), load);
      gsl_matrix_complex_set(scratchMatrix_, diagX, diagX, diagonal);
    }

    // extract filter coefficients from information vector and store
    invert_(scratchMatrix_);
    gsl_matrix_complex_memcpy(K_k_[m], inverse_);
    gsl_blas_zgemv(CblasNoTrans, ComplexOne_, inverse_, scratch_, ComplexZero_, Rk);
  }

  increment_();
  return vector_;
}


// ----- methods for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
SquareRootInformationFilterEchoCancellationFeature::
SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
                                                   unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth,
                                                   double loading, double amp4play, const String& nm)
  : InformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm),
    load_(gsl_complex_rect(sqrt(loading_), 0.0)),
    informationState_(new gsl_vector_complex*[fftLen_])
{
  // Reallocated scratchMatrix_ and scratchMatrix2_ for the temporal and observational updates respectively
  gsl_matrix_complex_free(scratchMatrix_);
  gsl_matrix_complex_free(scratchMatrix2_);
  scratchMatrix_  = gsl_matrix_complex_calloc((2 * sampleN_) + 1, 2 * sampleN_);
  scratchMatrix2_ = gsl_matrix_complex_calloc(sampleN_ + 1, sampleN_ + 1);

  // Initialize subband-dependent covariance matrices with the inverse Cholesky factors
  gsl_complex diagonal = gsl_complex_rect(1.0 / sqrt(sigmau2), 0.0);
  for (unsigned m = 0; m < fftLen_; m++) {
    gsl_matrix_complex_set_zero(K_k_[m]);
    gsl_matrix_complex_set_zero(Sigma2_u_[m]);
    for (unsigned n = 0; n < sampleN_; n++) {
      gsl_matrix_complex_set(K_k_[m], n, n, diagonal);
      gsl_matrix_complex_set(Sigma2_u_[m], n, n, diagonal);
    }

    informationState_[m] = gsl_vector_complex_calloc(sampleN_);
  }
}

SquareRootInformationFilterEchoCancellationFeature::~SquareRootInformationFilterEchoCancellationFeature()
{
  for (unsigned m = 0; m < fftLen_; m++)
    gsl_vector_complex_free(informationState_[m]);

  delete[] informationState_;
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
gsl_complex SquareRootInformationFilterEchoCancellationFeature::
calc_givens_rotation_(const gsl_complex& v1, const gsl_complex& v2,
                    gsl_complex& c, gsl_complex& s)
{
  double norm = sqrt(gsl_complex_abs2(v1) + gsl_complex_abs2(v2));

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = gsl_complex_div_real(v1, norm);
  s = gsl_complex_div_real(gsl_complex_conjugate(v2), norm);

  return gsl_complex_rect(norm, 0.0);
}

// apply a previously calculated Givens rotation
void SquareRootInformationFilterEchoCancellationFeature::
apply_givens_rotation_(const gsl_complex& v1, const gsl_complex& v2,
		     const gsl_complex& c, const gsl_complex& s,
		     gsl_complex& v1p, gsl_complex& v2p)
{
  v1p =
    gsl_complex_add(gsl_complex_mul(gsl_complex_conjugate(c), v1),
		    gsl_complex_mul(s, v2));
  v2p =
    gsl_complex_sub(gsl_complex_mul(c, v2),
		    gsl_complex_mul(gsl_complex_conjugate(s), v1));
}

// extract covariance state from square-root information state
void SquareRootInformationFilterEchoCancellationFeature::
extract_covariance_state_(const gsl_matrix_complex* K_k, const gsl_vector_complex* sk, gsl_vector_complex* xk)
{
  for (int sampX = sampleN_ - 1; sampX >= 0; sampX--) {
    gsl_complex skn = gsl_complex_conjugate(gsl_vector_complex_get(sk, sampX));
    for (int n = sampleN_ - 1; n > sampX; n--) {
      gsl_complex xkn    = gsl_vector_complex_get(xk, n);
      gsl_complex K_k_mn = gsl_complex_conjugate(gsl_matrix_complex_get(K_k, n, sampX));
      skn = gsl_complex_sub(skn, gsl_complex_mul(K_k_mn, xkn));
    }
    gsl_vector_complex_set(xk, sampX, gsl_complex_div(skn, gsl_complex_conjugate(gsl_matrix_complex_get(K_k, sampX, sampX))));
  }
}

void SquareRootInformationFilterEchoCancellationFeature::
negative_(gsl_matrix_complex* dest, const gsl_matrix_complex* src)
{
  for (unsigned rowX = 0; rowX < sampleN_; rowX++) {
    for (unsigned colX = 0; colX < sampleN_; colX++) {
      gsl_complex value = gsl_matrix_complex_get(src, rowX, colX);
      gsl_matrix_complex_set(dest, rowX, colX, gsl_complex_rect(-GSL_REAL(value), -GSL_IMAG(value)));
    }
  }
}

const gsl_vector_complex* SquareRootInformationFilterEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock	= played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock	= recorded_->next(frame_no_ + 1);
  buffer_.next_sample(playBlock,amp4play_);

  for (unsigned m = 0; m <= fftLen2_; m++) {
    gsl_complex			Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex*		Rk = filterCoefficient_[m];
    const gsl_vector_complex*	Vk = buffer_.get_samples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(vector_, m, Ek);
    if (m > 0 && m < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - m, gsl_complex_conjugate(Ek));

    if (update_(Vk) == false || update_band_(Ak, Ek, frame_no, m) < 0.0) continue;

    // Estimate the observation noise variance
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= beta_ * gsl_vector_get(sigma2_v_, m) + (1.0 - beta_) * Ek2;
    gsl_vector_set(sigma2_v_, m, sigma2_v);

    // perform prediction, correction, and add diagonal loading
    temporal_update_(m);
    observational_update_(m, Ak, sigma2_v);
    diagonal_loading_(m);

    // extract filter coefficients from information vector and store
    extract_covariance_state_(K_k_[m], informationState_[m], Rk);
  }

  increment_();
  return vector_;
}

static gsl_complex ComplexZero = gsl_complex_rect(0.0, 0.0);

void SquareRootInformationFilterEchoCancellationFeature::temporal_update_(unsigned m)
{
  // copy in elements of the pre-array
  gsl_matrix_complex_set_zero(scratchMatrix_);
  gsl_matrix_complex_view A11(gsl_matrix_complex_submatrix(scratchMatrix_,  /* k1= */ 0, /* k2= */ 0,
                                                           /* n1= */ sampleN_, /* n2= */ sampleN_));
  gsl_matrix_complex_memcpy(&A11.matrix, Sigma2_u_[m]);

  gsl_matrix_complex_view A12(gsl_matrix_complex_submatrix(scratchMatrix_,  /* k1= */ 0, /* k2= */ sampleN_,
                                                           /* n1= */ sampleN_, /* n2= */ sampleN_));
  negative_(&A12.matrix, K_k_[m]);

  gsl_matrix_complex_view A22(gsl_matrix_complex_submatrix(scratchMatrix_,  /* k1= */ sampleN_, /* k2= */ sampleN_,
                                                           /* n1= */ sampleN_, /* n2= */ sampleN_));
  gsl_matrix_complex_memcpy(&A22.matrix, K_k_[m]);

  gsl_vector_complex_view A32(gsl_matrix_complex_subrow(scratchMatrix_, /* rowX= */ 2 * sampleN_, /* offsetX= */ sampleN_, /* columnsN= */ sampleN_));
  gsl_vector_complex_memcpy(&A32.vector, informationState_[m]);

  // zero out A12
  for (unsigned colX = 0; colX < sampleN_; colX++) {
    for (unsigned rowX = colX; rowX < sampleN_; rowX++) {
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(&A11.matrix, rowX, rowX);
      gsl_complex v2 = gsl_matrix_complex_get(&A12.matrix, rowX, colX);
      gsl_matrix_complex_set(&A11.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
      gsl_matrix_complex_set(&A12.matrix, rowX, colX, ComplexZero);

      for (unsigned n = rowX + 1; n <= 2 * sampleN_; n++) {
	gsl_complex v1p, v2p;
	v1 = gsl_matrix_complex_get(scratchMatrix_, n, rowX);
	v2 = gsl_matrix_complex_get(scratchMatrix_, n, colX + sampleN_);

	apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
	gsl_matrix_complex_set(scratchMatrix_, n, rowX, v1p);
	gsl_matrix_complex_set(scratchMatrix_, n, colX + sampleN_, v2p);
      }
    }
  }

  // lower triangularize A22
  for (unsigned rowX = 0; rowX < sampleN_ - 1; rowX++) {
    for (unsigned colX = sampleN_ - 1; colX > rowX; colX--) {
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(&A22.matrix, rowX, rowX);
      gsl_complex v2 = gsl_matrix_complex_get(&A22.matrix, rowX, colX);
      gsl_matrix_complex_set(&A22.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
      gsl_matrix_complex_set(&A22.matrix, rowX, colX, ComplexZero);

      for (unsigned n = rowX + 1; n <= sampleN_; n++) {
        gsl_complex v1p, v2p;
        v1 = gsl_matrix_complex_get(scratchMatrix_, sampleN_ + n, sampleN_ + rowX);
        v2 = gsl_matrix_complex_get(scratchMatrix_, sampleN_ + n, sampleN_ + colX);

        apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
        gsl_matrix_complex_set(scratchMatrix_, sampleN_ + n, sampleN_ + rowX, v1p);
        gsl_matrix_complex_set(scratchMatrix_, sampleN_ + n, sampleN_ + colX, v2p);
      }
    }
  }

  // copy out inverse Cholesky factor and information state vector
  gsl_matrix_complex_memcpy(K_k_[m], &A22.matrix);
  gsl_vector_complex_memcpy(informationState_[m], &A32.vector);
}

void SquareRootInformationFilterEchoCancellationFeature::observational_update_(unsigned m, const gsl_complex& Ak, double sigma2_v)
{
  // copy in elements of the pre-array
  gsl_matrix_complex_view A11(gsl_matrix_complex_submatrix(scratchMatrix2_, /* k1= */ 0, /* k2= */ 0, /* n1= */ sampleN_, /* n2= */ sampleN_));
  gsl_matrix_complex_memcpy(&A11.matrix, K_k_[m]);

  gsl_vector_complex_view a12(gsl_matrix_complex_subcolumn(scratchMatrix2_, /* colX= */ sampleN_, /* offsetX= */ 0, /* columnsN= */ sampleN_));
  conjugate_(&a12.vector, buffer_.get_samples(m));
  double scale = 1.0 / sqrt(sigma2_v);
  for (unsigned n = 0; n < sampleN_; n++) {
    gsl_complex value = gsl_complex_mul_real(gsl_vector_complex_get(&a12.vector, n), scale);
    gsl_vector_complex_set(&a12.vector, n, value);
  }

  gsl_vector_complex_view a21(gsl_matrix_complex_subrow(scratchMatrix2_, /* rowX= */ sampleN_, /* offsetX= */ 0, /* columnsN= */ sampleN_));
  gsl_vector_complex_memcpy(&a21.vector, informationState_[m]);

  gsl_complex Akstar(gsl_complex_mul_real(gsl_complex_conjugate(Ak), scale));
  gsl_matrix_complex_set(scratchMatrix2_, sampleN_, sampleN_, Akstar);

  // zero out a12
  for (unsigned rowX = 0; rowX < sampleN_; rowX++) {
    gsl_complex c, s;
    gsl_complex v1 = gsl_matrix_complex_get(&A11.matrix, rowX, rowX);
    gsl_complex v2 = gsl_vector_complex_get(&a12.vector, rowX);
    gsl_matrix_complex_set(&A11.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
    gsl_vector_complex_set(&a12.vector, rowX, ComplexZero);

    for (unsigned n = rowX + 1; n <= sampleN_; n++) {
      gsl_complex v1p, v2p;
      v1 = gsl_matrix_complex_get(scratchMatrix2_, n, rowX);
      v2 = gsl_matrix_complex_get(scratchMatrix2_, n, sampleN_);

      apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
      gsl_matrix_complex_set(scratchMatrix2_, n, rowX, v1p);
      gsl_matrix_complex_set(scratchMatrix2_, n, sampleN_, v2p);
    }
  }

  // copy out inverse Cholesky factor and information state vector
  gsl_vector_complex_memcpy(informationState_[m], &a21.vector);
  gsl_matrix_complex_memcpy(K_k_[m], &A11.matrix);
}

void SquareRootInformationFilterEchoCancellationFeature::diagonal_loading_(unsigned m)
{
  gsl_matrix_complex* A = K_k_[m];
  for (unsigned diagX = 0; diagX < sampleN_; diagX++) {
    gsl_vector_complex_set_zero(scratch_);
    gsl_vector_complex_set(scratch_, diagX, load_);

    for (unsigned colX = diagX; colX < sampleN_; colX++) {
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(A, colX, colX);
      gsl_complex v2 = gsl_vector_complex_get(scratch_, colX);
      gsl_matrix_complex_set(A, colX, colX, calc_givens_rotation_(v1, v2, c, s));
      gsl_vector_complex_set(scratch_, colX, ComplexZero_);

      for (unsigned rowX = colX + 1; rowX < sampleN_; rowX++) {
        gsl_complex v1p, v2p;
        v1 = gsl_matrix_complex_get(A, rowX, colX);
        v2 = gsl_vector_complex_get(scratch_, rowX);

        apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
        gsl_matrix_complex_set(A, rowX, colX, v1p);
        gsl_vector_complex_set(scratch_, rowX, v2p);
      }
    }
  }
}


// ----- methods for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
DTDBlockKalmanFilterEchoCancellationFeature::
DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played,
                                            const VectorComplexFeatureStreamPtr& recorded,
                                            unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth, double amp4play, const String& nm)
  : BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, amp4play, nm),
    smoothSk_(smooth), smoothEk_(smooth), engTh_(engTh), snr_(0), EkEnergy_(0), SkEnergy_(0), fdb_(NULL)
{
  // if debug open
  fdb_ = fopen("/home/wei/src/wav/debug.txt", "w");
}


DTDBlockKalmanFilterEchoCancellationFeature::~DTDBlockKalmanFilterEchoCancellationFeature()
{
  if (fdb_ != NULL)
    fclose(fdb_);
  printf("finished.\n");
}


double DTDBlockKalmanFilterEchoCancellationFeature::update_band_(const gsl_complex Ak, const gsl_complex Ek, int frame_no)
{
  double smthEk, smthSk;
  // if it is the first 100 frames
  if (frame_no < 100) {
    smthEk = 1.0 - (double) frame_no * (1.0 - smoothEk_) / 100.0;
    smthSk = 1.0 - (double) frame_no * (1.0 - smoothSk_) / 100.0;
  } else {
    smthEk = smoothEk_;
    smthSk = smoothSk_;
  }

  double sf;
  const gsl_complex Sk = gsl_complex_sub(Ak, Ek);
  double currEkEng = gsl_complex_abs2(Ek);
  double currSkEng = gsl_complex_abs2(Sk); // Ek to Ak
  EkEnergy_ = currEkEng * smthEk + EkEnergy_ * (1.0 - smthEk);
  SkEnergy_ = currSkEng * smthSk + SkEnergy_ * (1.0 - smthSk);
  double currSnr = currSkEng / (currEkEng + 1.0e-15);
  snr_ = currSnr * smthEk + snr_ * (1.0 - smthEk);
  if (frame_no < 100 || (snr_ > threshold_ && SkEnergy_ > engTh_))
    sf = 2.0 / (1.0 + exp(-snr_)) - 1.0;             // snr -> inf, sf -> 1; snr-> 0
  else
    sf = -1.0;

  if (fdb_ != NULL) {
    fwrite(&sf, sizeof(double), 1, fdb_);
    fwrite(&snr_, sizeof(double), 1, fdb_);
    fwrite(&SkEnergy_, sizeof(double), 1, fdb_);
    fwrite(&EkEnergy_, sizeof(double), 1, fdb_);
  }
  return sf;
}


void DTDBlockKalmanFilterEchoCancellationFeature::conjugate_(gsl_vector_complex* dest, const gsl_vector_complex* src) const
{
  if (src->size != dest->size)
    throw jdimension_error("BlockKalmanFilterEchoCancellationFeature::conjugate_:: Vector sizes (%d vs. %d) do not match\n", src->size, dest->size);
  for (unsigned n = 0; n < dest->size; n++)
    gsl_vector_complex_set(dest, n, gsl_complex_conjugate((gsl_vector_complex_get(src, n))));
}


const gsl_vector_complex* DTDBlockKalmanFilterEchoCancellationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* playBlock	= played_->next(frame_no_ + 1);
  const gsl_vector_complex* recordBlock	= recorded_->next(frame_no_ + 1);
  buffer_.next_sample(playBlock,amp4play_);

  // Ek is stored in the _vector
  for (unsigned m = 0; m <= fftLen2_; m++) {
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex *Rk = filterCoefficient_[m];
    const gsl_vector_complex *Vk = buffer_.get_samples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(vector_, m, Ek);
    if (m > 0 && m < fftLen2_)
      gsl_vector_complex_set(vector_, fftLen_ - m, gsl_complex_conjugate(Ek));
  }

  for (unsigned m = 0; m <= fftLen2_; m++) {
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex *Rk = filterCoefficient_[m];
    const gsl_vector_complex *Vk = buffer_.get_samples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex Ek = gsl_vector_complex_get(vector_, m);
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);

    double sf = update_band_(Ak, Ek, frame_no);

    if (sf < 0.0) continue;

    // Estimate the observation noise variance
    gsl_complex zsf	= gsl_complex_rect(sf, 0.0);
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= beta_ * gsl_vector_get(sigma2_v_, m) + (1.0 - beta_) * Ek2;
    gsl_vector_set(sigma2_v_, m, sigma2_v);

    // Calculate the Kalman gain
    gsl_matrix_complex_memcpy(K_k_k1_, Sigma2_u_[m]);
    gsl_matrix_complex_scale(K_k_k1_, zsf);
    gsl_matrix_complex_add(K_k_k1_, K_k_[m]);

    conjugate_(scratch2_, Vk);
    gsl_blas_zgemv(CblasNoTrans, ComplexOne_, K_k_k1_, scratch2_, ComplexZero_, scratch_);
    gsl_blas_zdotu(Vk, scratch_, &iprod);

    double sigma2_s = GSL_REAL(iprod) + sigma2_v;
    gsl_vector_complex_set_zero(Gk_);
    gsl_blas_zaxpy(gsl_complex_rect(1.0 / sigma2_s, 0.0), scratch_, Gk_);

    // Update the filter weights
    gsl_blas_zaxpy(Ek, Gk_, Rk);

    // Store the state estimation error variance for next the iteration
    gsl_matrix_complex_set_zero(scratchMatrix_);
    for (unsigned rowX = 0; rowX < sampleN_; rowX++) {
      for (unsigned colX = 0; colX < sampleN_; colX++) {
        gsl_complex diagonal = ((rowX == colX) ? ComplexOne_ : ComplexZero_);
        gsl_complex value    =  gsl_complex_sub(diagonal, gsl_complex_mul(gsl_vector_complex_get(Gk_, rowX), gsl_vector_complex_get(Vk, colX)));
        gsl_matrix_complex_set(scratchMatrix_, rowX, colX, value);
      }
    }
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, ComplexOne_, scratchMatrix_, K_k_k1_, ComplexZero_, K_k_[m]);
  }

  increment_();
  return vector_;
}
