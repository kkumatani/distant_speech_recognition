/**
 * @file aec.h
 * @brief Cancelation of a voice prompt based on either NLMS or Kalman filter algorithms
 * @author John McDonough, Wei Chu and Kenichi Kumatani
 */

#ifndef AEC_H
#define AEC_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "common/jexception.h"
#include <gsl/gsl_eigen.h>

#include "stream/stream.h"
#include "btk.h"
#include "beamformer/tracker.h"

/**
* \defgroup NLMSAcousticEchoCancellationFeature NLMS Echo Cancellation Feature
*/
/*@{*/


// ----- definition for class `NLMSAcousticEchoCancellationFeature' -----
//
class NLMSAcousticEchoCancellationFeature : public VectorComplexFeatureStream {
 public:
   NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, double delta = 100.0, double epsilon = 1.0E-04, double threshold = 100.0, const String& nm = "AEC");
  virtual ~NLMSAcousticEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset() { played_->reset(); recorded_->reset(); gsl_vector_complex_set_zero(filterCoefficient_); }

private:
  bool update_(const gsl_complex Vk);

  VectorComplexFeatureStreamPtr         played_;                    // v(n)
  VectorComplexFeatureStreamPtr         recorded_;                  // a(n)

  unsigned                              fftLen_;
  unsigned                              fftLen2_;
  gsl_vector_complex*                   filterCoefficient_;

  const double                          delta_;
  const double                          epsilon_;
  const double                          threshold_;
};


typedef Inherit<NLMSAcousticEchoCancellationFeature, VectorComplexFeatureStreamPtr> NLMSAcousticEchoCancellationFeaturePtr;
/*@}*/


/**
* \defgroup KalmanFilterEchoCancellationFeature Kalman Filer Echo Cancellation Feature
*/
/*@{*/


// ----- definition for class `KalmanFilterEchoCancellationFeature' -----
//
class KalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
 public:
  KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, double beta = 0.95, double sigma2 = 100.0, double threshold = 100.0, const String& nm = "KFEchoCanceller");
  virtual ~KalmanFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset() { played_->reset(); recorded_->reset(); gsl_vector_complex_set_zero(filterCoefficient_); }

private:
  bool update_(const gsl_complex Vk);

  VectorComplexFeatureStreamPtr         played_;                    // v(n)
  VectorComplexFeatureStreamPtr         recorded_;                  // a(n)

  unsigned                              fftLen_;
  unsigned                              fftLen2_;

  gsl_vector_complex*                   filterCoefficient_;
  gsl_vector*                           sigma2_v_;
  gsl_vector*                           K_k_;

  const double                          beta_;
  const double                          threshold_;
  const double                          sigma2_u_;
};


typedef Inherit<KalmanFilterEchoCancellationFeature, VectorComplexFeatureStreamPtr> KalmanFilterEchoCancellationFeaturePtr;
/*@}*/

// ----- definition for class `BlockKalmanFilterEchoCancellationFeature' -----
//
class BlockKalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
  public:
    BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double threshold = 100.0, double amp4play = 1.0, const String& nm = "KFEchoCanceller");
    virtual ~BlockKalmanFilterEchoCancellationFeature();

    virtual const gsl_vector_complex* next(int frame_no = -5);

    virtual void reset()
    {
      played_->reset(); recorded_->reset();
    }

  protected:
    class ComplexBuffer_ {
      public:
    /*
        @brief Construct a circular buffer to hold past and current subband samples
        It keeps nsamp arrays which is completely updated with the period 'nsamp'.
        Each array holds actual values of the samples.
        @param unsigned len[in] is the size of each vector of samples
        @param unsigned nsamp[in] is the period of the circular buffer
    */
      ComplexBuffer_(unsigned len, unsigned sampleN)
        : len_(len), sampleN_(sampleN), zero_(sampleN_ - 1),
          samples_(new gsl_vector_complex*[sampleN_]), subbandSamples_(gsl_vector_complex_calloc(sampleN_))
      {
        for (unsigned i = 0; i < sampleN_; i++)
          samples_[i] = gsl_vector_complex_calloc(len_);
      }

      ~ComplexBuffer_()
      {
        for (unsigned i = 0; i < sampleN_; i++)
          gsl_vector_complex_free(samples_[i]);
        delete[] samples_;
        gsl_vector_complex_free(subbandSamples_);
      }

      gsl_complex sample(unsigned timeX, unsigned binX) const {
        unsigned idx = index_(timeX);
        const gsl_vector_complex* vec = samples_[idx];
        return gsl_vector_complex_get(vec, binX);
      }

      const gsl_vector_complex* getSamples(unsigned m)
      {
        for (unsigned timeX = 0; timeX < sampleN_; timeX++)
          gsl_vector_complex_set(subbandSamples_, timeX, sample(timeX, m));

        return subbandSamples_;
      }

      void nextSample(const gsl_vector_complex* s = NULL, double amp4play = 1.0 ) {
        zero_ = (zero_ + 1) % sampleN_;

        gsl_vector_complex* nextBlock = samples_[zero_];

        if (s == NULL) {
          gsl_vector_complex_set_zero(nextBlock);
        } else {
          if (s->size != len_)
            throw jdimension_error("'ComplexBuffer_': Sizes do not match (%d vs. %d)", s->size, len_);
          assert( s->size == len_ );
          gsl_vector_complex_memcpy(nextBlock, s);
          if( amp4play != 1.0 )
            gsl_blas_zdscal( amp4play, nextBlock );
        }
      }

      void zero() {
        for (unsigned i = 0; i < sampleN_; i++)
          gsl_vector_complex_set_zero(samples_[i]);
        zero_ = sampleN_ - 1;
      }

    private:
      unsigned index_(unsigned idx) const {
        assert(idx < sampleN_);
        unsigned ret = (zero_ + sampleN_ - idx) % sampleN_;
        return ret;
      }

      const unsigned                             len_;
      const unsigned                             sampleN_;
      unsigned                                   zero_; // index of most recent sample
      gsl_vector_complex**                       samples_;
      gsl_vector_complex*                        subbandSamples_;
    };

  static gsl_complex                             ComplexOne_;
  static gsl_complex                             ComplexZero_;

  bool update_(const gsl_vector_complex* Vk);
  void conjugate_(gsl_vector_complex* dest, const gsl_vector_complex* src) const;

  VectorComplexFeatureStreamPtr                 played_;   // v(n)
  VectorComplexFeatureStreamPtr                 recorded_; // a(n)

  unsigned                                      fftLen_;
  unsigned                                      fftLen2_;
  unsigned                                      sampleN_;

  ComplexBuffer_                                buffer_;

  gsl_vector_complex**                          filterCoefficient_;
  gsl_vector*                                   sigma2_v_;
  gsl_matrix_complex**                          K_k_;
  gsl_matrix_complex*                           K_k_k1_;

  const double                                  beta_;
  const double                                  threshold_;
  gsl_matrix_complex**                          Sigma2_u_;
  gsl_vector_complex*                           Gk_;
  gsl_vector_complex*                           scratch_;
  gsl_vector_complex*                           scratch2_;
  gsl_matrix_complex*                           scratchMatrix_;
  gsl_matrix_complex*                           scratchMatrix2_;
  double                                        amp4play_;
  double                                        floorVal_;
  int                                           skippedN_;
  int                                           maxSkippedN_;
};


typedef Inherit<BlockKalmanFilterEchoCancellationFeature, VectorComplexFeatureStreamPtr> BlockKalmanFilterEchoCancellationFeaturePtr;
/*@}*/

/**
* \defgroup DTDBlockKalmanFilterEchoCancellationFeature Block Kalman Filer Echo Cancellation Feature
*/
/*@{*/

// ----- definition for class `InformationFilterEchoCancellationFeature' -----
//
class InformationFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
 public:
  InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0, double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,  double amp4play = 1.0, const String& nm = "Information Echo Canceller");

  virtual ~InformationFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);

protected:
  static double _EigenValueThreshold;

  double updateBand_(const gsl_complex Ak, const gsl_complex Ek, int frame_no, unsigned m);
  void invert_(gsl_matrix_complex* matrix);
  static void printMatrix_(const gsl_matrix_complex* mat);
  static void printVector_(const gsl_vector_complex* vec);

  const double                smoothEk_; // for smoothing the error signal, ->1, less smooth
  const double                smoothSk_; // for smoothing the estimated signal
  const double                engTh_;    // threshold of energy

  double*                     snr_;
  double*                     EkEnergy_;
  double*                     SkEnergy_;
  const double                loading_;

  gsl_matrix_complex*         inverse_;
  gsl_eigen_hermv_workspace*  eigenWorkSpace_;
  gsl_vector*                 evalues_;

  gsl_vector_complex*         scratchInverse_;
  gsl_matrix_complex*         scratchMatrixInverse_;
  gsl_matrix_complex*         scratchMatrixInverse2_;
  gsl_matrix_complex*         matrixCopy_;
};

typedef Inherit<InformationFilterEchoCancellationFeature, BlockKalmanFilterEchoCancellationFeaturePtr> InformationFilterEchoCancellationFeaturePtr;


// ----- definition for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
class SquareRootInformationFilterEchoCancellationFeature : public InformationFilterEchoCancellationFeature {
 public:
  SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0, double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02, double amp4play = 1.0, const String& nm = "Square Root Information Echo Canceller");

  virtual ~SquareRootInformationFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);

private:
  const gsl_complex load_;
  static gsl_complex calcGivensRotation_(const gsl_complex& v1, const gsl_complex& v2, gsl_complex& c, gsl_complex& s);
  static void applyGivensRotation_(const gsl_complex& v1, const gsl_complex& v2, const gsl_complex& c, const gsl_complex& s,
                                   gsl_complex& v1p, gsl_complex& v2p);
  void negative_(gsl_matrix_complex* dest, const gsl_matrix_complex* src);
  void extractCovarianceState_(const gsl_matrix_complex* K_k, const gsl_vector_complex* sk, gsl_vector_complex* xk);

  void temporalUpdate_(unsigned m);
  void observationalUpdate_(unsigned m, const gsl_complex& Ak, double sigma2_v);
  void diagonalLoading_(unsigned m);

  gsl_vector_complex**                                informationState_;
};

typedef Inherit<SquareRootInformationFilterEchoCancellationFeature, InformationFilterEchoCancellationFeaturePtr> SquareRootInformationFilterEchoCancellationFeaturePtr;


// ----- definition for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
class DTDBlockKalmanFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
 public:
   DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0, double engTh = 100.0, double smooth = 0.9, double amp4play = 1.0, const String& nm = "DTDKFEchoCanceller");
  virtual ~DTDBlockKalmanFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);

private:
  double updateBand_(const gsl_complex Ak, const gsl_complex Ek, int frame_no);
  void conjugate_(gsl_vector_complex* dest, const gsl_vector_complex* src) const;

  const double        smoothSk_; // for smoothing the estimated signal
  const double        smoothEk_; // for smoothing the error signal, ->1, less smooth
  const double        engTh_;    // threshold of energy
  double              snr_;

  double              EkEnergy_;
  double              SkEnergy_;
  FILE*               fdb_;
};


typedef Inherit<DTDBlockKalmanFilterEchoCancellationFeature, BlockKalmanFilterEchoCancellationFeaturePtr> DTDBlockKalmanFilterEchoCancellationFeaturePtr;
/*@}*/

#endif // AEC_H
