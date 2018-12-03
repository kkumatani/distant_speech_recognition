/**
 * @file postfilter.h

 * @brief implementation of post-filters for a microphone array. 

   The following post-filters are implemented:
   [1] Zelinski post-filter
   [2] APAB post-filter
   [3] McCowan's post-filter
   [4] Lefkimmiatis's post-filter

   The correspondig references are:
   [1] C.Claude Marro et al. "Analysis of noise reduction and dereverberation techniques based on microphone arrays with postfiltering", IEEE Trans. ASP, vol. 6, pp 240-259, May 1998.
   [2] M.Brandstein, "Microphone Arrays", Springer, ISBN 3-540-41953-5, pp.39-60.
   [3] Iain A. Mccowan et al., "Microphone array post-filter based on noise field coherence", IEEE Trans. SAP, vol. 11, pp. Nov. 709--716, 2003.
   [4] Stamatios Lefkimmiatis et al., "A generalized estimation approach for linear and nonlinear microphone array post-filters",  Speech Communication, 2007.

  * @author Kenichi Kumatani
*/
#ifndef POSTFILTER_H
#define POSTFILTER_H

#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "postfilter/spectralsubtraction.h"
#include "postfilter/binauralprocessing.h"
#include "beamformer/spectralinfoarray.h"
#include "beamformer/beamformer.h"

typedef enum {
  TYPE_ZELINSKI1_REAL = 0x01,
  TYPE_ZELINSKI1_ABS  = 0x02,
  TYPE_APAB = 0x04,
  TYPE_ZELINSKI2 = 0x08,
  NO_USE_POST_FILTER = 0x00
} PostfilterType;

void ZelinskiFilter(gsl_vector_complex **arrayManifold,
		    SnapShotArrayPtr     snapShotArray, 
		    bool halfBandShift, 
		    gsl_vector_complex *beamformedSignal,
		    gsl_vector_complex **prevCSDs, 
		    gsl_vector_complex *pfweights,
		    double alpha, int Ropt );

void ApabFilter( gsl_vector_complex **arrayManifold,
		 SnapShotArrayPtr     snapShotArray, 
		 int fftLen, int nChan, bool halfBandShift,
		 gsl_vector_complex *beamformedSignal,
		 int channelX );

/**
   @class Zelinski post-filtering

   @brief filter beamformer's outputs under the assumption that noise signals between sensors are uncorrelated. 
   @usage
   1. construct an object,
   2. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   3. process data at each frame by caling next().
*/
class ZelinskiPostFilter: public VectorComplexFeatureStream {
public:
  ZelinskiPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, int minFrames=0, const String& nm = "ZelinskPostFilter" );
  ~ZelinskiPostFilter();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void set_beamformer(SubbandDSPtr &beamformer);
  void set_snapshot_array(SnapShotArrayPtr &snapShotArray);
  void set_array_manifold_vector(unsigned fbinX, gsl_vector_complex *arrayManifoldVector, bool halfBandShift, unsigned NC = 1);
  const gsl_vector_complex* postfilter_weights(){
    if( NULL == bf_weights_ )
      return NULL;
    return(bf_weights_->wp1());
  }

#ifdef ENABLE_LEGACY_BTK_API
  void setBeamformer(SubbandDSPtr &beamformer){ set_beamformer(beamformer); }
  void setSnapShotArray(SnapShotArrayPtr &snapShotArray){ set_snapshot_array(snapShotArray); }
  void setArrayManifoldVector(unsigned fbinX, gsl_vector_complex *arrayManifoldVector, bool halfBandShift, unsigned NC = 1){ set_array_manifold_vector(fbinX, arrayManifoldVector, halfBandShift, NC); }
  const gsl_vector_complex* getPostFilterWeights(){ return postfilter_weights(); }
#endif

protected:
  unsigned                      fftLen_;
  VectorComplexFeatureStreamPtr samp_; /* output of the beamformer */
  PostfilterType                type_; /* the type of the Zelinski-postfilters */
  double                        alpha_; /* forgetting factor */
  int                           min_frames_;
  SubbandDSPtr                  bf_ptr_; /* */
  BeamformerWeights*            bf_weights_;
  bool                          has_bf_ptr_; /* true if bf_ptr_ is set with setBeamformer() */
  SnapShotArrayPtr              snapshot_array_; /* multi-channel input */
};

typedef Inherit<ZelinskiPostFilter, VectorComplexFeatureStreamPtr> ZelinskiPostFilterPtr;

/**
   @class McCowan post-filtering

   @brief process the beamformer's outputs with McCowan's post-filtering
   @usage
   1. construct an object,
   2. compute the noise coherence matrix through setDiffuseNoiseModel( micPositions, ssampleRate)
   3. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   4. process data at each frame by caling next().
*/
class McCowanPostFilter: public ZelinskiPostFilter {
public:
  McCowanPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, int minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilter" );
  ~McCowanPostFilter();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  /* micPositions[][x,y,z] */

  const gsl_matrix_complex *noise_spatial_spectral_matrix( unsigned fbinX );
  bool set_noise_spatial_spectral_matrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool set_diffuse_noise_model( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void set_all_diagonal_loading(float diagonalWeight);
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  void divide_all_nondiagonal_elements(float mu);
  void divide_nondiagonal_elements(unsigned fbinX, float mu);

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix(unsigned fbinX){ return noise_spatial_spectral_matrix(fbinX); }
  bool setNoiseSpatialSpectralMatrix(unsigned fbinX, gsl_matrix_complex* Rnn){ return set_noise_spatial_spectral_matrix(fbinX, Rnn); }
  bool setDiffuseNoiseModel(const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0){ return set_diffuse_noise_model( micPositions, sampleRate, sspeed); }
  void setAllLevelsOfDiagonalLoading(float diagonalWeight){ set_all_diagonal_loading(diagonalWeight); }
  void setLevelOfDiagonalLoading(unsigned fbinX, float diagonalWeight){ set_diagonal_looading(fbinX, diagonalWeight); }
  void divideAllNonDiagonalElements( float mu );
  void divideNonDiagonalElements( unsigned fbinX, float mu );
#endif

protected:
  double estimate_average_clean_PSD_( unsigned fbinX, gsl_vector_complex* currCSDf );
  virtual void post_filtering_();

protected:
  gsl_matrix_complex**                           R_; /* Noise spatial spectral matrices */
  float*                                         diagonal_weights_;
  float                                          threshold_of_Rij_; /* to avoid the indeterminate solution*/
  gsl_vector_complex*                            time_aligned_signal_f_; /* workspace */
  bool                                           invR_computed_;
};

typedef Inherit<McCowanPostFilter, ZelinskiPostFilterPtr> McCowanPostFilterPtr;

/**
   @class Lefkimmiatis post-filtering
   @brief compute a Winer filter under the the diffuse noise field assumption
   @usage
   1. construct an object,
   2. compute the noise coherence matrix through setDiffuseNoiseModel( micPositions, ssampleRate)
   3. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   4. process data at each frame by caling next().
*/
class LefkimmiatisPostFilter: public McCowanPostFilter {
public:
  LefkimmiatisPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, int minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilte" );
  ~LefkimmiatisPostFilter();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  /* micPositions[][x,y,z] */

  void calc_inverse_noise_spatial_spectral_matrix();

#ifdef ENABLE_LEGACY_BTK_API
  void calcInverseNoiseSpatialSpectralMatrix(){ calc_inverse_noise_spatial_spectral_matrix(); }
#endif

protected:
  double estimate_average_noise_PSD_( unsigned fbinX, gsl_vector_complex* currCSDf );
  virtual void post_filtering_();

private:
  gsl_complex calcLambda( unsigned fbinX );

  gsl_matrix_complex** invR_;
  gsl_vector_complex*  tmpH_;
  double              minSV_;
  unsigned            fbinX1_;
};

typedef Inherit<LefkimmiatisPostFilter, McCowanPostFilterPtr> LefkimmiatisPostFilterPtr;

/**
   @class high pass filter
*/
class HighPassFilter: public VectorComplexFeatureStream {
public:
  HighPassFilter( VectorComplexFeatureStreamPtr &output, float cutOffFreq, int sampleRate, const String& nm = "HighPassFilter" );
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

private:
  VectorComplexFeatureStreamPtr src_;
  unsigned cutoff_fbinX_;
};

typedef Inherit<HighPassFilter, VectorComplexFeatureStreamPtr> HighPassFilterPtr;

#endif
