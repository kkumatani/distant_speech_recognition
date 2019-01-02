/**
 * @file beamformer.h
 * @brief Beamforming in the subband domain.
 * @author John McDonough and Kenichi Kumatani
 */
#ifndef BEAMFORMER_H
#define BEAMFORMER_H

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
#include "beamformer/spectralinfoarray.h"
#include "modulated/modulated.h"

#define SSPEED 343740.0

class BeamformerWeights {
public:
  BeamformerWeights( unsigned fftLen, unsigned chanN, bool halfBandShift, unsigned NC = 1 );
  ~BeamformerWeights();

  void calcMainlobe(  float sampleRate, const gsl_vector* delays,  bool isGSC );
  void calcMainlobe2( float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ, bool isGSC );
  void calcMainlobeN( float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC, bool isGSC );
  void calcSidelobeCancellerP_f( unsigned fbinX, const gsl_vector* packedWeight );
  void calcSidelobeCancellerU_f( unsigned fbinX, const gsl_vector_complex* wa );
  void calcBlockingMatrix( unsigned fbinX );

  bool write_fir_coeff(const String& fn, unsigned winType);

#ifdef ENABLE_LEGACY_BTK_API
  bool writeFIRCoeff(const String& fn, unsigned winType);
#endif

  void setSidelobeCanceller_f( unsigned fbinX, gsl_vector_complex* wl_f ){
    gsl_vector_complex_memcpy( wl_[fbinX], wl_f );
  }
  void setQuiescentVector( unsigned fbinX, gsl_vector_complex *wq_f, bool isGSC=false );
  void setQuiescentVectorAll( gsl_complex z, bool isGSC=false );
  void setTimeAlignment();

  bool     isHalfBandShift() const {return(halfBandShift_);}
  unsigned NC() const {return(NC_);}
  unsigned fftLen() const {return(fftLen_);}
  unsigned chanN() const {return(chanN_);}

  gsl_vector_complex** arrayManifold() const { return (ta_); }
  gsl_vector_complex* wq_f( unsigned fbinX ) const { return wq_[fbinX]; }
  gsl_vector_complex* wl_f( unsigned fbinX ) const { return wl_[fbinX]; }
  gsl_vector_complex** wq() const { return (wq_); }
  gsl_matrix_complex** B() const { return (B_); }
  gsl_vector_complex** wa() const { return (wa_); }
  gsl_vector_complex** CSDs() const { return CSDs_; }
  gsl_vector_complex* wp1() const { return wp1_; }

private:
  void alloc_weights_();
  void free_weights_();

  unsigned fftLen_;
  unsigned chanN_;
  bool halfBandShift_;
  unsigned NC_; // the numbef of constraints
  gsl_vector_complex** wq_; // a quiescent weight vector for each frequency bin, wq_[fbinX][chanN]
  gsl_matrix_complex** B_;  // a blocking matrix for each frequency bin,         B_[fbinX][chanN][chanN-NC]
  gsl_vector_complex** wa_; // an active weight vector for each frequency bin,   wa_[fbinX][chanN-NC]
  gsl_vector_complex** wl_; // wl_[fbinX] = B_[fbinX] * wa_[fbinX]
  gsl_vector_complex** ta_; // do time alignment for multi-channel waves. It is also called an array manifold. _ta[fbinX][chanN].
  gsl_vector_complex*  wp1_;  // a weight vector of postfiltering,   _wp[fbinX]
  gsl_vector_complex** CSDs_; // cross spectral density for the post-filtering
};

typedef refcount_ptr<BeamformerWeights>     BeamformerWeightsPtr;


// ----- definition for class `SubbandBeamformer' -----
// 
class SubbandBeamformer : public VectorComplexFeatureStream {
 public:
  SubbandBeamformer(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandBeamformer");
  ~SubbandBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  unsigned fftLen() const { return fftLen_; }
  unsigned fftLen2() const { return fftLen2_; }
  unsigned chanN() const { return channelList_.size(); }
  virtual unsigned dim() const { return chanN();}

  bool is_end() const {return is_end_;}
  const gsl_vector_complex* snapshot_array_f(unsigned fbinX) const { return (snapshot_array_->snapshot(fbinX)); }
  virtual SnapShotArrayPtr  snapshot_array() const { return(snapshot_array_); }
  void         set_channel(VectorComplexFeatureStreamPtr& chan);
  virtual void clear_channel();

#ifdef ENABLE_LEGACY_BTK_API
  bool isEnd() { return is_end(); }
  const gsl_vector_complex* snapShotArray_f(unsigned fbinX){ return snapshot_array_f(fbinX); }
  virtual SnapShotArrayPtr  getSnapShotArray(){ return(snapshot_array()); }
  void         setChannel(VectorComplexFeatureStreamPtr& chan){ set_channel(chan); }
  virtual void clearChannel(){ clear_channel(); }
#endif

protected:
  typedef list<VectorComplexFeatureStreamPtr>	ChannelList_;
  typedef ChannelList_::iterator		ChannelIterator_;

  SnapShotArrayPtr				snapshot_array_;
  unsigned					fftLen_;
  unsigned					fftLen2_;
  bool						halfBandShift_;
  ChannelList_					channelList_;
};

// ----- definition for class `SubbandDS' -----
// 

class SubbandDS : public SubbandBeamformer {
 public:
  SubbandDS(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandDS");
  ~SubbandDS();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual void clear_channel();
  virtual const gsl_vector_complex *get_weights(unsigned fbinX) const { return bfweight_vec_[0]->wq_f(fbinX); }
  virtual BeamformerWeights* beamformer_weight_object(unsigned srcX=0) const { return bfweight_vec_[srcX]; }

  virtual void calc_array_manifold_vectors(float sampleRate, const gsl_vector* delays);
  virtual void calc_array_manifold_vectors_2(float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ);
  virtual void calc_array_manifold_vectors_n(float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2);

#ifdef ENABLE_LEGACY_BTK_API
  virtual void clearChannel(){ clear_channel(); }
  virtual const gsl_vector_complex *getWeights(unsigned fbinX) const { return get_weights(fbinX); }
  virtual BeamformerWeights* getBeamformerWeightObject(unsigned srcX=0) const { return beamformer_weight_object(srcX); }
  virtual void calcArrayManifoldVectors(float sampleRate, const gsl_vector* delays){
    calc_array_manifold_vectors(sampleRate, delays);
  }
  virtual void calcArrayManifoldVectors2(float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ){
    calc_array_manifold_vectors_2(sampleRate, delaysT, delaysJ);
  }
  virtual void calcArrayManifoldVectorsN(float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2){
    calc_array_manifold_vectors_n(sampleRate, delaysT, delaysJ, NC);
  }
#endif /* #ifdef ENABLE_LEGACY_BTK_API */

protected:
  void alloc_image_();
  void alloc_bfweight_(int nSrc, int NC);

  vector<BeamformerWeights *>                   bfweight_vec_; // weights of a beamformer per source.
};

#define NO_PROCESSING 0x00
#define SCALING_MDP   0x01
class SubbandGSC : public SubbandDS {
public:
  SubbandGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandGSC")
    : SubbandDS( fftLen, halfBandShift, nm ),normalize_weight_(false){}
  ~SubbandGSC();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  void normalize_weight(bool flag){ normalize_weight_ = flag; }
  void set_quiescent_weights_f(unsigned fbinX, const gsl_vector_complex* srcWq);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);
  void zero_active_weights();
  void calc_gsc_weights(float sampleRate, const gsl_vector* delaysT);
  void calc_gsc_weights_2(float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ);
  void calc_gsc_weights_n(float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2);

  bool write_fir_coeff(const String& fn, unsigned winType=1);
  gsl_matrix_complex* blocking_matrix(unsigned srcX, unsigned fbinX){
    return (bfweight_vec_[srcX]->B())[fbinX];
  }

#ifdef ENABLE_LEGACY_BTK_API
  void normalizeWeight(bool flag){ normalize_weight(flag); }
  void setQuiescentWeights_f(unsigned fbinX, const gsl_vector_complex * srcWq){ set_quiescent_weights_f(fbinX, srcWq); }
  void setActiveWeights_f(unsigned fbinX, const gsl_vector* packedWeight){ set_active_weights_f(fbinX, packedWeight); }
  void zeroActiveWeights(){ zero_active_weights(); }
  void calcGSCWeights(float sampleRate, const gsl_vector* delaysT){ calc_gsc_weights(sampleRate, delaysT); }
  void calcGSCWeights2(float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ){ calc_gsc_weights_2(sampleRate, delaysT, delaysJ); }
  void calcGSCWeightsN(float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2){ calc_gsc_weights_n(sampleRate, delaysT, delaysJ, NC); }
  bool writeFIRCoeff(const String& fn, unsigned winType=1){ return write_fir_coeff(fn, winType); }
  gsl_matrix_complex* getBlockingMatrix(unsigned srcX, unsigned fbinX){ return blocking_matrix(srcX, fbinX); }
#endif /* #ifdef ENABLE_LEGACY_BTK_API */

protected:
  bool normalize_weight_;
};

/**
   @class SubbandGSCRLS
   @brief implementation of recursive least squares of a GSC
   @usage
   1. calcGSCWeights()
   2. initPrecisionMatrix() or setPrecisionMatrix()
   3. update_sctive_weight_vecotrs( false ) if you want to stop adapting the active weight vectors.
   @note notations are  based on Van Trees, "Optimum Array Processing", pp. 766-767.
 */
typedef enum {
  CONSTANT_NORM           = 0x01,
  THRESHOLD_LIMITATION    = 0x02,
  NO_QUADRATIC_CONSTRAINT = 0x00
} QuadraticConstraintType;

// ----- definition for class `SubbandGSCRLS' -----
// 

class SubbandGSCRLS : public SubbandGSC {
 public:
  SubbandGSCRLS(unsigned fftLen = 512, bool halfBandShift = false, float mu = 0.9, float sigma2=0.0, const String& nm = "SubbandGSCRLS");
  ~SubbandGSCRLS();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void init_precision_matrix(float sigma2 = 0.01);
  void set_precision_matrix(unsigned fbinX, gsl_matrix_complex *Pz);
  void update_active_weight_vecotrs(bool flag){ is_wa_updated_ = flag; }
  void set_quadratic_constraint(float alpha, int qctype=1){ alpha_=alpha; qctype_=(QuadraticConstraintType)qctype; }

#ifdef ENABLE_LEGACY_BTK_API
  void initPrecisionMatrix(float sigma2 = 0.01){ init_precision_matrix(sigma2); }
  void setPrecisionMatrix(unsigned fbinX, gsl_matrix_complex *Pz){ set_precision_matrix(fbinX, Pz); }
  void updateActiveWeightVecotrs(bool flag){ update_active_weight_vecotrs(flag); }
  void setQuadraticConstraint(float alpha, int qctype=1){ set_quadratic_constraint(alpha, qctype); }
#endif /* #ifdef ENABLE_LEGACY_BTK_API */

private:
  void update_active_weight_vector2_(int frame_no); /* the case of the half band shift = False */
  bool alloc_subbandGSCRLS_image_();
  void free_subbandGSCRLS_image_();

  gsl_vector_complex** gz_; /* Gain vectors */
  gsl_matrix_complex** Pz_; /* Precision matrices */
  gsl_vector_complex* Zf_;  /* output of the blocking matrix at each frequency */
  gsl_vector_complex* wa_;
  float  mu_;              /* Exponential factor for the covariance matrix */
  float* diagonal_weights_;
  float  alpha_;            /* Weight for the quadratic constraint*/
  QuadraticConstraintType qctype_;
  bool is_wa_updated_;

  /* work space for updating active weight vectors */
  gsl_vector_complex* PzH_Z_;
  gsl_matrix_complex* _I;
  gsl_matrix_complex* mat1_;
};

// ----- definition for class `SubbandMMI' -----
//

class SubbandMMI : public SubbandDS {
public:
  SubbandMMI(unsigned fftLen = 512, bool halfBandShift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, float alpha=0.9, const String& nm = "SubbandMMI")
    : SubbandDS( fftLen, halfBandShift, nm ),
    targetSourceX_(targetSourceX),
    nSource_(nSource),
    pftype_(pfType),
    alpha_(alpha),
    use_binary_mask_(false),
    binary_mask_type_(0),
    interference_outputs_(NULL),
    avg_output_(NULL)
  {}

  ~SubbandMMI();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  void use_binary_mask(float avgFactor=-1.0, unsigned fwidth=1, unsigned type=0);
  void calc_weights(  float sampleRate, const gsl_matrix* delays);
  void calc_weights_n( float sampleRate, const gsl_matrix* delays, unsigned NC=2);
  void set_hi_active_weights_f(unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0);
  void set_active_weights_f(unsigned fbinX, const gsl_matrix* packedWeights, int option=0);

#ifdef ENABLE_LEGACY_BTK_API
  void useBinaryMask(float avgFactor=-1.0, unsigned fwidth=1, unsigned type=0){ use_binary_mask(avgFactor, fwidth, type); }
  void calcWeights(  float sampleRate, const gsl_matrix* delays){ calc_weights(sampleRate, delays); }
  void calcWeightsN( float sampleRate, const gsl_matrix* delays, unsigned NC=2){ calc_weights_n(sampleRate, delays, NC); }
  void setHiActiveWeights_f(unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0){
    set_hi_active_weights_f(fbinX, pkdWa, pkdwb, option);
  }
  void setActiveWeights_f(unsigned fbinX, const gsl_matrix* packedWeights, int option=0){
    set_active_weights_f(fbinX, packedWeights, option);
  }
#endif /* #ifdef ENABLE_LEGACY_BTK_API */

private:
  void calc_interference_outputs_();
  void binary_masking_( gsl_vector_complex** interferenceOutputs, unsigned targetSourceX, gsl_vector_complex* output );

  unsigned                                      targetSourceX_; // the n-th source will be emphasized
  unsigned                                      nSource_;       // the number of sound sources
  int                                           pftype_;
  float                                        alpha_;
  bool                                          use_binary_mask_; // true if you use a binary mask
  unsigned                                      binary_mask_type_;// 0:use GSC's outputs, 1:use outputs of the upper branch.
  gsl_vector_complex**                          interference_outputs_;
  gsl_vector_complex*                           avg_output_;
  float                                        avg_factor_;
  unsigned                                      fwidth_;
};


// ----- definition for class `SubbandMVDR' -----
//

/**
   @class SubbandMVDR 

   @usage
   1. setChannel()
   2. calc_array_manifold_vectors(), calc_array_manifold_vectors2() or calc_array_manifold_vectorsN().
   3. set_noise_spatial_spectral_matrix() or set_diffuse_noise_model()
   4. calc_mvdr_weights()
 */
class SubbandMVDR : public SubbandDS {
 public:
  /**
     @brief Basic MVDR beamformer implementation
     @param int fftLen[in]
     @param bool halfBandShift[in]
   */
  SubbandMVDR(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDR();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void clear_channel();
  bool calc_mvdr_weights(float sampleRate, float dThreshold = 1.0E-8, bool calcInverseMatrix = true);
  const gsl_vector_complex* mvdir_weights(unsigned fbinX) const { return wmvdr_[fbinX]; }

  const gsl_matrix_complex *noise_spatial_spectral_matrix(unsigned fbinX) const { return R_[fbinX]; }
  bool set_noise_spatial_spectral_matrix(unsigned fbinX, gsl_matrix_complex* Rnn);
  bool set_diffuse_noise_model(const gsl_matrix* micPositions, float sampleRate, float sspeed = 343740.0); /* micPositions[][x,y,z] */
  void set_all_diagonal_loading(float diagonalWeight);
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  /**
     @brief Divide each non-diagonal elemnt by 1 + mu instead of diagonal loading. mu can be interpreted as the ratio of the sensor noise to the ambient noise power.
     @param float mu[in]
   */
  void divide_all_nondiagonal_elements(float mu){
    for(unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++)
      divide_nondiagonal_elements( fbinX, mu );
  }
  void divide_nondiagonal_elements(unsigned fbinX, float mu);
  gsl_matrix_complex**  noise_spatial_spectral_matrix() const { return R_; }

#ifdef ENABLE_LEGACY_BTK_API
  void clearChannel(){ clear_channel(); }
  bool calcMVDRWeights( float sampleRate, float dThreshold = 1.0E-8, bool calcInverseMatrix = true ){ return calc_mvdr_weights(sampleRate, dThreshold, calcInverseMatrix); }
  const gsl_vector_complex* getMVDRWeights(unsigned fbinX){ return mvdir_weights(fbinX); }
  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix(unsigned fbinX){ return noise_spatial_spectral_matrix(fbinX); }
  bool setNoiseSpatialSpectralMatrix(unsigned fbinX, gsl_matrix_complex* Rnn){ return set_noise_spatial_spectral_matrix(fbinX, Rnn); }
  bool setDiffuseNoiseModel(const gsl_matrix* micPositions, float sampleRate, float sspeed = 343740.0){ return set_diffuse_noise_model(micPositions, sampleRate, sspeed); }
  void setAllLevelsOfDiagonalLoading(float diagonalWeight){ set_all_diagonal_loading(diagonalWeight); }
  void setLevelOfDiagonalLoading(unsigned fbinX, float diagonalWeight){ set_diagonal_looading(fbinX, diagonalWeight); }
  void divideAllNonDiagonalElements( float mu ){ divide_all_nondiagonal_elements(mu); }
  void divideNonDiagonalElements( unsigned fbinX, float mu ){ divide_nondiagonal_elements(fbinX, mu); }
  gsl_matrix_complex**  getNoiseSpatialSpectralMatrix(){ return noise_spatial_spectral_matrix(); }
#endif /* #ifdef ENABLE_LEGACY_BTK_API */

protected:
  gsl_matrix_complex**                           R_; /* Noise spatial spectral matrices */
  gsl_matrix_complex**                           invR_;
  gsl_vector_complex**                           wmvdr_;
  float*                                         diagonal_weights_;
};

// ----- definition for class `SubbandMVDRGSC' -----
//

/**
   @class SubbandMVDRGSC 

   @usage
   1. setChannel()
   2. calc_array_manifold_vectors(), calc_array_manifold_vectors2() or calc_array_manifold_vectorsN().
   3. set_noise_spatial_spectral_matrix() or set_diffuse_noise_model()
   4. calc_mvdr_weights()
   5. calc_blocking_matrix1() or calc_blocking_matrix2()
   6. set_active_weights_f()
 */
class SubbandMVDRGSC : public SubbandMVDR {
 public:
  /**
     @brief MVDR beamforming implementation
     @param int fftLen[in]
     @param bool halfBandShift[in]
   */
  SubbandMVDRGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDRGSC();

  virtual const gsl_vector_complex* next(int frame_no = -5);

  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);
  void zero_active_weights();
  bool calc_blocking_matrix1(float sampleRate, const gsl_vector* delaysT);
  bool calc_blocking_matrix2();
  void upgrade_blocking_matrix();
  const gsl_vector_complex* blocking_matrix_output(int outChanX=0);

#ifdef ENABLE_LEGACY_BTK_API
  void setActiveWeights_f(unsigned fbinX, const gsl_vector* packedWeight){ set_active_weights_f(fbinX, packedWeight); }
  void zeroActiveWeights(){ zero_active_weights(); }
  bool calcBlockingMatrix1(float sampleRate, const gsl_vector* delaysT){ return calc_blocking_matrix1(sampleRate, delaysT); }
  bool calcBlockingMatrix2(){ return calc_blocking_matrix2(); }
  void upgradeBlockingMatrix(){ upgrade_blocking_matrix(); }
  const gsl_vector_complex* blockingMatrixOutput(int outChanX=0){ return blocking_matrix_output(outChanX); }
#endif

protected:
  bool normalize_weight_;
};

typedef Inherit<SubbandBeamformer, VectorComplexFeatureStreamPtr> SubbandBeamformerPtr;
typedef Inherit<SubbandDS, SubbandBeamformerPtr> SubbandDSPtr;
typedef Inherit<SubbandGSC, SubbandDSPtr> SubbandGSCPtr;
typedef Inherit<SubbandGSCRLS, SubbandGSCPtr> SubbandGSCRLSPtr;
typedef Inherit<SubbandMMI, SubbandDSPtr> SubbandMMIPtr;
typedef Inherit<SubbandMVDR, SubbandDSPtr> SubbandMVDRPtr;
typedef Inherit<SubbandMVDRGSC, SubbandMVDRPtr> SubbandMVDRGSCPtr;

// ----- members for class `SubbandOrthogonalizer' -----
//

class SubbandOrthogonalizer : public VectorComplexFeatureStream {
public:
  SubbandOrthogonalizer(SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer");
  ~SubbandOrthogonalizer();
  virtual const gsl_vector_complex* next(int frame_no = -5);

private:
  SubbandMVDRGSCPtr beamformer_;
  int outChanX_;
};

typedef Inherit<SubbandOrthogonalizer, VectorComplexFeatureStreamPtr> SubbandOrthogonalizerPtr;

class SubbandBlockingMatrix : public SubbandGSC {
public:
  SubbandBlockingMatrix(unsigned fftLen=512, bool halfBandShift=false, const String& nm = "SubbandBlockingMatrix")
    :SubbandGSC(fftLen, halfBandShift, nm ){;}

  ~SubbandBlockingMatrix();
  virtual const gsl_vector_complex* next(int frame_no = -5);
};

// ----- definition for class DOAEstimatorSRPBase' -----
//
class DOAEstimatorSRPBase {
public:
  DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax );
  virtual ~DOAEstimatorSRPBase();

  const gsl_vector *nbest_rps() const { return nBestRPs_; }
  const gsl_matrix *nbest_doas() const { return argMaxDOAs_;}
  const gsl_matrix *response_power_matrix() const { return rpMat_;}
  float energy() const {return energy_;}
  void  final_nbest_hypotheses(){get_nbest_hypotheses_from_accrp_();}
  void  set_energy_threshold(float engeryThreshold){ engery_threshold_ = engeryThreshold; }
  void  set_frequency_range(unsigned fbinMin, unsigned fbinMax){ fbinMin_ = fbinMin; fbinMax_ = fbinMax;}
  void  init_accs(){ init_accs_(); }
  void  set_search_param(float minTheta=-M_PI/2, float maxTheta=M_PI/2,
                         float minPhi=-M_PI/2,   float maxPhi=M_PI/2,
                         float widthTheta=0.1,   float widthPhi=0.1);

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_vector *getNBestRPs(){ return nbest_rps(); }
  const gsl_matrix *getNBestDOAs(){ return nbest_doas(); }
  const gsl_matrix *getResponsePowerMatrix(){ return response_power_matrix(); }
  float getEnergy(){return energy();}
  void getFinalNBestHypotheses(){ final_nbest_hypotheses(); }
  void setEnergyThreshold(float engeryThreshold){ set_energy_threshold(engeryThreshold); }
  void setFrequencyRange(unsigned fbinMin, unsigned fbinMax){ set_frequency_range(fbinMin, fbinMax); }
  void initAccs(){ init_accs(); }
  void setSearchParam(float minTheta=-M_PI/2, float maxTheta=M_PI/2,
                      float minPhi=-M_PI/2,   float maxPhi=M_PI/2,
                      float widthTheta=0.1, float widthPhi=0.1)
  {
    set_search_param(minTheta, maxTheta, minPhi, maxPhi, widthTheta, widthPhi);
  }
#endif

protected:
  void clear_table_();
  virtual void get_nbest_hypotheses_from_accrp_();
  virtual void init_accs_();

  float widthTheta_;
  float widthPhi_;
  float minTheta_;
  float maxTheta_;
  float minPhi_;
  float maxPhi_;
  unsigned nTheta_;
  unsigned nPhi_;
  unsigned fbinMin_;
  unsigned fbinMax_;
  unsigned nBest_;
  bool   table_initialized_;

  gsl_vector *accRPs_;
  gsl_vector *nBestRPs_;
  gsl_matrix *argMaxDOAs_;
  vector<gsl_vector_complex **> svTbl_; // [][fftL2+1][_dim]
  gsl_matrix         *rpMat_;

  float engery_threshold_;
  float energy_;

#ifdef  __MBDEBUG__
  void allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */
};

// ----- definition for class DOAEstimatorSRPDSBLA' -----
// 
/**
   @brief estimate the direction of arrival based on the maximum steered response power

   @usage
   1. construct an object
   2. set the geometry of the linear array
   3. call next()
 */
class DOAEstimatorSRPDSBLA :
  public DOAEstimatorSRPBase, public SubbandDS {
public:
  DOAEstimatorSRPDSBLA( unsigned nBest, unsigned sampleRate, unsigned fftLen, const String& nm="DOAEstimatorSRPDSBLA" );
  ~DOAEstimatorSRPDSBLA();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();

  void set_array_geometry(gsl_vector *positions);

#ifdef ENABLE_LEGACY_BTK_API
  void setArrayGeometry(gsl_vector *positions){ set_array_geometry(positions); }
#endif

protected:
  virtual void   calc_steering_unit_table_();
  virtual float calc_response_power_( unsigned uttX );

private:
  virtual void set_look_direction_( int nChan, float theta );

  unsigned    samplerate_;
  gsl_matrix *arraygeometry_; // [micX][x,y,z]
};

typedef refcount_ptr<DOAEstimatorSRPBase> DOAEstimatorSRPBasePtr;
typedef Inherit<DOAEstimatorSRPDSBLA, SubbandDSPtr> DOAEstimatorSRPDSBLAPtr;

// ----- definition for functions' -----
//

float calc_energy(SnapShotArrayPtr snapShotArray, unsigned fbinMin, unsigned fbinMax, unsigned fftLen2, bool  halfBandShift=false);

void calc_gsc_output(const gsl_vector_complex* snapShot,
                     gsl_vector_complex* wl_f, gsl_vector_complex* wq_f,
                     gsl_complex *pYf, bool normalizeWeight=false );

bool pseudoinverse( gsl_matrix_complex *A, gsl_matrix_complex *invA, float dThreshold =  1.0E-8 );

void calc_all_delays(float x, float y, float z, const gsl_matrix* mpos, gsl_vector* delays);

void calc_product(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product);

#ifdef ENABLE_LEGACY_BTK_API
inline void calcAllDelays(float x, float y, float z, const gsl_matrix* mpos, gsl_vector* delays)
{
  calc_all_delays(x, y, z, mpos, delays);
}

inline void calcProduct(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product)
{
  calc_product(synthesisSamples, gs_W, product);
}
#endif

#endif
