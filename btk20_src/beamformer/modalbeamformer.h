/**
 * @file modalbeamformer.h
 * @brief Beamforming in the spherical harmonics domain.
 * @author Kenichi Kumatani
 */
#ifndef MODALBEAMFORMER_H
#define MODALBEAMFORMER_H

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
//#include "stream/pyStream.h"
#include "beamformer/spectralinfoarray.h"
#include "modulated/modulated.h"
#include "beamformer/beamformer.h"


// ----- definition for class `ModeAmplitudeCalculator' -----
//
gsl_complex modeAmplitude(int order, double ka);

class ModeAmplitudeCalculator {
 public:
  ModeAmplitudeCalculator(int order, float minKa=0.01, float maxKa=20, float wid=0.01);

  ~ModeAmplitudeCalculator();

  gsl_vector_complex *get() const { return mode_amplitude_; }

private:
  gsl_vector_complex *mode_amplitude_;
  float minKa_;
  float maxKa_;
  float wid_;
};

typedef refcount_ptr<ModeAmplitudeCalculator> ModeAmplitudeCalculatorPtr;

// ----- definition for class `EigenBeamformer' -----
//

/**
   @class EigenBeamformer
   @brief This beamformer is implemented based on Meyer and Elko's ICASSP paper. 
          In Boaz Rafaely's paper, this method is referred to as the phase-mode beamformer
   @usage 
   1) construct this object, bf = EigenBeamformer(...)
   2) set the radious of the spherical array with bf.set_eigenmike_geometry() or bf.set_array_geometry(..).
   3) set the look direction with bf.set_look_direction(..)
   4) process each block with bf.next() until it hits the end
*/
class EigenBeamformer : public  SubbandDS {
public:
  EigenBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "EigenBeamformer");
  ~EigenBeamformer();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual unsigned dim() const { return dim_;}

  void set_sigma2(float sigma2){ sigma2_ = sigma2; }
  void set_weight_gain(float wgain){ wgain_ = wgain; }
  void set_eigenmike_geometry();
  void set_array_geometry(double a,  gsl_vector *theta_s, gsl_vector *phi_s);
  virtual void set_look_direction(double theta, double phi);

  const gsl_matrix_complex *mode_amplitudes();
  const gsl_vector *array_geometry(int type); // type==0 -> theta, type==1 -> phi
  virtual gsl_matrix *beampattern(unsigned fbinX, double theta = 0, double phi = 0,
                                  double minTheta=-M_PI, double maxTheta=M_PI,
                                  double minPhi=-M_PI, double maxPhi=M_PI,
                                  double widthTheta=0.1, double widthPhi=0.1 );
  /**
     @brief obtain the spherical transformation coefficients at each frame
     @return spherical harmonics transformation coefficients at the current frame
   */
  virtual SnapShotArrayPtr snapshot_array() const { return(st_snapshot_array_); }
  virtual SnapShotArrayPtr snapshot_array2() const { return(snapshot_array_); }

  const gsl_matrix_complex *blocking_matrix(unsigned fbinX, unsigned unitX=0 ) const {
    return (bfweight_vec_[unitX]->B())[fbinX];
  }

#ifdef ENABLE_LEGACY_BTK_API
  void setSigma2(float sigma2){ set_sigma2(sigma2); }
  void setWeightGain(float wgain){ set_weight_gain(wgain); }
  void setEigenMikeGeometry(){ set_eigenmike_geometry(); }
  void setArrayGeometry(double a,  gsl_vector *theta_s, gsl_vector *phi_s){ set_array_geometry(a, theta_s, phi_s); }
  virtual void setLookDirection(double theta, double phi){ set_look_direction(theta, phi); }
  const gsl_matrix_complex *getModeAmplitudes(){ return mode_amplitudes(); }
  const gsl_vector *getArrayGeometry(int type){ return array_geometry(type);}
  virtual gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
                                      double minTheta=-M_PI, double maxTheta=M_PI,
                                      double minPhi=-M_PI, double maxPhi=M_PI,
                                      double widthTheta=0.1, double widthPhi=0.1 ){
    return beampattern(fbinX, theta, phi, minTheta, maxTheta, minPhi, maxPhi, widthTheta, widthPhi);
  }
  virtual SnapShotArrayPtr getSnapShotArray(){ return snapshot_array(); }
  virtual SnapShotArrayPtr getSnapShotArray2(){ return snapshot_array2(); }
  const gsl_matrix_complex *getBlockingMatrix(unsigned fbinX, unsigned unitX=0){
    return blocking_matrix(fbinX, unitX);
  }
#endif

 protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool calc_spherical_harmonics_at_each_position_( gsl_vector *theta_s, gsl_vector *phi_s ); // need to be tested!!
  virtual bool calc_steering_unit_(  int unitX=0, bool isGSC=false );
  virtual bool alloc_steering_unit_( int unitN=1 );
  void alloc_image_( bool flag=true );
  bool calc_mode_amplitudes_();

  unsigned samplerate_;
  unsigned NC_;
  unsigned maxOrder_;
  unsigned dim_; // the number of the spherical harmonics transformation coefficients
  bool weights_normalized_;
  gsl_matrix_complex *mode_mplitudes_; // [maxOrder_] the mode amplitudes.
  gsl_vector_complex *F_; // Spherical Transform coefficients [dim_]
  gsl_vector_complex **sh_s_; // Conjugate of spherical harmonics at each sensor position [dim_][nChan]: Y_n^{m*}
  SnapShotArrayPtr st_snapshot_array_; // for compatibility with a post-filtering object

  double theta_; // look direction
  double phi_;   // look direction

  double a_;     // the radius of the rigid sphere.
  gsl_vector *theta_s_; // sensor positions
  gsl_vector *phi_s_;   // sensor positions
  gsl_matrix *beampattern_;
  gsl_vector *WNG_; // white noise gain
  float wgain_; //
  float sigma2_; // dialog loading
};

typedef Inherit<EigenBeamformer,  SubbandDSPtr> EigenBeamformerPtr;

// ----- definition for class DOAEstimatorSRPEB' -----
// 

/**
   @class DOAEstimatorSRPEB
   @brief estimate the direction of arrival based on the maximum steered response power
   @usage 
   1) construct this object, doaEstimator = DOAEstimatorSRPEB(...)
   2) set the radious of the spherical array, doaEstimator.set_eigenmike_geometry() or doaEstimator.set_array_geometry(..).
   3) process each block, doaEstimator.next() 
   4) get the N-best hypotheses at the current instantaneous frame through doaEstimator.nbest_doas()
   5) do doaEstimator.getFinalNBestHypotheses() after a static segment is processed.
      You can then obtain the averaged N-best hypotheses of the static segment with doaEstimator.nbest_doas().
*/
class DOAEstimatorSRPEB :
  public DOAEstimatorSRPBase, public EigenBeamformer {
  public:
  DOAEstimatorSRPEB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPBase");
  ~DOAEstimatorSRPEB();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();

protected:
  virtual void  calc_steering_unit_table_();
  virtual float calc_response_power_( unsigned uttX );
};

typedef Inherit<DOAEstimatorSRPEB, EigenBeamformerPtr> DOAEstimatorSRPEBPtr;

// ----- definition for class `SphericalDSBeamformer' -----
// 

/**
   @class SphericalDSBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalDSBeamformer : public EigenBeamformer {
public:
  SphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~SphericalDSBeamformer();
  virtual gsl_vector *calc_wng();

#ifdef ENABLE_LEGACY_BTK_API
  virtual gsl_vector *calcWNG(){ return calc_wng(); }
#endif

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool calc_spherical_harmonics_at_each_position_( gsl_vector *theta_s, gsl_vector *phi_s );
};

typedef Inherit<SphericalDSBeamformer, EigenBeamformerPtr> SphericalDSBeamformerPtr;

// ----- definition for class `DualSphericalDSBeamformer' -----
// 

/**
   @class DualSphericalDSBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note In addition to  SphericalDSBeamformer, this class has an object of the *normal* D&S beamformer
*/
class DualSphericalDSBeamformer : public SphericalDSBeamformer {
public:
  DualSphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~DualSphericalDSBeamformer();

  virtual SnapShotArrayPtr snapshot_array() const { return snapshot_array_; }
  virtual BeamformerWeights* beamformer_weight_object(unsigned srcX=0) const {
    return bfweight_vec2_[srcX];
  }

#ifdef ENABLE_LEGACY_BTK_API
  virtual SnapShotArrayPtr getSnapShotArray(){ return snapshot_array(); }
#endif

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool alloc_steering_unit_( int unitN=1 );

  vector<BeamformerWeights *>                   bfweight_vec2_; // weights of a normal D&S beamformer.
};

typedef Inherit<DualSphericalDSBeamformer, SphericalDSBeamformerPtr> DualSphericalDSBeamformerPtr;

// ----- definition for class DOAEstimatorSRPPSphDSB' -----
// 

class DOAEstimatorSRPSphDSB : public DOAEstimatorSRPBase, public SphericalDSBeamformer {
public:
  DOAEstimatorSRPSphDSB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DOAEstimatorSRPPSphDSB" );
  ~DOAEstimatorSRPSphDSB();
  const gsl_vector_complex* next(int frame_no = -5);
  void reset();

protected:
  virtual void  calc_steering_unit_table_();
  virtual float calc_response_power_( unsigned uttX );
};

typedef Inherit<DOAEstimatorSRPSphDSB, SphericalDSBeamformerPtr> DOAEstimatorSRPSphDSBPtr;

// ----- definition for class `SphericalHWNCBeamformer' -----
// 

/**
   @class SphericalHWNCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
*/
class SphericalHWNCBeamformer : public EigenBeamformer {
public:
  SphericalHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalHWNCBeamformer();

  virtual gsl_vector *calc_wng();
  virtual void set_wng( double ratio){ ratio_=ratio; calc_wng();}

#ifdef ENABLE_LEGACY_BTK_API
  gsl_vector *calcWNG(){ return calc_wng(); }
  void setWNG( double ratio){ set_wng(ratio);}
#endif

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );

protected:
  float ratio_;
};

typedef Inherit<SphericalHWNCBeamformer, EigenBeamformerPtr> SphericalHWNCBeamformerPtr;

// ----- definition for class `SphericalGSCBeamformer' -----
// 

/**
   @class SphericalGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalGSCBeamformer : public SphericalDSBeamformer {
public:
  SphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer");
  ~SphericalGSCBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void set_look_direction(double theta, double phi);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);

#ifdef ENABLE_LEGACY_BTK_API
  void setLookDirection(double theta, double phi){ set_look_direction(theta, phi); }
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight ){ set_active_weights_f(fbinX, packedWeight); }
#endif
};

typedef Inherit<SphericalGSCBeamformer, SphericalDSBeamformerPtr> SphericalGSCBeamformerPtr;

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 

/**
   @class SphericalHWNCGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalHWNCGSCBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
*/
class SphericalHWNCGSCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalHWNCGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer");
  ~SphericalHWNCGSCBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void set_look_direction(double theta, double phi);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);

#ifdef ENABLE_LEGACY_BTK_API
  void setLookDirection(double theta, double phi){ set_look_direction(theta, phi); }
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight ){ set_active_weights_f(fbinX, packedWeight); }
#endif
};

typedef Inherit<SphericalHWNCGSCBeamformer, SphericalHWNCBeamformerPtr> SphericalHWNCGSCBeamformerPtr;

// ----- definition for class `DualSphericalGSCBeamformer' -----
// 

/**
   @class DualSphericalGSCBeamformer
   @usage 
   1) construct this object, mb = DualSphericalGSCBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note In addition to DualSphericalGSCBeamformer, this class has an object of the *normal* D&S beamformer
*/
class DualSphericalGSCBeamformer : public SphericalGSCBeamformer {
public:
  DualSphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer");
  ~DualSphericalGSCBeamformer();

  virtual SnapShotArrayPtr snapshot_array() const {return(snapshot_array_);}
  virtual BeamformerWeights* beamformer_weight_object(unsigned srcX=0) const {
    return bfweight_vec2_[srcX];
  }

#ifdef ENABLE_LEGACY_BTK_API
  virtual SnapShotArrayPtr getSnapShotArray(){return(snapshot_array_);}
#endif

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool alloc_steering_unit_( int unitN=1 );

  vector<BeamformerWeights *>                   bfweight_vec2_; // weights of a normal D&S beamformer.
};

typedef Inherit<DualSphericalGSCBeamformer, SphericalGSCBeamformerPtr> DualSphericalGSCBeamformerPtr;

// ----- definition for class `SphericalMOENBeamformer' -----
// 

/**
   @class SphericalGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note this implementation is based on Z. Li and R. Duraiswami's letter,
   "Flexible and Optimal Design of Spherical Microphone Arrays for Beamforming", IEEE Trans. SAP.
*/
class SphericalMOENBeamformer : public SphericalDSBeamformer {
public:
  SphericalMOENBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer");
  ~SphericalMOENBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void fix_terms(bool flag){ is_term_fixed_ = flag; }
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  virtual SnapShotArrayPtr snapshot_array() const { return snapshot_array_; }
  virtual gsl_matrix *beampattern(unsigned fbinX, double theta = 0, double phi = 0,
                                  double minTheta=-M_PI, double maxTheta=M_PI,
                                  double minPhi=-M_PI, double maxPhi=M_PI,
                                  double widthTheta=0.1, double widthPhi=0.1 );

#ifdef ENABLE_LEGACY_BTK_API
  void fixTerms( bool flag ){ fix_terms(flag); }
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight){ set_diagonal_looading(fbinX, diagonalWeight); }
  virtual gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
                                      double minTheta=-M_PI, double maxTheta=M_PI,
                                      double minPhi=-M_PI, double maxPhi=M_PI,
                                      double widthTheta=0.1, double widthPhi=0.1 ){
    return beampattern(fbinX, theta, phi, minTheta, maxTheta, minPhi, maxPhi, widthTheta, widthPhi);
  }
#endif

protected:
  virtual bool alloc_steering_unit_( int unitN=1 );
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  bool calc_moen_weights_( unsigned fbinX, gsl_vector_complex *weights, double dThreshold = 1.0E-8, bool calcInverseMatrix = true, unsigned unitX=0 );

private:
  // maxOrder_ == Neff in the Li's paper.
  unsigned             bf_order_; // N in the Li's paper.
  float                CN_;
  gsl_matrix_complex** A_; /* A_[fftLen2+1][dim_][nChan]; Coeffcients of the spherical harmonics expansion; See Eq. (31) & (32) */
  gsl_matrix_complex** fixedW_; /* _fixedW[fftLen2+1][nChan][dim_]; [ A^H A + l^2 I ]^{-1} A^H */
  gsl_vector_complex** BN_;     // _BN[fftLen2+1][dim_]
  float*               diagonal_weights_;
  bool                 is_term_fixed_;
  float                dthreshold_;
};

typedef Inherit<SphericalMOENBeamformer, SphericalDSBeamformerPtr> SphericalMOENBeamformerPtr;


// ----- definition for class `SphericalSpatialDSBeamformer' -----
// 

/**
   @class SphericalSpatialDSBeamformer
   @usage 
   1) construct this object, mb = SphericalSpatialDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalSpatialDSBeamformer : public SphericalDSBeamformer {
public:
  SphericalSpatialDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalSpatialDSBeamformer");
  ~SphericalSpatialDSBeamformer();
  virtual const gsl_vector_complex* next(int frame_no = -5);

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool alloc_steering_unit_( int unitN = 1 );
  virtual bool calc_steering_unit_( int unitX = 0, bool isGSC = false );
};

typedef Inherit<SphericalSpatialDSBeamformer, SphericalDSBeamformerPtr> SphericalSpatialDSBeamformerPtr;


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
// 

/**
   @class SphericalSpatialHWNCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.set_array_geometry(..) or 
   3) set the look direction mb.set_look_direction()
   4) process each block mb.next()
*/
class SphericalSpatialHWNCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalSpatialHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalSpatialHWNCBeamformer();
  virtual const gsl_vector_complex* next(int frame_no = -5);

protected:
  virtual void calc_weights_( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool alloc_steering_unit_( int unitN = 1 );
  virtual bool calc_steering_unit_( int unitX = 0, bool isGSC = false );

private:
  gsl_matrix_complex *calc_diffuse_noise_model_( unsigned fbinX );

  gsl_matrix_complex **SigmaSI_; // SigmaSI_[fftLen/2+1][chanN]
  double  dthreshold_;
};

typedef Inherit<SphericalSpatialHWNCBeamformer, SphericalHWNCBeamformerPtr> SphericalSpatialHWNCBeamformerPtr;

#endif
