/**
 * @file tracker.h
 * @brief Beamforming and speaker tracking with a spherical microphone array.
 * @author John McDonough
 */
#ifndef TRACKER_H
#define TRACKER_H

#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_complex_math.h>

#include "common/refcount.h"
#include "common/jexception.h"

#include "stream/stream.h"
#include "beamformer/spectralinfoarray.h"
#include "modulated/modulated.h"
#include "aec/aec.h"

// ----- definition of class 'BaseDecomposition' -----
//
class BaseDecomposition : public Countable {
public:
  class SubbandEntry {
  public:
    SubbandEntry(unsigned subbandX, const gsl_complex& Bkl);

    unsigned subbandX() const { return subbandX_; }
    gsl_complex bkl()   const { return bkl_; }

    void* operator new(size_t sz) { return memoryManager().new_elem(); }
    void  operator delete(void* e) { memoryManager().delete_elem(e); }

    static MemoryManager<SubbandEntry>& memoryManager();

  private:
    const unsigned			subbandX_;
    const gsl_complex			bkl_;
  };

  class GreaterThan {
  public:
    bool operator()(SubbandEntry* sbX1, SubbandEntry* sbX2) {
      return (gsl_complex_abs(sbX1->bkl()) > gsl_complex_abs(sbX2->bkl()));
    }
  };

  class Iterator;

  class SubbandList : public Countable {
  public:
    SubbandList(const gsl_vector_complex* bkl, unsigned useSubbandsN = 0);
    ~SubbandList();

    unsigned useSubbandsN() const { return useSubbandsN_; }
    SubbandEntry** subbands() { return subbands_; }

    friend class Iterator;

  private:
    const unsigned			subbandsN_;
    const unsigned			useSubbandsN_;
    SubbandEntry**			subbands_;
  };

  typedef refcountable_ptr<SubbandList> SubbandListPtr;

  class Iterator {
  public:
    Iterator(const SubbandListPtr& subbandList)
      : subbandX_(0), useSubbandsN_(subbandList->useSubbandsN()), subbands_(subbandList->subbands()) { }
    void operator++(int) { subbandX_++; }
    bool more()          { return subbandX_ < useSubbandsN_; }
    const SubbandEntry& operator*() { return *(subbands_[subbandX_]); }

  private:
    unsigned				subbandX_;
    const unsigned			useSubbandsN_;
    SubbandEntry**			subbands_;
  };

 public:
  BaseDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0, bool spatial = false);
  ~BaseDecomposition();

  unsigned orderN()         const { return orderN_;         }
  unsigned modesN()         const { return modesN_;         }
  unsigned subbandsN2()     const { return subbandsN2_;     }
  unsigned subbandsN()      const { return subbandsN_;      }
  unsigned useSubbandsN()   const { return useSubbandsN_;   }
  unsigned subbandLengthN() const { return subbandLengthN_; }

  SubbandListPtr subbandList()  { return subbandList_;    }

  virtual void reset();

  static gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonic(int order, int degree, unsigned channelX) const;
  static gsl_complex harmonic_deriv_polar_angle(int order, int degree, double theta, double phi);
  static gsl_complex harmonic_deriv_azimuth(int order, int degree, double theta, double phi);
  static gsl_complex modal_coefficient(unsigned order, double ka);
  gsl_complex modal_coefficient(unsigned order, unsigned subbandX) const;
  virtual void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed) { }
  virtual void estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX) = 0;
  virtual void calculate_gkl(double theta, double phi, unsigned subbandX) = 0;
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frame_no) = 0;
  virtual const gsl_vector_complex* predicted_observation(gsl_vector* xk, int frame_no) = 0;

#ifdef ENABLE_LEGACY_BTK_API
  static gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi){
    return harmonic_deriv_polar_angle(order, degree, theta, phi);
  }
  static gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi){
    return harmonic_deriv_azimuth(order, degree, theta, phi);
  }
  static gsl_complex modalCoefficient(unsigned order, double ka){
    return modal_coefficient(order, ka);
  }
#endif

  static const unsigned				StateN_;
  static const unsigned				ChannelsN_;
  static const gsl_complex 			ComplexZero_;
  static const gsl_complex 			ComplexOne_;
  static const double				SpeedOfSound_;

 protected:
  static gsl_complex calc_in_(int n);
  void set_eigenmike_geometry_();
  gsl_complex calc_Gnm_(unsigned subbandX, int n, int m, double theta, double phi);
  gsl_complex calc_dGnm_dtheta_(unsigned subbandX, int n, int m, double theta, double phi);
  static double calc_Pnm_(int order, int degree, double theta);
  static double calc_dPnm_dtheta_(int n, int m, double theta);
  static double calc_normalization_(int order, int degree);

  const unsigned				orderN_;
  const unsigned				modesN_;
  const unsigned				subbandsN_;
  const unsigned				subbandsN2_;
  const unsigned				useSubbandsN_;
  const unsigned				subbandLengthN_;
  const double					samplerate_;
  const double					a_;
  gsl_vector_complex**				bn_;
  gsl_vector*					theta_s_;
  gsl_vector*					phi_s_;
  gsl_vector_complex**				spherical_component_;
  gsl_vector_complex*				bkl_;
  gsl_vector_complex*				dbkl_dtheta_;
  gsl_vector_complex*				dbkl_dphi_;
  gsl_vector_complex**				gkl_;
  gsl_vector_complex**				dgkl_dtheta_;
  gsl_vector_complex**				dgkl_dphi_;
  gsl_vector_complex*				vkl_;
  gsl_matrix_complex*				Hbar_k_;
  gsl_vector_complex*				yhat_k_;

  SubbandListPtr				subbandList_;
};

typedef refcountable_ptr<BaseDecomposition> BaseDecompositionPtr;


// ----- definition of class 'ModalDecomposition' -----
//
class ModalDecomposition : public BaseDecomposition {
public:
  ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~ModalDecomposition() { }

  virtual void estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  virtual void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frame_no);
  virtual const gsl_vector_complex* predicted_observation(gsl_vector* xk, int frame_no);
  virtual void calculate_gkl(double theta, double phi, unsigned subbandX);
};

typedef Inherit<ModalDecomposition, BaseDecompositionPtr>	ModalDecompositionPtr;


// ----- definition of class 'SpatialDecomposition' -----
//
class SpatialDecomposition : public BaseDecomposition {
public:
  SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~SpatialDecomposition() { }

  virtual void estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frame_no);
  virtual const gsl_vector_complex* predicted_observation(gsl_vector* xk, int frame_no);
  virtual void calculate_gkl(double theta, double phi, unsigned subbandX);
};

typedef Inherit<SpatialDecomposition, BaseDecompositionPtr>	SpatialDecompositionPtr;


// ----- definition of class 'BaseSphericalArrayTracker' -----
//
class BaseSphericalArrayTracker : public VectorFloatFeatureStream {
  typedef list<VectorComplexFeatureStreamPtr>	ChannelList_;
  typedef ChannelList_::iterator		ChannelIterator_;
  typedef BaseDecomposition::SubbandEntry	SubbandEntry;
  typedef BaseDecomposition::SubbandListPtr	SubbandListPtr;
  typedef BaseDecomposition::Iterator		Iterator;

public:
  BaseSphericalArrayTracker(BaseDecompositionPtr& baseDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			    unsigned maxLocalN = 1, const String& nm = "BaseSphericalArrayTracker");
  ~BaseSphericalArrayTracker();

  virtual const gsl_vector_float* next(int frame_no = -5) = 0;

  virtual void reset();

  unsigned chanN() const { return channelList_.size(); }
  void set_channel(VectorComplexFeatureStreamPtr& chan);
  void set_V(const gsl_matrix_complex* Vk, unsigned subbandX);
  void next_speaker();
  void set_initial_position(double theta, double phi);

#ifdef ENABLE_LEGACY_BTK_API
  void setChannel(VectorComplexFeatureStreamPtr& chan){ set_channel(chan); }
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX){ set_V(Vk, subbandX); }
  void nextSpeaker(){ next_speaker(); }
  void setInitialPosition(double theta, double phi){ set_initial_position(theta, phi); }
#endif

protected:
  static void printMatrix_(const gsl_matrix_complex* mat);
  static void printMatrix_(const gsl_matrix* mat);
  static void printVector_(const gsl_vector_complex* vec);
  static void printVector_(const gsl_vector* vec);

  static double calc_givens_rotation_(double v1, double v2, double& c, double& s);
  static void apply_givens_rotation_(double v1, double v2, double c, double s, double& v1p, double& v2p);

  void alloc_image_();
  void update_(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k, const SubbandListPtr& subbandList);
  void lower_triangularize_();
  void copy_position_();
  void check_physical_constraints_();
  double calc_residual_();
  void realify_(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k);
  void realify_residual_();

  static const unsigned 			StateN_;
  static const gsl_complex 			ComplexZero_;
  static const gsl_complex 			ComplexOne_;
  static const double      			Epsilon_;
  static const double				Tolerance_;

  bool						firstFrame_;
  const unsigned				subbandsN_;
  const unsigned				subbandsN2_;
  const unsigned				useSubbandsN_;
  const unsigned 				modesN_;
  const unsigned 				subbandLengthN_;
  const unsigned 				observationN_;
  const unsigned 				maxLocalN_;
  bool						is_end_;
  const double 					sigma_init_;

  SnapShotArrayPtr				snapshot_array_;
  BaseDecompositionPtr				base_decomposition_;
  ChannelList_					channelList_;

  // these quantities are stored as Cholesky factors
  gsl_matrix*					U_;
  gsl_matrix*					V_;
  gsl_matrix*					K_k_k1_;

  // work space for state estimate update
  gsl_matrix*					prearray_;
  gsl_vector_complex*				vk_;
  gsl_matrix*					Hbar_k_;
  gsl_vector*					yhat_k_;
  gsl_vector*					correction_;
  gsl_vector*					position_;
  gsl_vector*					eta_i_;
  gsl_vector*					delta_;
  gsl_vector_complex*				residual_;
  gsl_vector*					residual_real_;
  gsl_vector*					scratch_;
};

typedef Inherit<BaseSphericalArrayTracker, VectorFloatFeatureStreamPtr> BaseSphericalArrayTrackerPtr;


// ----- definition of class 'ModalSphericalArrayTracker' -----
//
class ModalSphericalArrayTracker : public BaseSphericalArrayTracker {
  typedef list<VectorComplexFeatureStreamPtr>	ChannelList_;
  typedef ChannelList_::iterator		ChannelIterator_;

public:
  ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			     unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker");
  ~ModalSphericalArrayTracker() { }

  const gsl_vector_float* next(int frame_no = -5);
};

typedef Inherit<ModalSphericalArrayTracker, VectorFloatFeatureStreamPtr> ModalSphericalArrayTrackerPtr;


// ----- definition of class 'SpatialSphericalArrayTracker' -----
//
class SpatialSphericalArrayTracker : public BaseSphericalArrayTracker {
  typedef list<VectorComplexFeatureStreamPtr>	ChannelList_;
  typedef ChannelList_::iterator		ChannelIterator_;

public:
 SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			       unsigned maxLocalN = 1, const String& nm = "SpatialSphericalArrayTracker");
  ~SpatialSphericalArrayTracker() { }

  const gsl_vector_float* next(int frame_no= -5);
};

typedef Inherit<SpatialSphericalArrayTracker, VectorFloatFeatureStreamPtr> SpatialSphericalArrayTrackerPtr;


// ----- definition of class 'PlaneWaveSimulator' -----
//
class PlaneWaveSimulator : public VectorComplexFeatureStream {
public:
  PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
		     unsigned channelX, double theta, double phi, const String& nm = "Plane Wave Simulator");
  ~PlaneWaveSimulator();

  const gsl_vector_complex* next(int frame_no = -5);

  virtual void reset();

private:
  static const gsl_complex			ComplexZero_;

  const unsigned				subbandsN_;
  const unsigned				subbandsN2_;
  const unsigned				channelX_;
  const double					theta_;
  const double					phi_;

  VectorComplexFeatureStreamPtr			source_;
  ModalDecompositionPtr				modalDecomposition_;
  gsl_vector_complex*				subbandCoefficients_;
};

typedef Inherit<PlaneWaveSimulator, VectorComplexFeatureStreamPtr> PlaneWaveSimulatorPtr;

#ifdef HAVE_GSL_V11X

int gsl_matrix_complex_add( gsl_matrix_complex *am, gsl_matrix_complex *bm )
{
  if( am->size1 != bm->size1 ){
    fprintf(stderr,"Dimension error \n",am->size1, bm->size1 );
    return 0;
  }

  if( am->size2 != bm->size2 ){
    fprintf(stderr,"Dimension error \n",am->size2, bm->size2 );
    return 0;
  }

  for(size_t i=0;i<am->size1;i++){
    for(size_t j=0;j<am->size2;j++){
      gsl_complex val = gsl_complex_add( gsl_matrix_complex_get(am,i,j), gsl_matrix_complex_get(bm,i,j) );
      gsl_matrix_complex_set(am,i,j,val);
    }
  }

  return 1;
}

int gsl_vector_complex_add( gsl_vector_complex *av, gsl_vector_complex *bv )
{
  if( av->size != bv->size ){
    fprintf(stderr,"Dimension error \n",av->size, bv->size);
    return 0;
  }
  
  for(size_t i=0;i<av->size;i++){
    gsl_complex val = gsl_complex_add( gsl_vector_complex_get(av,i), gsl_vector_complex_get(bv,i) );
    gsl_vector_complex_set(av,i,val);
  }

  return 1;
}


int gsl_vector_complex_sub( const gsl_vector_complex *av, const gsl_vector_complex *bv )
{
  if( av->size != bv->size ){
    fprintf(stderr,"Dimension error \n",av->size, bv->size);
    return 0;
  }

  for(size_t i=0;i<av->size;i++){
    gsl_complex val = gsl_complex_sub( gsl_vector_complex_get(av,i), gsl_vector_complex_get(bv,i) );
    gsl_vector_complex_set((gsl_vector_complex*)av,i,val);
  }

  return 1;
}

#endif /* End of HAVE_GSL_V11X */

#endif
