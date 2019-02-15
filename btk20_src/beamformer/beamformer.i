/**
 * @file beamformer.i
 * @brief Beamforming in the subband domain.
 * @author John McDonough and Kenichi Kumatani
 */

%module(package="btk20") beamformer

%{
#include "beamformer/beamformer.h"
#include "beamformer/taylorseries.h"
#include "beamformer/modalbeamformer.h"
#include "beamformer/tracker.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
#include "postfilter/postfilter.h"
#include "postfilter/spectralsubtraction.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

%include gsl/gsl_types.h
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%pythoncode %{
import btk20
from btk20 import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

%import postfilter/postfilter.i

// ----- definition for class `SnapShotArray' -----
// 
%ignore SnapShotArray;
class SnapShotArray {
  %feature("kwargs") snapshot;
  %feature("kwargs") set_samples;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") getSnapShot;
  %feature("kwargs") newSample;
#endif
 public:
  SnapShotArray(unsigned fftLn, unsigned nChn);
  virtual ~SnapShotArray();

  const gsl_vector_complex* snapshot(unsigned fbinX) const;
  void set_samples(const gsl_vector_complex* samp, unsigned chanX);

  unsigned fftLen() const;
  unsigned nChan()  const;
  virtual void update();
  virtual void zero();

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_vector_complex* getSnapShot(unsigned fbinX);
  void newSample(const gsl_vector_complex* samp, unsigned chanX);
#endif
};

class SnapShotArrayPtr {
  %feature("kwargs") SnapShotArrayPtr;
 public:
  %extend {
    SnapShotArrayPtr(unsigned fftlen, unsigned chan_num) {
      return new SnapShotArrayPtr(new SnapShotArray(fftlen, chan_num));
    }
  }

  SnapShotArray* operator->();
};

// ----- definition for class `SpectralMatrixArray' -----
//
%ignore SpectralMatrixArray;
class SpectralMatrixArray : public SnapShotArray {
  %feature("kwargs") matrix_f;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") getSpecMatrix;
#endif
public:
  SpectralMatrixArray(unsigned fftLn, unsigned nChn, float forgetFact = 0.95);
  virtual ~SpectralMatrixArray();

  gsl_matrix_complex* matrix_f(unsigned idx) const;
  virtual void update();
  virtual void zero();

#ifdef ENABLE_LEGACY_BTK_API
  gsl_matrix_complex* getSpecMatrix(unsigned idx);
#endif
};

class SpectralMatrixArrayPtr {
  %feature("kwargs") SpectralMatrixArrayPtr;
 public:
  %extend {
    SpectralMatrixArrayPtr(unsigned fftLn, unsigned nChn, float forgetFact = 0.95) {
      return new SpectralMatrixArrayPtr(new SpectralMatrixArray(fftLn, nChn, forgetFact));
    }
  }

  SpectralMatrixArray* operator->();
};

// ----- definition for class `SubbandBeamformer' -----
// 
%ignore SubbandBeamformer;
class SubbandBeamformer : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_channel;
  %feature("kwargs") snapshot_array_f;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") snapShotArray_f;
  %feature("kwargs") setChannel;
#endif
 public:
  SubbandBeamformer(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandBeamformer");
  ~SubbandBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  const gsl_vector_complex* snapshot_array_f(unsigned fbinX) const;
  SnapShotArrayPtr snapshot_array();
  void set_channel(VectorComplexFeatureStreamPtr& chan);
  virtual void clear_channel();
  bool is_end();
  virtual unsigned dim();
  unsigned fftLen();
  unsigned chanN();

#ifdef ENABLE_LEGACY_BTK_API
  bool isEnd();
  const gsl_vector_complex* snapShotArray_f(unsigned fbinX);
  SnapShotArrayPtr getSnapShotArray();
  void setChannel(VectorComplexFeatureStreamPtr& chan);
  virtual void clearChannel();
#endif
};

class SubbandBeamformerPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") SubbandBeamformerPtr;
 public:
  %extend {
    SubbandBeamformerPtr(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandBeamformer") {
       return new SubbandBeamformerPtr(new SubbandBeamformer(fftlen, half_band_shift, nm));
    }

    SubbandBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandBeamformer* operator->();
};

// ----- definition for class `SubbandDS' -----
//
%ignore SubbandDS;
class SubbandDS : public SubbandBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") get_weights;
  %feature("kwargs") calc_array_manifold_vectors;
  %feature("kwargs") calc_array_manifold_vectors_2;
  %feature("kwargs") calc_array_manifold_vectors_n;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") getWeights;
  %feature("kwargs") getBeamformerWeightObject;
  %feature("kwargs") calcArrayManifoldVectors;
  %feature("kwargs") calcArrayManifoldVectors2;
  %feature("kwargs") calcArrayManifoldVectorsN;
#endif
public:
  SubbandDS(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandDS");
  ~SubbandDS();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  virtual void clear_channel();
  virtual const gsl_vector_complex *get_weights(unsigned fbinX);

  virtual void calc_array_manifold_vectors(float samplerate, const gsl_vector* delays);
  virtual void calc_array_manifold_vectors_2(float samplerate, const gsl_vector* delays_t, const gsl_vector* delays_j);
  virtual void calc_array_manifold_vectors_n(float samplerate, const gsl_vector* delays_t, const gsl_matrix* delays_j, unsigned NC=2);

#ifdef ENABLE_LEGACY_BTK_API
  virtual void clearChannel();
  virtual const gsl_vector_complex *getWeights(unsigned fbinX);
  virtual BeamformerWeights* getBeamformerWeightObject(unsigned srcX=0) const { return beamformer_weight_object(srcX); }
  virtual void calcArrayManifoldVectors(float samplerate, const gsl_vector* delays);
  virtual void calcArrayManifoldVectors2(float samplerate, const gsl_vector* delaysT, const gsl_vector* delaysJ );
  virtual void calcArrayManifoldVectorsN(float samplerate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, int NC );
#endif
};

class SubbandDSPtr : public SubbandBeamformerPtr {
  %feature("kwargs") SubbandDSPtr;
 public:
  %extend {
    SubbandDSPtr(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandDS") {
      return new SubbandDSPtr(new SubbandDS(fftlen, half_band_shift, nm));
    }

    SubbandDSPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandDS* operator->();
};

// ----- definition for class `SubbandGSC' -----
//
%ignore SubbandGSC;
class SubbandGSC : public SubbandDS {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") normalize_weight;
  %feature("kwargs") set_quiescent_weights_f;
  %feature("kwargs") calc_gsc_weights;
  %feature("kwargs") calc_gsc_weights_2;
  %feature("kwargs") calc_gsc_weights_n;
  %feature("kwargs") write_fir_coeff;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") normalizeWeight;
  %feature("kwargs") calcGSCWeights;
  %feature("kwargs") calcGSCWeights2;
  %feature("kwargs") calcGSCWeightsN;
  %feature("kwargs") setActiveWeights_f;
  %feature("kwargs") writeFIRCoeff;
  %feature("kwargs") getBlockingMatrix;
#endif
 public:
 SubbandGSC(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandGSC");
  ~SubbandGSC();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void normalize_weight(bool flag);
  void set_quiescent_weights_f(unsigned fbinX, const gsl_vector_complex * srcWq);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);
  void zero_active_weights();
  void calc_gsc_weights(float samplerate, const gsl_vector* delaysT);
  void calc_gsc_weights_2(float samplerate, const gsl_vector* delaysT, const gsl_vector* delaysJ);
  void calc_gsc_weights_n(float samplerate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2);
  bool write_fir_coeff( const String& fn, unsigned winType=1 );

#ifdef ENABLE_LEGACY_BTK_API
  void normalizeWeight(bool flag );
  void calcGSCWeights(float sampleRate, const gsl_vector* delaysT);
  void calcGSCWeights2(float sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ);
  void calcGSCWeightsN(float sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, int NC=2);
  void setActiveWeights_f(int fbinX, const gsl_vector* packedWeight);
  void zeroActiveWeights();
  bool writeFIRCoeff(const String& fn, unsigned winType=1);
  gsl_matrix_complex* getBlockingMatrix(unsigned srcX, unsigned fbinX);
#endif
};

class SubbandGSCPtr : public SubbandDSPtr {
  %feature("kwargs") SubbandGSCPtr;
 public:
  %extend {
    SubbandGSCPtr(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandGSC") {
      return new SubbandGSCPtr(new SubbandGSC(fftlen, half_band_shift, nm));
    }

    SubbandGSCPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandGSC* operator->();
};

// ----- definition for class `SubbandGSCRLS' -----
//
%ignore SubbandGSCRLS;
class SubbandGSCRLS : public SubbandGSC {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") init_precision_matrix;
  %feature("kwargs") set_precision_matrix;
  %feature("kwargs") update_active_weight_vecotrs;
  %feature("kwargs") set_quadratic_constraint;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") initPrecisionMatrix;
  %feature("kwargs") updateActiveWeightVecotrs;
  %feature("kwargs") setPrecisionMatrix;
  %feature("kwargs") setQuadraticConstraint;
#endif
 public:
 SubbandGSCRLS(unsigned fftlen, bool half_band_shift = false, float myu = 0.9, float sigma2=0.0, const String& nm = "SubbandGSCRLS");
  ~SubbandGSCRLS();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void init_precision_matrix(float sigma2 = 0.01);
  void set_precision_matrix(unsigned fbinX, gsl_matrix_complex *Pz);
  void update_active_weight_vecotrs(bool flag);
  void set_quadratic_constraint(float alpha, int qctype=1);

#ifdef ENABLE_LEGACY_BTK_API
  void initPrecisionMatrix(float sigma2 = 0.01);
  void updateActiveWeightVecotrs(bool flag);
  void setPrecisionMatrix(unsigned fbinX, gsl_matrix_complex *Pz);
  void setQuadraticConstraint(float alpha, int qctype=1);
#endif
};

class SubbandGSCRLSPtr : public SubbandGSCPtr {
  %feature("kwargs") SubbandGSCRLSPtr;
 public:
  %extend {
    SubbandGSCRLSPtr( unsigned fftlen, bool half_band_shift = false, float myu = 0.9, float sigma2=0.01, const String& nm = "SubbandGSCRLS") {
      return new SubbandGSCRLSPtr(new SubbandGSCRLS(fftlen, half_band_shift, myu, sigma2, nm));
    }

    SubbandGSCRLSPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandGSCRLS* operator->();
};

// ----- definition for class `SubbandMMI' -----
//
%ignore SubbandMMI;
class SubbandMMI : public SubbandDS {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") use_binary_mask;
  %feature("kwargs") calc_weights;
  %feature("kwargs") calc_weights_n;
  %feature("kwargs") set_hi_active_weights_f;
  %feature("kwargs") set_active_weights_f;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") useBinaryMask;
  %feature("kwargs") calcWeights;
  %feature("kwargs") calcWeightsN;
  %feature("kwargs") setHiActiveWeights_f;
  %feature("kwargs") setActiveWeights_f;
#endif
 public:
  SubbandMMI(unsigned fftlen, bool half_band_shift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, float alpha=0.9, const String& nm = "SubbandMMI");
  ~SubbandMMI();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void use_binary_mask(float avgFactor=-1.0, unsigned fwidth=1, unsigned type=0);
  void calc_weights(  float samplerate, const gsl_matrix* delays);
  void calc_weights_n( float samplerate, const gsl_matrix* delays, unsigned NC=2);
  void set_hi_active_weights_f(unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0);
  void set_active_weights_f(unsigned fbinX, const gsl_matrix* packedWeights, int option=0);

#ifdef ENABLE_LEGACY_BTK_API
  void useBinaryMask( float avgFactor=-1.0, unsigned fwidth=1, unsigned type=0);
  void calcWeights(  float sampleRate, const gsl_matrix* delays);
  void calcWeightsN( float sampleRate, const gsl_matrix* delays, unsigned NC=2);
  void setHiActiveWeights_f( unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0);
  void setActiveWeights_f(unsigned fbinX, const gsl_matrix* packedWeights, int option=0);
#endif
};

class SubbandMMIPtr : public SubbandDSPtr {
  %feature("kwargs") SubbandMMIPtr;
 public:
  %extend {
   SubbandMMIPtr(unsigned fftlen, bool half_band_shift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, float alpha=0.9, const String& nm = "SubbandMMI") {
      return new SubbandMMIPtr(new SubbandMMI(fftlen, half_band_shift, targetSourceX, nSource, pfType, alpha, nm));
    }

    SubbandMMIPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMMI* operator->();
};

%feature("kwargs") calc_all_delays;
%feature("kwargs") calc_product;

void calc_all_delays(float x, float y, float z, const gsl_matrix* mpos, gsl_vector* delays);
void calc_product(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product);

#ifdef ENABLE_LEGACY_BTK_API
%feature("kwargs") calcAllDelays;
%feature("kwargs") calcProduct;

void calcAllDelays(float x, float y, float z, const gsl_matrix* mpos, gsl_vector* delays);
void calcProduct(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product);
#endif

// ----- definition for class `SubbandMVDR' -----
//
%ignore SubbandMVDR;
class SubbandMVDR : public SubbandDS {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") clear_channel;
  %feature("kwargs") calc_mvdr_weights;
  %feature("kwargs") mvdr_weights;
  %feature("kwargs") noise_spatial_spectral_matrix;
  %feature("kwargs") set_noise_spatial_spectral_matrix;
  %feature("kwargs") set_diffuse_noise_model;
  %feature("kwargs") set_all_diagonal_loading;
  %feature("kwargs") set_diagonal_looading;
  %feature("kwargs") divide_all_nondiagonal_elements;
  %feature("kwargs") divide_nondiagonal_elements;
  %feature("kwargs") noise_spatial_spectral_matrix;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") calcMVDRWeights;
  %feature("kwargs") getMVDRWeights;
  %feature("kwargs") getNoiseSpatialSpectralMatrix;
  %feature("kwargs") setNoiseSpatialSpectralMatrix;
  %feature("kwargs") setDiffuseNoiseModel;
  %feature("kwargs") setAllLevelsOfDiagonalLoading;
  %feature("kwargs") setLevelOfDiagonalLoading;
  %feature("kwargs") divideAllNonDiagonalElements;
  %feature("kwargs") divideNonDiagonalElements;
#endif
 public:
   SubbandMVDR(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDR();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual void clear_channel();
  bool calc_mvdr_weights(float samplerate, float dthreshold = 1.0E-8, bool calc_inverse_matrix = true);
  const gsl_vector_complex* mvdr_weights(unsigned fbinX);
  const gsl_matrix_complex *noise_spatial_spectral_matrix(unsigned fbinX);
  bool set_noise_spatial_spectral_matrix(unsigned fbinX, gsl_matrix_complex* Rnn);
  bool set_diffuse_noise_model(const gsl_matrix* micPositions, float samplerate, float sspeed = 343740.0);
  void set_all_diagonal_loading(float diagonalWeight);
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  void divide_all_nondiagonal_elements(float mu);
  void divide_nondiagonal_elements(unsigned fbinX, float mu);
  gsl_matrix_complex**  noise_spatial_spectral_matrix();

#ifdef ENABLE_LEGACY_BTK_API
  void clearChannel();
  bool calcMVDRWeights( float sampleRate, float dThreshold = 1.0E-8, bool calcInverseMatrix = true );
  const gsl_vector_complex* getMVDRWeights(unsigned fbinX);
  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix(unsigned fbinX);
  bool setNoiseSpatialSpectralMatrix(unsigned fbinX, gsl_matrix_complex* Rnn);
  bool setDiffuseNoiseModel(const gsl_matrix* micPositions, float sampleRate, float sspeed = 343740.0);
  void setAllLevelsOfDiagonalLoading(float diagonalWeight);
  void setLevelOfDiagonalLoading(unsigned fbinX, float diagonalWeight);
  void divideAllNonDiagonalElements( float mu );
  void divideNonDiagonalElements( unsigned fbinX, float mu );
  gsl_matrix_complex**  getNoiseSpatialSpectralMatrix();
#endif
};

class SubbandMVDRPtr : public SubbandDSPtr {
  %feature("kwargs") SubbandMVDRPtr;
 public:
  %extend {
    SubbandMVDRPtr(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandMVDR"){
      return new SubbandMVDRPtr(new SubbandMVDR( fftlen, half_band_shift, nm ));
    }

    SubbandMVDRPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMVDR* operator->();
};

// ----- definition for class `SubbandMVDRGSC' -----
//
%ignore SubbandMVDRGSC;

class SubbandMVDRGSC : public SubbandMVDR {
  %feature("kwargs") reset;
  %feature("kwargs") set_active_weights_f;
  %feature("kwargs") zero_active_weights;
  %feature("kwargs") calc_blocking_matrix1;
  %feature("kwargs") calc_blocking_matrix2;
  %feature("kwargs") upgrade_blocking_matrix;
  %feature("kwargs") blocking_matrix_output;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setActiveWeights_f;
  %feature("kwargs") calcBlockingMatrix1;
#endif
 public:
  SubbandMVDRGSC(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandMVDRGSC");
  ~SubbandMVDRGSC();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);
  void zero_active_weights();
  bool calc_blocking_matrix1(float samplerate, const gsl_vector* delaysT);
  bool calc_blocking_matrix2();
  void upgrade_blocking_matrix();
  const gsl_vector_complex* blocking_matrix_output(int outChanX=0);

#ifdef ENABLE_LEGACY_BTK_API
  void setActiveWeights_f(unsigned fbinX, const gsl_vector* packedWeight);
  void zeroActiveWeights();
  bool calcBlockingMatrix1(float sampleRate, const gsl_vector* delaysT);
  bool calcBlockingMatrix2();
  void upgradeBlockingMatrix();
#endif
};

class SubbandMVDRGSCPtr : public SubbandMVDRPtr {
  %feature("kwargs") SubbandMVDRGSCPtr;
 public:
  %extend {
    SubbandMVDRGSCPtr(unsigned fftlen, bool half_band_shift = false, const String& nm = "SubbandMVDRGSC"){
      return new SubbandMVDRGSCPtr(new SubbandMVDRGSC( fftlen, half_band_shift, nm ));
    }

    SubbandMVDRGSCPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMVDRGSC* operator->();
};

// ----- definition for class `SubbandOrthogonalizer' -----
//
%ignore SubbandOrthogonalizer;
class SubbandOrthogonalizer : public VectorComplexFeatureStream {
  %feature("kwargs") next;
 public:
  SubbandOrthogonalizer( SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer");
  ~SubbandOrthogonalizer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
};

class SubbandOrthogonalizerPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") SubbandOrthogonalizerPtr;
public:
  %extend {
    SubbandOrthogonalizerPtr( SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer"){
      return new SubbandOrthogonalizerPtr(new SubbandOrthogonalizer( beamformer, outChanX, nm ));
    }

    SubbandOrthogonalizerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandOrthogonalizer* operator->();
};

%feature("kwargs") modeAmplitude;
gsl_complex modeAmplitude( int order, double ka );

// ----- definition for class `ModeAmplitudeCalculator' -----
//
%ignore ModeAmplitudeCalculator;
class   ModeAmplitudeCalculator {
  %feature("kwargs") get;
public:
  ModeAmplitudeCalculator( int order, double minKa=0.01, double maxKa=20, double wid=0.01 );
  ~ModeAmplitudeCalculator();
  gsl_vector_complex *get();
};

class ModeAmplitudeCalculatorPtr {
  %feature("kwargs") ModeAmplitudeCalculatorPtr;
public:
  %extend {
    ModeAmplitudeCalculatorPtr( int order, double minKa=0.01, double maxKa=20, double wid=0.01 ){
      return new ModeAmplitudeCalculatorPtr(new ModeAmplitudeCalculator( order, minKa, maxKa, wid ));
    }
  }

  ModeAmplitudeCalculator* operator->();
};

// ----- definition for class `EigenBeamformer' -----
//
%ignore EigenBeamformer;
class EigenBeamformer : public  SubbandDS {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_sigma2;
  %feature("kwargs") set_weight_gain;
  %feature("kwargs") set_array_geometry;
  %feature("kwargs") set_look_direction;
  %feature("kwargs") array_geometry;
  %feature("kwargs") beampattern;
  %feature("kwargs") blocking_matrix;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setSigma2;
  %feature("kwargs") setWeightGain;
  %feature("kwargs") setArrayGeometry;
  %feature("kwargs") setLookDirection;
  %feature("kwargs") getArrayGeometry;
  %feature("kwargs") getBeamPattern;
  %feature("kwargs") getBlockingMatrix;
#endif
public:
  EigenBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "EigenBeamformer");
  ~EigenBeamformer();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual unsigned dim();

  void set_sigma2(float sigma2);
  void set_weight_gain(float wgain);
  void set_eigenmike_geometry();
  void set_array_geometry(double a,  gsl_vector *theta_s, gsl_vector *phi_s);
  virtual void set_look_direction(double theta, double phi);
  const gsl_matrix_complex *mode_amplitudes();
  const gsl_vector *array_geometry(int type); // type==0 -> theta, type==1 -> phi
  virtual gsl_matrix *beampattern(unsigned fbinX, double theta = 0, double phi = 0,
                                  double minTheta=-M_PI, double maxTheta=M_PI,
                                  double minPhi=-M_PI, double maxPhi=M_PI,
                                  double widthTheta=0.1, double widthPhi=0.1 );
  virtual SnapShotArrayPtr snapshot_array() const;
  virtual SnapShotArrayPtr snapshot_array2() const;
  const gsl_matrix_complex *blocking_matrix(unsigned fbinX, unsigned unitX=0 ) const;

#ifdef ENABLE_LEGACY_BTK_API
  void setSigma2(float simga2);
  void setWeightGain(float wgain);
  void setEigenMikeGeometry();
  void setArrayGeometry(double a,  gsl_vector *theta_s, gsl_vector *phi_s);
  void setLookDirection(double theta, double phi);
  const gsl_matrix_complex *getModeAmplitudes();
  gsl_vector *getArrayGeometry(int type);
  gsl_matrix *getBeamPattern(unsigned fbinX, double theta = 0, double phi = 0,
                             double minTheta=-M_PI, double maxTheta=M_PI,
                             double minPhi=-M_PI, double maxPhi=M_PI,
                             double widthTheta=0.1, double widthPhi=0.1);
  virtual SnapShotArrayPtr getSnapShotArray();
  virtual SnapShotArrayPtr getSnapShotArray2();
  const gsl_matrix_complex *getBlockingMatrix(unsigned fbinX, unsigned unitX=0);
#endif
};

class EigenBeamformerPtr : public  SubbandDSPtr {
  %feature("kwargs") EigenBeamformerPtr;
public:
  %extend {
    EigenBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "EigenBeamformer"){
      return new EigenBeamformerPtr(new EigenBeamformer( samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, nm ));
    }

    EigenBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EigenBeamformer* operator->();
};

// ----- definition for class `DOAEstimatorSRPBase' -----
//
%ignore DOAEstimatorSRPBase;
class DOAEstimatorSRPBase {
  %feature("kwargs") set_energy_threshold;
  %feature("kwargs") set_frequency_range;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setEnergyThreshold;
  %feature("kwargs") setFrequencyRange;
#endif

public:
  DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax );
  virtual ~DOAEstimatorSRPBase();

  const gsl_vector *nbest_rps();
  const gsl_matrix *nbest_doas();
  const gsl_matrix *response_power_matrix();
  float energy();
  void  final_nbest_hypotheses();
  void  set_energy_threshold(float engeryThreshold);
  void  set_frequency_range(unsigned fbinMin, unsigned fbinMax);
  void  init_accs();

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_vector *getNBestRPs();
  const gsl_matrix *getNBestDOAs();
  const gsl_matrix *getResponsePowerMatrix();
  float getEnergy();
  void  getFinalNBestHypotheses();
  void  setEnergyThreshold(float engeryThreshold);
  void  setFrequencyRange(unsigned fbinMin, unsigned fbinMax);
  void initAccs();
#endif
};

class DOAEstimatorSRPBasePtr {
  %feature("kwargs") DOAEstimatorSRPBasePtr;
public:
  %extend {
    DOAEstimatorSRPBasePtr( unsigned nBest, unsigned fbinMax ){
      return new DOAEstimatorSRPBasePtr( new DOAEstimatorSRPBase( nBest, fbinMax ) );
    }
  }
  DOAEstimatorSRPBase* operator->();
};

// ----- definition for class `DOAEstimatorSRPDSBLA' -----
//
%ignore DOAEstimatorSRPDSBLA;
class DOAEstimatorSRPDSBLA : public SubbandDS {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_array_geometry;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setArrayGeometry;
#endif
public:
  DOAEstimatorSRPDSBLA( unsigned nBest, unsigned samplerate, unsigned fftlen, const String& nm="DOAEstimatorSRPDSBLAPtr" );
  ~DOAEstimatorSRPDSBLA();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();
  void set_array_geometry(gsl_vector *positions);

#ifdef ENABLE_LEGACY_BTK_API
  void setArrayGeometry(gsl_vector *positions);
#endif
};

class DOAEstimatorSRPDSBLAPtr : public SubbandDSPtr {
  %feature("kwargs") DOAEstimatorSRPDSBLAPtr;
public:
  %extend {
    DOAEstimatorSRPDSBLAPtr( unsigned nBest, unsigned samplerate, unsigned fftlen, const String& nm="DOAEstimatorSRPDSBLAPtr" ){
      return new DOAEstimatorSRPDSBLAPtr( new DOAEstimatorSRPDSBLA( nBest, samplerate, fftlen, nm ) );
    }
    DOAEstimatorSRPDSBLAPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPDSBLA* operator->();
};

// ----- definition for class `DOAEstimatorSRPEB' -----
//
%ignore DOAEstimatorSRPEB;
class DOAEstimatorSRPEB : public EigenBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  DOAEstimatorSRPEB( unsigned nBest, unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB");
  ~DOAEstimatorSRPEB();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();
};

class DOAEstimatorSRPEBPtr : public EigenBeamformerPtr {
  %feature("kwargs") DOAEstimatorSRPEBPtr;
public:
  %extend {
    DOAEstimatorSRPEBPtr( unsigned nBest, unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB" ){
      return new DOAEstimatorSRPEBPtr( new DOAEstimatorSRPEB( nBest, samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, nm ) );
    }
    DOAEstimatorSRPEBPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPEB* operator->();
};

// ----- definition for class `SphericalDSBeamformer' -----
//
%ignore SphericalDSBeamformer;
class SphericalDSBeamformer : public EigenBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") calc_wng;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") calcWNG;
#endif
public:
  SphericalDSBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~SphericalDSBeamformer();
  const gsl_vector_complex* next(int frame_no = -5);
  void reset();
  virtual gsl_vector *calc_wng();

#ifdef ENABLE_LEGACY_BTK_API
  virtual gsl_vector *calcWNG();
#endif
};

class SphericalDSBeamformerPtr : public EigenBeamformerPtr {
  %feature("kwargs") SphericalDSBeamformerPtr;
public:
  %extend {
    SphericalDSBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer"){
      return new SphericalDSBeamformerPtr( new SphericalDSBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder,  normalizeWeight, nm ) );
    }
    SphericalDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalDSBeamformer* operator->();
};

// ----- definition for class `DualSphericalDSDSBeamformer' -----
//
%ignore DualSphericalDSBeamformer;
class DualSphericalDSBeamformer : public SphericalDSBeamformer {
public:
  DualSphericalDSBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalDSBeamformer");
  ~DualSphericalDSBeamformer();
  virtual SnapShotArrayPtr snapshot_array();

#ifdef ENABLE_LEGACY_BTK_API
  virtual SnapShotArrayPtr getSnapShotArray();
#endif
};

class DualSphericalDSBeamformerPtr : public SphericalDSBeamformerPtr {
  %feature("kwargs") DualSphericalDSBeamformerPtr;
public:
  %extend {
    DualSphericalDSBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalDSBeamformer"){
      return new DualSphericalDSBeamformerPtr( new DualSphericalDSBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder,  normalizeWeight, nm ) );
    }
    DualSphericalDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DualSphericalDSBeamformer* operator->();
};

// ----- definition for class DOAEstimatorSRPSphDSB' -----
// 
%ignore DOAEstimatorSRPSphDSB;
class DOAEstimatorSRPSphDSB : public SphericalDSBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  DOAEstimatorSRPSphDSB( unsigned nBest, unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DOAEstimatorSRPSphDSB");
  ~DOAEstimatorSRPSphDSB();
  const gsl_vector_complex* next(int frame_no = -5);
  void reset();
};

class DOAEstimatorSRPSphDSBPtr : public SphericalDSBeamformerPtr {
  %feature("kwargs") DOAEstimatorSRPSphDSBPtr;
public:
  %extend {
    DOAEstimatorSRPSphDSBPtr( unsigned nBest, unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB" ){
      return new DOAEstimatorSRPSphDSBPtr( new DOAEstimatorSRPSphDSB( nBest, samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, nm ) );
    }
    DOAEstimatorSRPSphDSBPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPSphDSB* operator->();
};

// ----- definition for class `SphericalHWNCBeamformer' -----
//
%ignore SphericalHWNCBeamformer;
class SphericalHWNCBeamformer : public EigenBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") calc_wng;
  %feature("kwargs") set_wng;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") calcWNG;
  %feature("kwargs") setWNG;
#endif
public:
  SphericalHWNCBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalHWNCBeamformer();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  void reset();
  virtual gsl_vector *calc_wng();
  void set_wng( double ratio);

#ifdef ENABLE_LEGACY_BTK_API
  gsl_vector *calcWNG();
  void setWNG( double ratio);
#endif
};

class SphericalHWNCBeamformerPtr : public EigenBeamformerPtr {
  %feature("kwargs") SphericalHWNCBeamformerPtr;
public:
  %extend {
    SphericalHWNCBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalHWNCBeamformer"){
      return new SphericalHWNCBeamformerPtr( new SphericalHWNCBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, ratio, nm ) );
    }
    SphericalHWNCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalHWNCBeamformer* operator->();
};

// ----- definition for class `SphericalGSCBeamformer' -----
// 
%ignore SphericalGSCBeamformer;
class SphericalGSCBeamformer : public SphericalDSBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_look_direction;
  %feature("kwargs") set_active_weights_f;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setLookDirection;
  %feature("kwargs") setActiveWeights_f;
#endif
public:
  SphericalGSCBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer");
  ~SphericalGSCBeamformer();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();

  void set_look_direction(double theta, double phi);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);

#ifdef ENABLE_LEGACY_BTK_API
  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
#endif
};

class SphericalGSCBeamformerPtr : public SphericalDSBeamformerPtr {
  %feature("kwargs") SphericalGSCBeamformerPtr;
public:
  %extend {
    SphericalGSCBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer" ){
      return new SphericalGSCBeamformerPtr( new SphericalGSCBeamformer( samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, nm ) );
    }
    SphericalGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalGSCBeamformer* operator->();
};

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 
%ignore SphericalHWNCGSCBeamformer;
class SphericalHWNCGSCBeamformer : public SphericalHWNCBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_look_direction;
  %feature("kwargs") set_active_weights_f;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setLookDirection;
  %feature("kwargs") setActiveWeights_f;
#endif
public:
  SphericalHWNCGSCBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer");
  ~SphericalHWNCGSCBeamformer();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();
  void set_look_direction(double theta, double phi);
  void set_active_weights_f(unsigned fbinX, const gsl_vector* packedWeight);

#ifdef ENABLE_LEGACY_BTK_API
  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
#endif
};

class SphericalHWNCGSCBeamformerPtr : public SphericalHWNCBeamformerPtr {
  %feature("kwargs") SphericalHWNCGSCBeamformerPtr;
public:
  %extend {
    SphericalHWNCGSCBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer" ){
      return new SphericalHWNCGSCBeamformerPtr( new SphericalHWNCGSCBeamformer( samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, ratio, nm ) );
    }
    SphericalHWNCGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalHWNCGSCBeamformer* operator->();
};

// ----- definition for class `DualSphericalGSCBeamformer' -----
//
%ignore DualSphericalGSCBeamformer;
class DualSphericalGSCBeamformer : public SphericalGSCBeamformer {
public:
  DualSphericalGSCBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer");
  ~DualSphericalGSCBeamformer();

  virtual SnapShotArrayPtr snapshot_array() const;

#ifdef ENABLE_LEGACY_BTK_API
  virtual SnapShotArrayPtr getSnapShotArray();
#endif
};

class DualSphericalGSCBeamformerPtr : public SphericalGSCBeamformerPtr {
  %feature("kwargs") DualSphericalGSCBeamformerPtr;
public:
  %extend {
    DualSphericalGSCBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer"){
      return new DualSphericalGSCBeamformerPtr( new DualSphericalGSCBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder,  normalizeWeight, nm ) );
    }
    DualSphericalGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DualSphericalGSCBeamformer* operator->();
};

// ----- definition for class `SphericalMOENBeamformer' -----
// 
%ignore SphericalMOENBeamformer;
class SphericalMOENBeamformer : public SphericalDSBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") fix_terms;
  %feature("kwargs") set_diagonal_looading;
  %feature("kwargs") snapshot_array;
  %feature("kwargs") beampattern;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") fixTerms;
  %feature("kwargs") setLevelOfDiagonalLoading;
  %feature("kwargs") getBeamPattern;
#endif
public:
  SphericalMOENBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer");
  ~SphericalMOENBeamformer();

  const gsl_vector_complex* next(int frame_no = -5);
  void reset();

  void fix_terms(bool flag);
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  virtual SnapShotArrayPtr snapshot_array() const;
  gsl_matrix *beampattern( unsigned fbinX, double theta = 0, double phi = 0,
                           double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI,
                           double widthTheta=0.1, double widthPhi=0.1 );

#ifdef ENABLE_LEGACY_BTK_API
  void fixTerms(bool flag);
  void setLevelOfDiagonalLoading(unsigned fbinX, float diagonalWeight);
  gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
                              double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI,
                              double widthTheta=0.1, double widthPhi=0.1 );
#endif
};

class SphericalMOENBeamformerPtr : public SphericalDSBeamformerPtr {
  %feature("kwargs") SphericalMOENBeamformerPtr;
public:
  %extend {
    SphericalMOENBeamformerPtr( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer" ){
      return new SphericalMOENBeamformerPtr( new SphericalMOENBeamformer( samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, nm ) );
    }
    SphericalMOENBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalMOENBeamformer* operator->();
};

// ----- definition for class `TaylorSeries' -----
//
%ignore nonamePdf;
class nonamePdf {
 public:
  nonamePdf();
  ~nonamePdf();

  bool loadCoeffDescFile( const String &coefDescfn );
};

class nonamePdfPtr {
 public:
  %extend {
    nonamePdfPtr(){
      return new nonamePdfPtr(new nonamePdf());
    }
  }

  nonamePdf* operator->();
};

%ignore gammaPdf;
class gammaPdf : public nonamePdf {
 public:
  gammaPdf(int numberOfVariate = 2 );
  ~gammaPdf();
  double calcLog( double x, int N );
  double calcDerivative1( double x, int N );
  void  bi();
  void  four();
  void  printCoeff();
};

class gammaPdfPtr : public nonamePdfPtr {
 public:
  %extend {
    gammaPdfPtr(int numberOfVariate = 2){
      return new gammaPdfPtr(new gammaPdf(numberOfVariate));
    }
  }

  gammaPdf* operator->();
};

// ----- definition of class 'ModalDecomposition' -----
//
%ignore ModalDecomposition;
class ModalDecomposition {
  %feature("kwargs") harmonic;
  %feature("kwargs") harmonic_deriv_polar_angle;
  %feature("kwargs") harmonic_deriv_azimuth;
  %feature("kwargs") modal_coefficient;
  %feature("kwargs") estimate_Bkl;
  %feature("kwargs") transform;
  %feature("kwargs") linearize;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") harmonicDerivPolarAngle;
  %feature("kwargs") harmonicDerivAzimuth;
  %feature("kwargs") modalCoefficient;
#endif
 public:
  ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double samplerate, unsigned useSubbandsN = 0);
  ~ModalDecomposition();

  gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonic_deriv_polar_angle(int order, int degree, double theta, double phi);
  gsl_complex harmonic_deriv_azimuth(int order, int degree, double theta, double phi);
  gsl_complex modal_coefficient(unsigned order, double ka);
  void estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  gsl_matrix_complex* linearize(gsl_vector* xk);

#ifdef ENABLE_LEGACY_BTK_API
  gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi);
  gsl_complex modalCoefficient(unsigned order, double ka);
#endif
};

class ModalDecompositionPtr {
  %feature("kwargs") ModalDecompositionPtr;
public:
  %extend {
    ModalDecompositionPtr(unsigned orderN, unsigned subbandsN, double a, double samplerate, unsigned useSubbandsN = 0) {
      return new ModalDecompositionPtr(new ModalDecomposition(orderN, subbandsN, a, samplerate, useSubbandsN));
    }
  }
};


// ----- definition of class 'SpatialDecomposition' -----
//
%ignore SpatialDecomposition;
class SpatialDecomposition {
  %feature("kwargs") harmonic;
  %feature("kwargs") harmonic_deriv_polar_angle;
  %feature("kwargs") harmonic_deriv_azimuth;
  %feature("kwargs") modal_coefficient;
  %feature("kwargs") estimate_Bkl;
  %feature("kwargs") transform;
  %feature("kwargs") linearize;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") harmonicDerivPolarAngle;
  %feature("kwargs") harmonicDerivAzimuth;
  %feature("kwargs") modalCoefficient;
#endif
 public:
  SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double samplerate, unsigned useSubbandsN = 0);
  ~SpatialDecomposition();

  gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonic_deriv_polar_angle(int order, int degree, double theta, double phi);
  gsl_complex harmonic_deriv_azimuth(int order, int degree, double theta, double phi);
  gsl_complex modal_coefficient(unsigned order, double ka);
  void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  void estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  gsl_matrix_complex* linearize(gsl_vector* xk);

#ifdef ENABLE_LEGACY_BTK_API
  gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi);
  gsl_complex modalCoefficient(unsigned order, double ka);
#endif
};

class SpatialDecompositionPtr {
  %feature("kwargs") SpatialDecompositionPtr;
public:
  %extend {
    SpatialDecompositionPtr(unsigned orderN, unsigned subbandsN, double a, double samplerate, unsigned useSubbandsN = 0) {
      return new SpatialDecompositionPtr(new SpatialDecomposition(orderN, subbandsN, a, samplerate, useSubbandsN));
    }
  }
};


// ----- definition of class 'ModalSphericalArrayTracker' -----
//
%ignore ModalSphericalArrayTracker;
class ModalSphericalArrayTracker {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") chanN;
  %feature("kwargs") set_channel;
  %feature("kwargs") set_V;
  %feature("kwargs") set_initial_position;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setChannel;
  %feature("kwargs") setV;
  %feature("kwargs") setInitialPosition;
#endif
 public:
  ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker");
  ~ModalSphericalArrayTracker();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();

  unsigned chanN() const { return _channelList.size(); }
  void set_channel(VectorComplexFeatureStreamPtr& chan);
  void set_V(const gsl_matrix_complex* Vk, unsigned subbandX);
  void next_speaker();
  void set_initial_position(double theta, double phi);

#ifdef ENABLE_LEGACY_BTK_API
  void setChannel(VectorComplexFeatureStreamPtr& chan);
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX);
  void nextSpeaker();
  void setInitialPosition(double theta, double phi);
#endif
};

class ModalSphericalArrayTrackerPtr {
  %feature("kwargs") ModalSphericalArrayTrackerPtr;
public:
  %extend {
    ModalSphericalArrayTrackerPtr(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
				  unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker") {
      return new ModalSphericalArrayTrackerPtr(new ModalSphericalArrayTracker(modalDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm));
    }

    ModalSphericalArrayTrackerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ModalSphericalArrayTracker* operator->();
};


// ----- definition of class 'SpatialSphericalArrayTracker' -----
//
%ignore SpatialSphericalArrayTracker;
class SpatialSphericalArrayTracker {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") chanN;
  %feature("kwargs") set_channel;
  %feature("kwargs") set_V;
  %feature("kwargs") set_initial_position;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setChannel;
  %feature("kwargs") setV;
  %feature("kwargs") setInitialPosition;
#endif
 public:
  SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			       unsigned maxLocalN = 1, const String& nm = "SpatialSphericalArrayTracker");
  ~SpatialSphericalArrayTracker();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();

  unsigned chanN() const { return _channelList.size(); }
  void set_channel(VectorComplexFeatureStreamPtr& chan);
  void set_V(const gsl_matrix_complex* Vk, unsigned subbandX);
  void next_speaker();
  void set_initial_position(double theta, double phi);

#ifdef ENABLE_LEGACY_BTK_API
  void setChannel(VectorComplexFeatureStreamPtr& chan);
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX);
  void nextSpeaker();
  void setInitialPosition(double theta, double phi);
#endif
};

class SpatialSphericalArrayTrackerPtr {
  %feature("kwargs") SpatialSphericalArrayTrackerPtr;
public:
  %extend {
    SpatialSphericalArrayTrackerPtr(SpatialDecompositionPtr& spatialDecomposition,
                                    double sigma2_u = 10.0,
                                    double sigma2_v = 10.0,
                                    double sigma2_init = 10.0,
                                    unsigned maxLocalN = 1,
                                    const String& nm = "SpatialSphericalArrayTracker") {
      return new SpatialSphericalArrayTrackerPtr(new SpatialSphericalArrayTracker(spatialDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm));
    }

    SpatialSphericalArrayTrackerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpatialSphericalArrayTracker* operator->();
};


// ----- definition of class 'PlaneWaveSimulator' -----
//
%ignore PlaneWaveSimulator;
class PlaneWaveSimulator : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
		     unsigned channelX, double theta, double phi, const String& nm = "Plane Wave Simulator");
  ~PlaneWaveSimulator();

  virtual const gsl_complex_float* next(int frame_no = -5);
  virtual void reset();
};

class PlaneWaveSimulatorPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") PlaneWaveSimulatorPtr;
public:
  %extend {
    PlaneWaveSimulatorPtr(const VectorComplexFeatureStreamPtr& source,
                          ModalDecompositionPtr& modalDecomposition,
                          unsigned channelX,
                          double theta,
                          double phi,
                          const String& nm = "Plane Wave Simulator") {
      return new PlaneWaveSimulatorPtr(new PlaneWaveSimulator(source, modalDecomposition, channelX, theta, phi, nm));
    }

    PlaneWaveSimulatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PlaneWaveSimulator* operator->();
};


// ----- definition for class `SphericalSpatialDSBeamformer' -----
//
%ignore SphericalSpatialDSBeamformer;
class SphericalSpatialDSBeamformer : public SphericalDSBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  SphericalSpatialDSBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalSpatialDSBeamformer");
  ~SphericalSpatialDSBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  void reset();
};

class SphericalSpatialDSBeamformerPtr : public SphericalDSBeamformerPtr {
  %feature("kwargs") SphericalSpatialDSBeamformerPtr;
public:
  %extend {
    SphericalSpatialDSBeamformerPtr( unsigned samplerate,
                                     unsigned fftlen,
                                     bool half_band_shift = false,
                                     unsigned NC=1,
                                     unsigned maxOrder=3,
                                     bool normalizeWeight=false,
                                     const String& nm = "SphericalSpatialDSBeamformer"){
      return new SphericalSpatialDSBeamformerPtr( new SphericalSpatialDSBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder,  normalizeWeight, nm ) );
    }
    SphericalSpatialDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalSpatialDSBeamformer* operator->();
};


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
//
%ignore SphericalSpatialHWNCBeamformer;
class SphericalSpatialHWNCBeamformer : public SphericalHWNCBeamformer {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  SphericalSpatialHWNCBeamformer( unsigned samplerate, unsigned fftlen, bool half_band_shift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalSpatialHWNCBeamformer");
  ~SphericalSpatialHWNCBeamformer();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  void reset();
};

class SphericalSpatialHWNCBeamformerPtr : public SphericalHWNCBeamformerPtr {
  %feature("kwargs") SphericalSpatialHWNCBeamformerPtr;
public:
  %extend {
    SphericalSpatialHWNCBeamformerPtr( unsigned samplerate,
                                       unsigned fftlen,
                                       bool half_band_shift = false,
                                       unsigned NC=1,
                                       unsigned maxOrder=3,
                                       bool normalizeWeight=false,
                                       float ratio=0.1,
                                       const String& nm = "SphericalSpatialHWNCBeamformer"){
      return new SphericalSpatialHWNCBeamformerPtr( new SphericalSpatialHWNCBeamformer(samplerate, fftlen, half_band_shift, NC, maxOrder, normalizeWeight, ratio, nm ) );
    }
    SphericalSpatialHWNCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalSpatialHWNCBeamformer* operator->();
};
