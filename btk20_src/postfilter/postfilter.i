/**
 * @file postfilter.i
 * @brief Array post-filter for array and single channel filter
 * @author Kenichi Kumatani
 */

%module(package="btk20") postfilter

%{
#include "stream/stream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "postfilter/postfilter.h"
#include "postfilter/spectralsubtraction.h"
#include "postfilter/binauralprocessing.h"
#include "stream/pyStream.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

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


// ----- definition for class 'ZelinskiPostFilter' -----
//
%ignore ZelinskiPostFilter;
class ZelinskiPostFilter: public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_beamformer;
  %feature("kwargs") set_snapshot_array;
  %feature("kwargs") set_array_manifold_vector;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setBeamformer;
  %feature("kwargs") setSnapShotArray;
  %feature("kwargs") setArrayManifoldVector;
#endif
public:
  ZelinskiPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, in minFrames=0, const String& nm = "ZelinskPostFilter" );
  ~ZelinskiPostFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void set_beamformer(SubbandDSPtr &beamformer);
  void set_snapshot_array(SnapShotArrayPtr &snapShotArray);
  void set_array_manifold_vector(unsigned fbinX, gsl_vector_complex *arrayManifoldVector, bool halfBandShift, unsigned NC = 1);
  const gsl_vector_complex* postfilter_weights();

#ifdef ENABLE_LEGACY_BTK_API
  void setBeamformer( SubbandDSPtr &beamformer );
  void setSnapShotArray( SnapShotArrayPtr &snapShotArray );
  void setArrayManifoldVector( unsigned fbinX, gsl_vector_complex *arrayManifold, bool halfBandShift, unsigned NC = 1 );
  const gsl_vector_complex* getPostFilterWeights();
#endif
};

class ZelinskiPostFilterPtr: public VectorComplexFeatureStreamPtr {
  %feature("kwargs") ZelinskiPostFilterPtr;
public:
  %extend {
    ZelinskiPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned M, double alpha=0.6, int type=2, unsigned minFrames=0, const String& nm = "ZelinskPostFilter" ){
      return new ZelinskiPostFilterPtr(new ZelinskiPostFilter( output, M, alpha, type, minFrames, nm));
    }

    ZelinskiPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ZelinskiPostFilter* operator->();
};

// ----- definition for class 'McCowanPostFilter' -----
// 
%ignore McCowanPostFilter;
class McCowanPostFilter: public ZelinskiPostFilter {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") noise_spatial_spectral_matrix;
  %feature("kwargs") set_noise_spatial_spectral_matrix;
  %feature("kwargs") set_diffuse_noise_model;
  %feature("kwargs") set_all_diagonal_loading;
  %feature("kwargs") set_diagonal_looading;
  %feature("kwargs") divide_all_nondiagonal_elements;
  %feature("kwargs") divide_nondiagonal_elements;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") getNoiseSpatialSpectralMatrix;
  %feature("kwargs") setNoiseSpatialSpectralMatrix;
  %feature("kwargs") setDiffuseNoiseModel;
  %feature("kwargs") setAllLevelsOfDiagonalLoading;
  %feature("kwargs") setLevelOfDiagonalLoading;
  %feature("kwargs") divideAllNonDiagonalElements;
  %feature("kwargs") divideNonDiagonalElements;
#endif
public:
  McCowanPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, in minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilterPtr" );
  ~McCowanPostFilter();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  const gsl_matrix_complex *noise_spatial_spectral_matrix( unsigned fbinX );
  bool set_noise_spatial_spectral_matrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool set_diffuse_noise_model( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void set_all_diagonal_loading(float diagonalWeight);
  void set_diagonal_looading(unsigned fbinX, float diagonalWeight);
  void divide_all_nondiagonal_elements(float mu);
  void divide_nondiagonal_elements(unsigned fbinX, float mu);

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix( unsigned fbinX );
  bool setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void setAllLevelsOfDiagonalLoading( float diagonalWeight );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  void divideAllNonDiagonalElements( float mu );
  void divideNonDiagonalElements( unsigned fbinX, float mu );
#endif
};

class McCowanPostFilterPtr: public ZelinskiPostFilterPtr {
  %feature("kwargs") McCowanPostFilterPtr;
public:
  %extend {
    McCowanPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, unsigned minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilterPtr" ){
      return new McCowanPostFilterPtr(new McCowanPostFilter( output, fftLen, alpha, type, minFrames, threshold, nm));
    }

    McCowanPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  McCowanPostFilter* operator->();
};

// ----- definition for class 'LefkimmiatisPostFilter' -----
// 
%ignore LefkimmiatisPostFilter;
class LefkimmiatisPostFilter: public McCowanPostFilter  {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  LefkimmiatisPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, in minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilterPtr" );
  ~LefkimmiatisPostFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

  void calc_inverse_noise_spatial_spectral_matrix();

#ifdef ENABLE_LEGACY_BTK_API
  void calcInverseNoiseSpatialSpectralMatrix();
#endif
};

class LefkimmiatisPostFilterPtr: public McCowanPostFilterPtr {
  %feature("kwargs") LefkimmiatisPostFilterPtr;
public:
  %extend {
    LefkimmiatisPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, unsigned minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilterPtr" ){
      return new LefkimmiatisPostFilterPtr(new LefkimmiatisPostFilter( output, fftLen, minSV, fbinX1, alpha, type, minFrames, threshold, nm));
    }

    LefkimmiatisPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LefkimmiatisPostFilter* operator->();
};

// ----- definition for class 'SpectralSubtractor' -----
// 
%ignore SpectralSubtractor;

class SpectralSubtractor : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_channel;
  %feature("kwargs") set_noise_over_estimation_factor;
  %feature("kwargs") read_noise_file;
  %feature("kwargs") write_noise_file;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setChannel;
  %feature("kwargs") setNoiseOverEstimationFactor;
  %feature("kwargs") readNoiseFile;
  %feature("kwargs") writeNoiseFile;
#endif
 public:
  SpectralSubtractor(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor");
  ~SpectralSubtractor();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void clear();
  void set_noise_over_estimation_factor(float ft);
  void set_channel(VectorComplexFeatureStreamPtr& chan, double alpha=-1 );
  void start_training();
  void stop_training();
  void clear_noise_samples();
  void start_noise_subtraction();
  void stop_noise_subtraction();
  bool read_noise_file(const String& fn, unsigned idx=0);
  bool write_noise_file(const String& fn, unsigned idx=0);

#ifdef ENABLE_LEGACY_BTK_API
  void setChannel(VectorComplexFeatureStreamPtr& chan, double alpha=-1);
  void setNoiseOverEstimationFactor(float ft);
  void startTraining();
  void stopTraining();
  void clearNoiseSamples();
  void startNoiseSubtraction();
  void stopNoiseSubtraction();
  bool readNoiseFile( const String& fn, unsigned idx=0 );
  bool writeNoiseFile( const String& fn, unsigned idx=0 );
#endif
};

class SpectralSubtractorPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") SpectralSubtractorPtr;
 public:
  %extend {
    SpectralSubtractorPtr(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor"){
      return new SpectralSubtractorPtr(new SpectralSubtractor( fftLen, halfBandShift, ft, flooringV, nm));
    }

    SpectralSubtractorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralSubtractor* operator->();
};

// ----- definition for class 'WienerFilter' -----
// 
%ignore WienerFilter;
class WienerFilter : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setNoiseAmplificationFactor;
#endif
 public:
  WienerFilter(VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter");
  ~WienerFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

#ifdef ENABLE_LEGACY_BTK_API
  void    setNoiseAmplificationFactor( double beta );
  void    startUpdatingNoisePSD();
  void    stopUpdatingNoisePSD();
#endif
};

class WienerFilterPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") WienerFilterPtr;
 public:
  %extend {
    WienerFilterPtr(VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter"){
      return new WienerFilterPtr(new WienerFilter( targetSignal, noiseSignal, halfBandShift, alpha, flooringV, beta, nm));
    }

    WienerFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WienerFilter* operator->();
};

// ----- definition for class 'HighPassFilter' -----
// 
%ignore HighPassFilter;

class HighPassFilter : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  HighPassFilter( VectorComplexFeatureStreamPtr &output, float cutOffFreq = 150, int sampleRate, const String& nm = "HighPassFilter" );
  ~HighPassFilter();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
};

class HighPassFilterPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") HighPassFilterPtr;
 public:
  %extend {
    HighPassFilterPtr( VectorComplexFeatureStreamPtr &output, float cutOffFreq, int sampleRate, const String& nm = "HighPassFilter" ){
      return new HighPassFilterPtr(new HighPassFilter( output, cutOffFreq, sampleRate, nm));
    }

    HighPassFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  HighPassFilter* operator->();
};

// ----- definition for class 'BinaryMaskFilter' -----
// 
%ignore BinaryMaskFilter;
class BinaryMaskFilter : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_threshold;
  %feature("kwargs") set_thresholds;
  %feature("kwargs") threshold;
  %feature("kwargs") thresholds;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setThreshold;
  %feature("kwargs") setThresholds;
#endif
public:
  BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha,
		       float dEta = 0.01, const String& nm = "BinaryMaskFilter" );
  ~BinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void set_threshold( float threshold );
  void set_thresholds( const gsl_vector *thresholds );
  double threshold();
  gsl_vector *thresholds();

#ifdef ENABLE_LEGACY_BTK_API
  void setThreshold( float threshold );
  void setThresholds( const gsl_vector *thresholds );
  double getThreshold();
  gsl_vector *getThresholds();
#endif
};

class BinaryMaskFilterPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") BinaryMaskFilterPtr;
public:
  %extend {
    BinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float threshold, float alpha, float dEta = 0.01, const String& nm = "BinaryMaskFilter" ){
      return new BinaryMaskFilterPtr(new BinaryMaskFilter(chanX, srcL, srcR, M, threshold, alpha, dEta, nm ));
    }
    BinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BinaryMaskFilter* operator->();
};

// ----- definition for class 'KimBinaryMaskFilter' -----
// 
%ignore KimBinaryMaskFilter;

class KimBinaryMaskFilter : public BinaryMaskFilter {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") masking1;
 public:
  KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,float threshold, float alpha, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimBinaryMaskFilter" );
  ~KimBinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );
};

class KimBinaryMaskFilterPtr : public BinaryMaskFilterPtr {
  %feature("kwargs") KimBinaryMaskFilterPtr;
public:
  %extend {
    KimBinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float threshold, float alpha, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimBinaryMaskFilter" ){
      return new KimBinaryMaskFilterPtr(new KimBinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta,dPowerCoeff, nm ));
    }

    KimBinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  KimBinaryMaskFilter* operator->();
};

// ----- definition for class 'KimITDThresholdEstimator' -----
// 
%ignore KimITDThresholdEstimator;

class KimITDThresholdEstimator : public KimBinaryMaskFilter {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  KimITDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
			    float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			    float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimITDThresholdEstimator" );
  ~KimITDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual double calc_threshold();
  const gsl_vector* cost_function();

#ifdef ENABLE_LEGACY_BTK_API
  virtual double calcThreshold();
  const gsl_vector* getCostFunction();
#endif
};

class KimITDThresholdEstimatorPtr : public KimBinaryMaskFilterPtr {
  %feature("kwargs") KimITDThresholdEstimatorPtr;
public:
  %extend {
    KimITDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
				 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
				 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimITDThresholdEstimator" ){
      return new KimITDThresholdEstimatorPtr( new KimITDThresholdEstimator( srcL, srcR,M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, dEta, dPowerCoeff, nm ) );
    }
    KimITDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  KimITDThresholdEstimator* operator->();
};

// ----- definition for class 'IIDBinaryMaskFilter' -----
// 
%ignore IIDBinaryMaskFilter;

class IIDBinaryMaskFilter : public BinaryMaskFilter {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") masking1;
public:
  IIDBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
                       unsigned M, float threshold, float alpha,
                       float dEta = 0.01, const String& nm = "IIDBinaryMaskFilter" );
  ~IIDBinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );
};

class IIDBinaryMaskFilterPtr : public BinaryMaskFilterPtr {
  %feature("kwargs") IIDBinaryMaskFilterPtr;
public:
  %extend {
    IIDBinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
			    unsigned M, float threshold, float alpha,
			    float dEta = 0.01, const String& nm = "IIDBinaryMaskFilter" ){
      return new IIDBinaryMaskFilterPtr( new IIDBinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm ) );
    }
    IIDBinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IIDBinaryMaskFilter* operator->();
};


// ----- definition for class 'IIDThresholdEstimator' -----
// 
%ignore IIDThresholdEstimator;

class IIDThresholdEstimator : public KimITDThresholdEstimator {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
			 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "IIDThresholdEstimator" );
  ~IIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  virtual double calc_threshold();

#ifdef ENABLE_LEGACY_BTK_API
  virtual double calcThreshold();
#endif
};

class IIDThresholdEstimatorPtr : public KimITDThresholdEstimatorPtr {
  %feature("kwargs") IIDThresholdEstimatorPtr;
public:
  %extend {
    IIDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
			      float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			      float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "IIDThresholdEstimator" ){
      return new IIDThresholdEstimatorPtr( new IIDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, dEta, dPowerCoeff, nm ) );
    }

    IIDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IIDThresholdEstimator* operator->();
};

// ----- definition for class 'FDIIDThresholdEstimator' -----
// 
%ignore FDIIDThresholdEstimator;

class FDIIDThresholdEstimator : public BinaryMaskFilter {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") cost_function;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") getCostFunction;
#endif
public:
  FDIIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 1000, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "FDIIDThresholdEstimator" );
  ~FDIIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void   reset();
  virtual double calc_threshold();
  const gsl_vector* cost_function(unsigned freqX);

#ifdef ENABLE_LEGACY_BTK_API
  virtual double calcThreshold();
  const gsl_vector* getCostFunction( unsigned freqX );
#endif
};

class FDIIDThresholdEstimatorPtr : public BinaryMaskFilterPtr {
  %feature("kwargs") FDIIDThresholdEstimatorPtr;
public:
  %extend {
    FDIIDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 1000, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "FDIIDThresholdEstimator" ){
      return new FDIIDThresholdEstimatorPtr( new FDIIDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, dEta, dPowerCoeff, nm) );
    }

    FDIIDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FDIIDThresholdEstimator* operator->();
};
