/**
 * @file localization.i
 * @brief source localization
 * @author John McDonough and Kenichi Kumatani
 */

%module(package="btk20") localization

%{
#include "stream/stream.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
#include <stdio.h>
#include "localization/localization.h"
#include "localization/mcc_localizer.h"
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

gsl_vector* getSrpPhat(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeX, gsl_vector *searchRangeY, gsl_matrix *arrgeom, int zPos);
void calcDelays(int x, int y, int z, const gsl_matrix* mpos, gsl_vector* delays);

gsl_vector* getDelays(double delta_f, gsl_matrix_complex* mFramePerChannel, gsl_vector* searchRange);

double getAzimuth(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_matrix *arrgeom, gsl_vector *delays);

double getPlaneWaveSrp(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeY, gsl_matrix *arrgeom);

//void halfComplexPack(double* tgt, const gsl_vector_complex* src);

void getGCCRaw(const gsl_matrix_complex* spectralSample, double sampleRate, gsl_vector* gcc);

const gsl_vector* getGCC(gsl_matrix_complex *spectralSample, double sampleRate);
const gsl_vector* getWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCratio(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCdirect(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCabs(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getDynWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay, double wMinDelay, double wMaxDelay, double threshold);

double getInterpolation(gsl_matrix *crossResult, int delayPos);

gsl_vector* get3DPosition(gsl_vector *yCoord, gsl_vector* azimuth1, gsl_vector* azimuth2, double xPos, double zPos);

gsl_vector* get3DPosition_T_shape(gsl_matrix *arrgeom1, int arrayNr1,  gsl_matrix *arrgeom2, int arrayNr2,  gsl_matrix *arrgeom3, double azimuth1, double azimuth2, double azimuth3);

gsl_vector* getGCC_old(gsl_matrix_complex *spectralSample, double delta_f, gsl_vector *delays);

gsl_matrix* getLowerTriangMatrix(gsl_matrix* fullMatrix);

gsl_vector* getXi(gsl_matrix* D1_2);

class GCC
{
 public:
  GCC(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
  virtual ~GCC();
  void calculate(const gsl_vector_complex *spectralSample1, int chan1, const gsl_vector_complex *spectralSample2, int chan2, int pair, double timestamp, bool sad = false, bool smooth = true);
  const gsl_vector* findMaximum(double minDelay = -HUGE, double maxDelay = HUGE);
  double getPeakDelay();
  double getPeakCorr();
  double getRatio();
  const gsl_vector* getNoisePowerSpectrum(int chan);
  const gsl_vector_complex* getNoiseCrossSpectrum(int pair);
  const gsl_vector_complex* getCrossSpectrum();
  const gsl_vector* getCrossCorrelation();
  void setAlpha(double alpha);
  double getAlpha();
};

class GCCRaw : public GCC
{
 public:
  GCCRaw(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};

class GCCGnnSub : public GCC
{
 public:
  GCCGnnSub(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};
class GCCPhat : public GCC
{
 public:
  GCCPhat(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};
class GCCGnnSubPhat : public GCC
{
 public:
  GCCGnnSubPhat(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};
class GCCMLRRaw : public GCC
{
 public:
  GCCMLRRaw(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};

class GCCMLRGnnSub : public GCC
{
 public:
  GCCMLRGnnSub(double sampleRate = 44100.0 , int fftLen = 2048, int nChan = 16, int pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
};

%ignore SearchGridBuilder;
class SearchGridBuilder 
{
 public:
  SearchGridBuilder( int nChan, bool isFarField, unsigned int samplingFreq=16000);
  ~SearchGridBuilder();

  virtual bool nextSearchGrid();
  virtual const gsl_vector *getTimeDelays();

  const gsl_vector *getSearchPosition();
  float maxTimeDelay();
  size_t chanN();
  unsigned int samplingFrequency();
  void reset();
};

class SearchGridBuilderPtr 
{
 public:
  %extend {
    SearchGridBuilderPtr( int nChan, bool isFarField, unsigned int samplingFreq=16000 ){
      return new SearchGridBuilderPtr( new SearchGridBuilder( nChan, isFarField, samplingFreq ));
    }
  }

  SearchGridBuilder* operator->();
};

%ignore SGB4LinearArray;
class SGB4LinearArray : public SearchGridBuilder
{
 public:
  void setDistanceBtwMicrophones( float distance );
  void setPositionsOfMicrophones( const gsl_matrix* mpos );
  virtual bool nextSearchGrid();
  virtual const gsl_vector *getTimeDelays();
};

class SGB4LinearArrayPtr : public SearchGridBuilderPtr
{
 public:
  %extend {
    SGB4LinearArrayPtr( int nChan, bool isFarField, unsigned int samplingFreq=16000 ){
      return new SGB4LinearArrayPtr( new SGB4LinearArray( nChan, isFarField, samplingFreq ));
    }
  }

  SGB4LinearArray* operator->();
};

%ignore SGB4CircularArray;
class SGB4CircularArray : public SearchGridBuilder
{
 public:
  void setRadius( float radius, float height=0.0 );
  virtual bool nextSearchGrid();
  virtual const gsl_vector *getTimeDelays();
};

class SGB4CircularArrayPtr : public SearchGridBuilderPtr
{
 public:
  %extend {
    SGB4CircularArrayPtr( int nChan, bool isFarField, unsigned int samplingFreq=16000 ){
      return new SGB4CircularArrayPtr( new SGB4CircularArray( nChan, isFarField, samplingFreq ));
    }
  }

  SGB4CircularArray* operator->();
};


%ignore MCCLocalizer;
class MCCLocalizer : public VectorFeatureStream {
 public:
  MCCLocalizer( SearchGridBuilderPtr &sgbPtr, size_t maxSource=1, const String& nm= "MCCSourceLocalizer" );
  ~MCCLocalizer();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();
  //void setChannel(SampleFeaturePtr& chan);
  void setChannel(VectorFloatFeatureStreamPtr& chan);
  int getDelayedSample( int chanX );
  double getMaxMCCC();
  const gsl_vector* getPosition();
  int getNthBestDelayedSample( int nth, int chanX );
  double getNthBestMCCC( int nth );
  const gsl_vector* getNthBestPosition( int nth );
  const gsl_vector* getEigenValues();
  gsl_matrix *getR();
};

class MCCLocalizerPtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    MCCLocalizerPtr( SearchGridBuilderPtr &sgbPtr, size_t maxSource=1, const String& nm= "MCCSourceLocalizer" ) {
      return new MCCLocalizerPtr(new MCCLocalizer( sgbPtr, maxSource, nm ));
    }

    MCCLocalizerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MCCLocalizer* operator->();
};

%ignore MCCCalculator;
class MCCCalculator : public MCCLocalizer {
 public:
  MCCCalculator( SearchGridBuilderPtr &sgbPtr, bool normalizeVariance=true, const String& nm= "MCCCalculator" );
  ~MCCCalculator();
  void setTimeDelays( gsl_vector *delays );
  double getMCCC();
  double getCostV();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();
};

class MCCCalculatorPtr : public MCCLocalizerPtr {
 public:
  %extend {
    MCCCalculatorPtr( SearchGridBuilderPtr &sgbPtr, bool normalizeVariance=true, const String& nm= "MCCCalculator" ) {
      return new MCCCalculatorPtr(new MCCCalculator( sgbPtr, normalizeVariance, nm ));
    }

    MCCCalculatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MCCCalculator* operator->();
};

%ignore RMCCLocalizer;
class RMCCLocalizer : public MCCLocalizer {
 public:
  RMCCLocalizer( SearchGridBuilderPtr &sgbPtr, float lambda, size_t maxSource=1, const String& nm= "RMCCSourceLocalizer" );
   ~RMCCLocalizer();
   virtual const gsl_vector* next(int frameX = -5);
   virtual void  reset();
};

class RMCCLocalizerPtr : public MCCLocalizerPtr {
 public:
  %extend {
    RMCCLocalizerPtr( SearchGridBuilderPtr &sgbPtr, float lambda, size_t maxSource=1, const String& nm= "RMCCSourceLocalizer" ) {
      return new RMCCLocalizerPtr(new RMCCLocalizer( sgbPtr, lambda, maxSource, nm ));
    }

    RMCCLocalizerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  RMCCLocalizer* operator->();
};


