/**
 * @file tde.i
 * @brief Time delay estimation
 * @author Kenichi Kumatani
 */

%module(package="btk20") tde

%{
#include "tde/tde.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
  //#include "modulated/prototypeDesign.h"
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

// ----- definition for class `CCTDE' -----
// 
%ignore CCTDE;
class CCTDE : public VectorFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") allsamples;
  %feature("kwargs") set_target_frequency_range;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setTargetFrequencyRange;
#endif
 public:
  CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" );
  ~CCTDE();

  virtual const gsl_vector* next(int frame_no = -5);
  virtual void  reset();
  void  allsamples( int fftLen=-1 );
  void set_target_frequency_range( int freqLowerLimit, int freqUpperLimit );
  const unsigned *sample_delays() const;
  const double *cc_values() const;

#ifdef ENABLE_LEGACY_BTK_API
  void setTargetFrequencyRange( int freqLowerLimit, int freqUpperLimit );
  const unsigned *getSampleDelays();
  const double *getCCValues();
#endif
};


class CCTDEPtr : public VectorFeatureStreamPtr {
  %feature("kwargs") CCTDEPtr;
 public:
  %extend {
    CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, bool isRTProcessing=false, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" ){
      return new CCTDEPtr(new CCTDE(SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, bool isRTProcessing, unsigned nHeldMaxCC, int freqLowerLimit, int freqUpperLimit, const String& nm));
    }

    CCTDEPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CCTDE* operator->();
};

