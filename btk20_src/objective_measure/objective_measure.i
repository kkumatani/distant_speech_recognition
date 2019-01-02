/**
 * @file objective_measure.i
 * @brief calculate objective measures for speech enhancement
 * @author Kenichi Kumatani
 */

%module(package="btk20") objective_measure
%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include "objective_measure/objective_measure.h"
#include <numpy/arrayobject.h>
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

//%import objectiveMeasure.h

// ----- definition for class `SNR' -----
//
%ignore SNR;
class SNR {
 public:
  float getSNR( const String& fn1, const String& fn2, int normalizationOption, int chX=1, int samplerate=16000, int cfrom=-1, int to=-1 );
  float getSNR2( gsl_vector_float *original, gsl_vector_float *enhanced, int normalizationOption);
};

class SNRPtr {
 public:
  %extend {
    SNRPtr() {
      return new SNRPtr(new SNR );
    }
  }

  SNR* operator->();
};

// ----- definition for class `ItakuraSaitoMeasurePS' -----
//
%ignore ItakuraSaitoMeasurePS;
class ItakuraSaitoMeasurePS {
 public:
  ItakuraSaitoMeasurePS( unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, const String& nm = "ItakuraSaitoMeasurePS" );
  float getDistance( const String& fn1, const String& fn2, int chX=1, int samplerate=16000, int bframe =0, int eframe = -1 );
  int frameShiftLength();
};

class ItakuraSaitoMeasurePSPtr {
 public:
  %extend {
    ItakuraSaitoMeasurePSPtr( unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, const String& nm = "ItakuraSaitoMeasurePS" ) {
      return new ItakuraSaitoMeasurePSPtr(new ItakuraSaitoMeasurePS(fftLen,  r, windowType, nm ) );
    }
  }

  ItakuraSaitoMeasurePS* operator->();
};
