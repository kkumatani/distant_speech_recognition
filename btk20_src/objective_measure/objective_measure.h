/**
 * @file objective_measure.h
 * @brief objective measure for speech quality or distance
 * @author Kenichi Kumatani
 */

#ifndef OBJECTIVE_MEASURE_H
#define OBJECTIVE_MEASURE_H

#include <stdio.h>
#include <assert.h>
#include <float.h>


#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"
#include "modulated/modulated.h"

// ----- definition for class `SNR' -----
//

 /**
     @class calculate the signal-to-ratio (SNR) of two speech data
     @usage
   */
class SNR{
 public:
  //SNR();
  //SNR( const String& nm = "SNR" );
  float getSNR( const String& fn1, const String& fn2, int normalizationOption, int chX=1, int samplerate=16000, int cfrom=-1, int to=-1 );
  float getSNR2( gsl_vector_float *original, gsl_vector_float *enhanced, int normalizationOption);
};

typedef refcount_ptr<SNR> SNRPtr;

class segmentalSNR{
 public:
};

class ItakuraSaitoMeasurePS {
 public:
 ItakuraSaitoMeasurePS( unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, const String& nm = "ItakuraSaitoMeasurePS" ):_fftLen(fftLen),_r(r),_windowType(windowType)
  {
    _D = _fftLen / (int)pow( (float)2, (int)_r );
  }

  float getDistance( const String& fn1, const String& fn2, int chX=1, int samplerate=16000, int bframe = 0, int eframe = -1 );
  //float getDistance( VectorFloatFeatureStreamPtr& original,
  //VectorFloatFeatureStreamPtr& enhanced );

  int frameShiftLength(){ 
    return _D; 
  }

 private:
  unsigned _fftLen;
  unsigned _r;
  unsigned _windowType;
  unsigned _D;
};

typedef refcount_ptr<ItakuraSaitoMeasurePS> ItakuraSaitoMeasurePSPtr;


#endif 
