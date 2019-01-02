/**
 * @file tde.h
 * @brief Time delay estimation
 * @author Kenichi Kumatani
 */

#ifndef TDE_H
#define TDE_H

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
#include "btk.h"

// ----- definition for class `CCTDE' -----
// 
/**
   @class find the time difference which provides the maximum correlation between tow signals.
   @usage
   1. constrct sample feature objects :
       samp1 = 
       samp2 =
   2. construct this object and feed the sample features into it:
*/
class CCTDE : public VectorFeatureStream {
public:
  /**
     @brief 
     @param 
     @param 
     @param 
     @param 
   */
  CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, int fftLen=512, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" );
  ~CCTDE();

  virtual const gsl_vector* next(int frame_no = -5 );
  virtual const gsl_vector* nextX( unsigned chanX = 0, int frame_no = -5);
  virtual void  reset();
  void  allsamples( int fftLen = -1 );
  void set_target_frequency_range( int freqLowerLimit, int freqUpperLimit ){
    freq_lower_limit_ = freqLowerLimit;
    freq_upper_limit_ = freqUpperLimit;
  }
  const unsigned *sample_delays() const { return sample_delays_; }
  const double *cc_values() const {return cc_values_; }

#ifdef ENABLE_LEGACY_BTK_API
  void setTargetFrequencyRange( int freqLowerLimit, int freqUpperLimit ){ set_target_frequency_range(freqLowerLimit, freqUpperLimit); }
  const unsigned *getSampleDelays(){ return sample_delays(); }
  const double *getCCValues(){ return cc_values(); }
#endif

private:
  const gsl_vector* detect_cc_peaks_( double **samples, size_t stride );

  typedef list<SampleFeaturePtr>	ChannelList_;
  typedef ChannelList_::iterator	ChannelIterator_;
  ChannelList_	channelL_;   // must be 2.
  unsigned	*sample_delays_; // sample delays
  double	*cc_values_;     // cross-correlation (CC) values
  unsigned	nHeldMaxCC_;    // how many CC values are held
  unsigned	fftLen_;
  int		samplerate_;
  gsl_vector	*window_;
  int		freq_lower_limit_;
  int		freq_upper_limit_;
  vector<unsigned > _frameCounter;
};

typedef Inherit<CCTDE, VectorFeatureStreamPtr> CCTDEPtr;

#endif

