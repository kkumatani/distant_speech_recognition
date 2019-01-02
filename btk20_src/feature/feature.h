/**
 * @file feature.h
 * @brief Speech recognition front end.
 * @author John McDonough, Tobias Gehrig, Kenichi Kumatani, Friedrich Faubel
 */

#ifndef FEATURE_H
#define FEATURE_H

#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector_complex.h>
#include "matrix/gslmatrix.h"
#include "stream/stream.h"
#include "common/mlist.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>

#include <pthread.h>

#include "btk.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif /* #ifdef HAVE_LIBFFTW3 */

#include "feature/spectralestimator.h"

/* sflib and sndfile both define sf_perror so we put them into seperate namespaces */
namespace sndfile {
#include <sndfile.h>
}

void unpack_half_complex(gsl_vector_complex* tgt, const double* src);
void unpack_half_complex(gsl_vector_complex* tgt, const unsigned N2, const double* src, const unsigned N);
void pack_half_complex(double* tgt, const gsl_vector_complex* src, unsigned size = 0);


/**
* \defgroup AudioFeature Audio Feature Hierarchy
* This hierarchy of classes provides the capability for extracting audio
* features for use in automatic speech recognition.
*/
/*@{*/

/**
* \defgroup FileFeature File Feature.
*/
/*@{*/

// ----- definition for class `FileFeature' -----
//
class FileFeature : public VectorFloatFeatureStream {
 public:
  FileFeature(unsigned sz, const String& nm = "FileFeature") :
    VectorFloatFeatureStream(sz, nm), feature_(NULL) {}

  virtual ~FileFeature() {
    if (feature_ != NULL)
      gsl_matrix_float_free(feature_);
  }

  FileFeature& operator=(const FileFeature& f);

  virtual const gsl_vector_float* next(int frame_no = -5);

  unsigned size() const {
    if (GSL_MATRIX_NCOLS(feature_) == 0)
      throw j_error("Matrix not loaded yet.");
    return (unsigned) GSL_MATRIX_NCOLS(feature_);
  }

  void bload(const String& fileName, bool old = false);

  void copy(gsl_matrix_float* matrix);

 private:
  gsl_matrix_float*  feature_;
};

typedef Inherit<FileFeature, VectorFloatFeatureStreamPtr> FileFeaturePtr;

/*@}*/

/**
* \defgroup ConversionbitToShort Conversion bit Short
*/
/*@{*/

// ----- definition of 'Conversion24bit2Short' -----
//
class Conversion24bit2Short : public VectorShortFeatureStream {
 public:
 Conversion24bit2Short(VectorCharFeatureStreamPtr& src,
                       const String& nm = "Conversion24bit2Short") :
  VectorShortFeatureStream(src->size()/3, nm), src_(src) {};
  virtual void reset() { src_->reset(); VectorShortFeatureStream::reset(); }

  virtual const gsl_vector_short* next(int frame_no = -5);
 private:
  VectorCharFeatureStreamPtr src_;
};

typedef Inherit<Conversion24bit2Short, VectorShortFeatureStreamPtr> Conversion24bit2ShortPtr;

/*@}*/

/**
* \defgroup Conversion24bit2Float Conversion 24 bit 2 Float
*/
/*@{*/

// ----- definition of Conversion24bit2Float -----
//
class Conversion24bit2Float : public VectorFloatFeatureStream {
 public:
  Conversion24bit2Float(VectorCharFeatureStreamPtr& src,
                        const String& nm = "Conversion24bit2Float") :
    VectorFloatFeatureStream(src->size()/3, nm), src_(src) {};
  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

  virtual const gsl_vector_float* next(int frame_no = -5);
 private:
  VectorCharFeatureStreamPtr src_;
};

typedef Inherit<Conversion24bit2Float, VectorFloatFeatureStreamPtr> Conversion24bit2FloatPtr;


/**
* \defgroup SampleFeature Sample Feature
*/
/*@{*/

// ----- definition for class `SampleFeature' -----
//
class SampleFeature;
typedef Inherit<SampleFeature, VectorFloatFeatureStreamPtr> SampleFeaturePtr;
class SampleFeature : public VectorFloatFeatureStream {
 public:
  SampleFeature(const String& fn = "", unsigned blockLen = 320,
                unsigned shiftLen = 160, bool padZeros = false, const String& nm = "Sample");
  virtual ~SampleFeature();

  unsigned read(const String& fn, int format = 0, int samplerate = 16000,
                int chX = 1, int chN = 1, int cfrom = 0, int to = -1, int outsamplerate = -1, float norm = 0.0);

  void write(const String& fn, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_16, int sampleRate = -1);

  void cut(unsigned cfrom, unsigned cto);

  void randomize(int startX, int endX, double sigma2);

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { cur_ = 0; VectorFloatFeatureStream::reset(); is_end_ = false; }

  void exit(){ reset(); throw jiterator_error("end of samples!");}

  const gsl_vector_float* data();

  const gsl_vector* dataDouble();

  void copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to);

  unsigned samplesN() const { return ttlsamples_; }

  int getSampleRate() const { return samplerate_; }

  int getChanN() const { return nChan_; }

  void zeroMean();

  void addWhiteNoise( float snr );

  void setSamples(const gsl_vector* samples, unsigned sampleRate);

protected:
  float*              samples_;
  float               norm_;
  unsigned            ttlsamples_;
  const unsigned      shiftLen_;
  int                 samplerate_;
  int                 nChan_;
  int                 format_;
  unsigned            cur_;
  bool                pad_zeros_;
  gsl_vector_float*   copy_fsamples_;
  gsl_vector*         copy_dsamples_;

private:
  SampleFeature(const SampleFeature& s);
  SampleFeature& operator=(const SampleFeature& s);
};


// ----- definition for class `SampleFeatureRunon' -----
//
class SampleFeatureRunon : public SampleFeature {
 public:
  SampleFeatureRunon(const String& fn = "", unsigned blockLen = 320,
                     unsigned shiftLen = 160, bool padZeros = false, const String& nm = "Sample") :
    SampleFeature(fn, blockLen, shiftLen, padZeros, nm) { }

  virtual void reset() { VectorFloatFeatureStream::reset(); }

  virtual int frame_no() const { return (cur_ / shiftLen_) - 1; }

  virtual int frameN() const { return (ttlsamples_ / shiftLen_) - 1; }
};

typedef Inherit<SampleFeatureRunon, SampleFeaturePtr> SampleFeatureRunonPtr;

/*@}*/

/**
* \defgroup IterativeSampleFeature Iterative Sample Feature for the single channel data
*/
/*@{*/

// ----- definition for class `IterativeSingleChannelSampleFeature' -----
//
class IterativeSingleChannelSampleFeature : public VectorFloatFeatureStream {
 public:
 public:
  IterativeSingleChannelSampleFeature(unsigned blockLen = 320, const String& nm = "IterativeSingleChannelSampleFeature");
  virtual ~IterativeSingleChannelSampleFeature();

  void read(const String& fileName, int format = 0, int samplerate = 44100, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const { return ttlsamples_; }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:

  float*				samples_;
  sndfile::SNDFILE*			sndfile_;
  sndfile::SF_INFO			sfinfo_;
  unsigned				interval_;
  unsigned				blockN_;
  unsigned				sampleN_;
  unsigned				ttlsamples_;

  const unsigned			blockLen_;
  unsigned				cur_;
  bool					last_;
  int					cto_;
};

typedef Inherit<IterativeSingleChannelSampleFeature, VectorFloatFeatureStreamPtr> IterativeSingleChannelSampleFeaturePtr;

/*@}*/

/**
* \defgroup IterativeSampleFeature Iterative Sample Feature
*/
/*@{*/

// ----- definition for class `IterativeSampleFeature' -----
//
class IterativeSampleFeature : public VectorFloatFeatureStream {
 public:
  IterativeSampleFeature(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample");
  virtual ~IterativeSampleFeature();

  void read(const String& fileName, int format = 0, int samplerate = 44100, int chN = 1, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const { return ttlsamples_; }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  void changeFirstChannelID( unsigned firstChanX ){ firstChanX_ = firstChanX; }

 private:
  IterativeSampleFeature(const IterativeSampleFeature& s);
  IterativeSampleFeature& operator=(const IterativeSampleFeature& s);

  static float*					allSamples_;
  static sndfile::SNDFILE*			sndfile_;
  static sndfile::SF_INFO			sfinfo_;
  static unsigned				interval_;
  static unsigned				blockN_;
  static unsigned				sampleN_;
  static unsigned				allSampleN_;
  static unsigned				ttlsamples_;

  const unsigned				blockLen_;
  const unsigned				chanX_;
  unsigned				        firstChanX_;
  unsigned					cur_;
  bool						last_;
  int						cto_;
};

typedef Inherit<IterativeSampleFeature, VectorFloatFeatureStreamPtr> IterativeSampleFeaturePtr;

/*@}*/

/**
* \defgroup BlockSizeConversionFeature Block Size Conversion Feature
*/
/*@{*/

// ----- definition for class `BlockSizeConversionFeature' -----
//
class BlockSizeConversionFeature : public VectorFloatFeatureStream {
 public:
  BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
                             unsigned blockLen = 320,
                             unsigned shiftLen = 160,
                             const String& nm = "BlockSizeConversionFeature");

  virtual void reset() { curin_ = curout_ = 0; src_frame_no_ = -1; src_->reset();  VectorFloatFeatureStream::reset(); }

  virtual const gsl_vector_float* next(int frame_no = -5);

 private:
  void inputLonger_();
  void outputLonger_();

  VectorFloatFeatureStreamPtr			src_;
  const unsigned				inputLen_;
  const unsigned				blockLen_;
  const unsigned				shiftLen_;
  const unsigned				overlapLen_;
  unsigned					curin_;
  unsigned					curout_;
  int                                           src_frame_no_;
  const gsl_vector_float*			src_feat_;
};

typedef Inherit<BlockSizeConversionFeature, VectorFloatFeatureStreamPtr> BlockSizeConversionFeaturePtr;

/*@}*/

/**
* \defgroup BlockSizeConversionFeatureShort Block Size Conversion Feature Short
*/
/*@{*/

// ----- definition for class `BlockSizeConversionFeatureShort' -----
//
class BlockSizeConversionFeatureShort : public VectorShortFeatureStream {
 public:
  BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
				  unsigned blockLen = 320,
				  unsigned shiftLen = 160, const String& nm = "Block Size Conversion");

  virtual void reset() { curin_ = curout_ = 0;  src_->reset();  VectorShortFeatureStream::reset(); }

  virtual const gsl_vector_short* next(int frame_no = -5);

 private:
  void inputLonger_();
  void outputLonger_();

  VectorShortFeatureStreamPtr			src_;
  const unsigned				inputLen_;
  const unsigned				blockLen_;
  const unsigned				shiftLen_;
  const unsigned				overlapLen_;
  unsigned					curin_;
  unsigned					curout_;
  const gsl_vector_short*			src_feat_;
};

typedef Inherit<BlockSizeConversionFeatureShort, VectorShortFeatureStreamPtr> BlockSizeConversionFeatureShortPtr;

/*@}*/


#ifdef SMARTFLOW

namespace sflib {
#include "sflib.h"
}

/**
* \defgroup SmartFlowFeature Smart Flow Feature
*/
/*@{*/

// ----- definition for class `SmartFlowFeature' -----
//
class SmartFlowFeature : public VectorShortFeatureStream {
 public:
  SmartFlowFeature(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
                   unsigned shiftLen = 160, const String& nm = "SmartFloatFeature") :
    VectorShortFeatureStream(blockLen, nm),
    sfflow_(sfflow), blockLen_(blockLen), shiftLen_(shiftLen) { }

  virtual ~SmartFlowFeature() { }

  virtual const gsl_vector_short* next(int frame_no = -5);

  virtual void reset() { VectorShortFeatureStream::reset(); }

 private:
  sflib::sf_flow_sync* 				sfflow_;
  const unsigned				blockLen_;
  const unsigned				shiftLen_;
};

typedef Inherit<SmartFlowFeature, VectorShortFeatureStreamPtr> SmartFlowFeaturePtr;

#endif

/*@}*/

/**
* \defgroup PreemphasisFeature Preemphasis Feature
*/
/*@{*/

// ----- definition for class `PreemphasisFeature' -----
//
class PreemphasisFeature : public VectorFloatFeatureStream {
 public:
 PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis");
  virtual ~PreemphasisFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker(){ next_speaker(); }
#endif

 private:
  VectorFloatFeatureStreamPtr			samp_;
  float						prior_;
  const double					mu_;
};

typedef Inherit<PreemphasisFeature, VectorFloatFeatureStreamPtr> PreemphasisFeaturePtr;

/*@}*/

/**
* \defgroup HammingFeatureShort Hamming Feature Short
*/
/*@{*/

// ----- definition for class `HammingFeatureShort' -----
//
class HammingFeatureShort : public VectorFloatFeatureStream {
 public:
  HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm = "HammingShort");
  virtual ~HammingFeatureShort() { delete[] window_; }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorShortFeatureStreamPtr			samp_;
  unsigned					windowLen_;
  double*					window_;
};

typedef Inherit<HammingFeatureShort, VectorFloatFeatureStreamPtr> HammingFeatureShortPtr;

/*@}*/

/**
* \defgroup HammingFeature Hamming Feature
*/
/*@{*/

// ----- definition for class `HammingFeature' -----
//
class HammingFeature : public VectorFloatFeatureStream {
 public:
  HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming");
  virtual ~HammingFeature() { delete[] window_; }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			samp_;
  unsigned					windowLen_;
  double*					window_;
};

typedef Inherit<HammingFeature, VectorFloatFeatureStreamPtr> HammingFeaturePtr;

/*@}*/

/**
* \defgroup FFTFeature FFT Feature
*/
/*@{*/

// ----- definition for class `FFTFeature' -----
//
class FFTFeature : public VectorComplexFeatureStream {
 public:
  FFTFeature(const VectorFloatFeatureStreamPtr& samp, unsigned fftLen, const String& nm = "FFT");
  virtual ~FFTFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset() { samp_->reset(); VectorComplexFeatureStream::reset(); }

  unsigned fftLen()    const { return fftLen_;    }
  unsigned windowLen() const { return windowLen_; }

  unsigned nBlocks()     const { return 4; }
  unsigned subsamplerate() const { return 2; }

#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() { return subsamplerate(); }
#endif

 private:
  VectorFloatFeatureStreamPtr			samp_;
  unsigned					fftLen_;
  unsigned					windowLen_;
  double*					samples_;

#ifdef HAVE_LIBFFTW3
  fftw_plan					fftwPlan_;
  fftw_complex*					output_;
#endif
};

typedef Inherit<FFTFeature, VectorComplexFeatureStreamPtr> FFTFeaturePtr;

/**
* \defgroup SpectralPowerFeature Spectral Power Feature
*/
/*@{*/

// ----- definition for class `SpectralPowerFeature' -----
//
class SpectralPowerFloatFeature : public VectorFloatFeatureStream {
 public:
  SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String& nm = "PowerFloat");

  virtual ~SpectralPowerFloatFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { fft_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorComplexFeatureStreamPtr    fft_;
};

typedef Inherit<SpectralPowerFloatFeature, VectorFloatFeatureStreamPtr> SpectralPowerFloatFeaturePtr;

/*@}*/


/**
* \defgroup SpectralPowerFeature Spectral Power Feature
*/
/*@{*/

// ----- definition for class `SpectralPowerFeature' -----
//
class SpectralPowerFeature : public VectorFeatureStream {
 public:
  SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String& nm = "Power");

  virtual ~SpectralPowerFeature() { }

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { fft_->reset(); VectorFeatureStream::reset(); }

 private:
  VectorComplexFeatureStreamPtr			fft_;
};

typedef Inherit<SpectralPowerFeature, VectorFeatureStreamPtr> SpectralPowerFeaturePtr;

/*@}*/

/**
* \defgroup SignalPowerFeature Signal Power Feature
*/
/*@{*/

// ----- definition for class `SignalPowerFeature' -----
//
static const int ADCRANGE = 65536;

class SignalPowerFeature : public VectorFloatFeatureStream {
 public:
  SignalPowerFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Signal Power") :
    VectorFloatFeatureStream(/* size= */ 1, nm), samp_(samp), range_(float(ADCRANGE) * float(ADCRANGE) / 4.0) { }

  virtual ~SignalPowerFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			samp_;
  double					range_;
};

typedef Inherit<SignalPowerFeature, VectorFloatFeatureStreamPtr> SignalPowerFeaturePtr;

/*@}*/

/**
* \defgroup ALogFeature A-Log Feature
*/
/*@{*/

// ----- definition for class `ALogFeature' -----
//
class ALogFeature : public VectorFloatFeatureStream {
 public:
  ALogFeature(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
              bool runon = false, const String& nm = "ALogPower") :
    VectorFloatFeatureStream(/* size= */ 1, nm), samp_(samp), m_(m), a_(a),
      min_(HUGE), max_(-HUGE), minMaxFound_(false), runon_(runon) { }

  virtual ~ALogFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  void next_speaker() { min_ = HUGE; max_ = -HUGE; minMaxFound_ = false; }

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker(){ next_speaker(); }
#endif

 private:
  void find_min_max_(const gsl_vector_float* block);

  VectorFloatFeatureStreamPtr			samp_;
  double					m_;
  double					a_;
  double					min_;
  double					max_;
  bool						minMaxFound_;
  bool						runon_;
};

typedef Inherit<ALogFeature, VectorFloatFeatureStreamPtr> ALogFeaturePtr;

/*@}*/

/**
* \defgroup NormalizeFeature Normalize Feature
*/
/*@{*/

// ----- definition for class `NormalizeFeature' -----
//
class NormalizeFeature : public VectorFloatFeatureStream {
 public:
  NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
                   bool runon = false, const String& nm = "Normalize");

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  void next_speaker() { xmin_ = HUGE; xmax_ = -HUGE; minMaxFound_ = false; }

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker(){ next_speaker(); }
#endif
 private:
  void find_min_max_(const gsl_vector_float* block);

  VectorFloatFeatureStreamPtr			samp_;
  double					min_;
  double					max_;
  double					range_;

  double					xmin_;
  double					xmax_;
  bool						minMaxFound_;
  bool						runon_;
};

typedef Inherit<NormalizeFeature, VectorFloatFeatureStreamPtr> NormalizeFeaturePtr;

/*@}*/

/**
* \defgroup ThresholdFeature Threshold Feature
*/
/*@{*/

// ----- definition for class `ThresholdFeature' -----
//
class ThresholdFeature : public VectorFloatFeatureStream {
 public:
  ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
		   const String& mode = "upper", const String& nm = "Threshold");

  virtual ~ThresholdFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			samp_;
  double					value_;
  double					thresh_;
  int						compare_;
};

typedef Inherit<ThresholdFeature, VectorFloatFeatureStreamPtr> ThresholdFeaturePtr;

/*@}*/

/**
* \defgroup SpectralResamplingFeature Spectral Resampling Feature
*/
/*@{*/

// ----- definition for class `SpectralResamplingFeature' -----
//
class SpectralResamplingFeature : public VectorFeatureStream {
  static const double SampleRatio;
 public:
  SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio = SampleRatio, unsigned len = 0,
			    const String& nm = "Resampling");

  virtual ~SpectralResamplingFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFeatureStreamPtr			src_;
  const double					ratio_;
};

typedef Inherit<SpectralResamplingFeature, VectorFeatureStreamPtr> SpectralResamplingFeaturePtr;

/*@}*/

/**
* \defgroup SamplerateConversionFeature Samplerate Conversion Feature
*/
/*@{*/

// ----- definition for class `SamplerateConversionFeature' -----
//
#ifdef SRCONV

#include <samplerate.h>

class SamplerateConversionFeature : public VectorFloatFeatureStream {
 public:
  // ratio : Equal to input_sample_rate / output_sample_rate.
  SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate = 22050, unsigned destrate = 16000,
			      unsigned len = 0, const String& method = "fastest", const String& nm = "SamplerateConversion");
  virtual ~SamplerateConversionFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			src_;
  SRCSTATE_*                                    state_;
  SRC_DATA                                      data_;
  int                                           error_;

  unsigned					dataInSamplesN_;
  unsigned					dataOutStartX_;
  unsigned					dataOutSamplesN_;
};

typedef Inherit<SamplerateConversionFeature, VectorFloatFeatureStreamPtr> SamplerateConversionFeaturePtr;

#endif

/*@}*/

/**
* \defgroup VTLNFeature VTLN Feature
*/
/*@{*/

// ----- definition for class `VTLNFeature' -----
//
// -------------------------------------------
// Piecewise linear: Y = X/Ratio  for X < edge
// -------------------------------------------
class VTLNFeature : public VectorFeatureStream {
 public:
  VTLNFeature(const VectorFeatureStreamPtr& pow,
	      unsigned coeffN = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
	      const String& nm = "VTLN");
  virtual ~VTLNFeature() {  }

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { pow_->reset(); VectorFeatureStream::reset(); }

  // specify the warp factor
  void warp(double w) { ratio_ = w; }

  void matrix(gsl_matrix* mat) const;

private:
  virtual const gsl_vector* nextFF(int frame_no);
  virtual const gsl_vector* nextOrg(int frame_no);

 private:
  VectorFeatureStreamPtr pow_;
  double                 ratio_;
  const double           edge_;
  const int              version_;
  gsl_vector*            auxV_;
};

typedef Inherit<VTLNFeature, VectorFeatureStreamPtr> VTLNFeaturePtr;

/*@}*/

/**
* \defgroup MelFeature Mel Feature
*/
/*@{*/

// ----- definition for class 'MelFeature' -----
//
class MelFeature : public VectorFeatureStream {
  class SparseMatrix_ {
  public:
    SparseMatrix_(unsigned m, unsigned n, unsigned version);
    ~SparseMatrix_();

    void melScale(int powN,  float rate, float low, float up, int filterN);
    void melScaleOrg(int powN,  float rate, float low, float up, int filterN);
    void melScaleFF(int powN,  float rate, float low, float up, int filterN);
    gsl_vector* fmatrixBMulotOrg( gsl_vector* C, const gsl_vector* A) const;
    gsl_vector* fmatrixBMulotFF( gsl_vector* C, const gsl_vector* A) const;
    void fmatrixBMulot( gsl_vector* C, const gsl_vector* A) const;
    void readBuffer(const String& fb);

    void matrix(gsl_matrix* mat) const;

  private:
    void alloc_(unsigned m, unsigned n);
    void dealloc_();

    float mel_(float hz);
    float hertz_(float m);

    float**					data_;
    unsigned					m_;
    unsigned					n_;
    unsigned*					offset_;// offset
    unsigned*					coefN_; // number of coefficients
    float					rate_;  // sampling rate in Hz
    int version_; // SparseMatrix_ version number, 1:Org, 2:Friedich's changes
  };

 public:
  MelFeature(const VectorFeatureStreamPtr& mag, int powN = 0,
             float rate = 16000.0, float low = 0.0, float up = 0.0,
             unsigned filterN = 30, unsigned version = 1, const String& nm = "MelFFT");

  virtual ~MelFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { mag_->reset(); VectorFeatureStream::reset(); }

  void read(const String& fileName);

  void matrix(gsl_matrix* mat) const { mel_.matrix(mat); }

 private:
  // gsl_vector* _fmatrixBMulot(gsl_vector* C, const gsl_vector* A, FBMatrix* B) const;

  const unsigned				nmel_;
  const unsigned				powN_;
  VectorFeatureStreamPtr			mag_;
  SparseMatrix_					mel_;
};

typedef Inherit<MelFeature, VectorFeatureStreamPtr> MelFeaturePtr;


// ----- definition for class 'SphinxMelFeature' -----
//
class SphinxMelFeature : public VectorFeatureStream {
  class Boundary_ {
  public:
    Boundary_(unsigned min_k, unsigned max_k)
      : min_k_(min_k), max_k_(max_k) { }

    unsigned					min_k_;
    unsigned					max_k_;
  };
  typedef vector<Boundary_>			Boundaries_;

 public:
  SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
                   float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
                   unsigned filterN = 30, const String& nm = "SphinxMelFilterBank");

  virtual ~SphinxMelFeature();

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { mag_->reset(); VectorFeatureStream::reset(); }

  void read(const String& fileName);

 private:
  static double melFrequency_(double frequency);
  static double melInverseFrequency_(double frequency);

  const unsigned				fftN_;
  const unsigned				filterN_;
  const unsigned				powN_;
  const double					samplerate_;
  VectorFeatureStreamPtr			mag_;
  gsl_matrix*					filters_;
  Boundaries_					boundaries_;
};

typedef Inherit<SphinxMelFeature, VectorFeatureStreamPtr> SphinxMelFeaturePtr;

/*@}*/

/**
* \defgroup LogFeature Log Feature
*/
/*@{*/

// ----- definition for class `LogFeature' -----
//
class LogFeature : public VectorFloatFeatureStream {
 public:
  LogFeature(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
             bool sphinxFlooring = false, const String& nm = "LogMel");
  virtual ~LogFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { mel_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  const unsigned                                nmel_;
  VectorFeatureStreamPtr                        mel_;
  const double                                  m_;
  const double                                  a_;
  const bool                                    SphinxFlooring_;
};

typedef Inherit<LogFeature, VectorFloatFeatureStreamPtr> LogFeaturePtr;

// ----- definition for class `FloatToDoubleConversionFeature' -----
//
class FloatToDoubleConversionFeature : public VectorFeatureStream {
 public:
  FloatToDoubleConversionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "FloatToDoubleConversion") : VectorFeatureStream(src->size(), nm), src_(src) {};

  virtual ~FloatToDoubleConversionFeature() { }

  virtual const gsl_vector* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr     src_;

};

typedef Inherit<FloatToDoubleConversionFeature, VectorFeatureStreamPtr> FloatToDoubleConversionFeaturePtr;

/*@}*/

/**
* \defgroup CepstralFeature Cepstral Feature
*/
/*@{*/

// ----- definition for class `CepstralFeature' -----
//
// type:
//   0 =
//   1 = Type 2 DCT
//   2 = Sphinx Legacy
class CepstralFeature : public VectorFloatFeatureStream {
 public:
  CepstralFeature(const VectorFloatFeatureStreamPtr& mel, unsigned ncep = 13,
                  int type = 1, const String& nm = "Cepstral");

  virtual ~CepstralFeature() { gsl_matrix_float_free(cos_); }

  virtual const gsl_vector_float* next(int frame_no = -5);

  gsl_matrix* matrix() const;

  virtual void reset() { mel_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  void sphinxLegacy_();

  gsl_matrix_float*				cos_;
  VectorFloatFeatureStreamPtr			mel_;
};

typedef Inherit<CepstralFeature, VectorFloatFeatureStreamPtr> CepstralFeaturePtr;

/*@}*/

/**
* \defgroup MeanSubtractionFeature Mean Subtraction Feature
*/
/*@{*/

// ----- definition for class `MeanSubtractionFeature' -----
//
class MeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL, double devNormFactor = 0.0, bool runon = false, const String& nm = "Mean Subtraction");

  virtual ~MeanSubtractionFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  const gsl_vector_float* mean() const { return mean_; }
  void write(const String& fileName, bool variance = false) const;
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif

 private:
  const gsl_vector_float* nextRunon_(int frame_no);
  const gsl_vector_float* nextBatch_(int frame_no);
  void calcMeanVariance_();
  void normalize_(const gsl_vector_float* srcVec);

  static const float				variance_floor_;
  static const float				before_wgt_;
  static const float				after_wgt;
  static const unsigned				framesN2change_;

  VectorFloatFeatureStreamPtr			src_;
  VectorFloatFeatureStreamPtr			wgt_;

  gsl_vector_float*				mean_;
  gsl_vector_float*				var_;
  const double					devNormFactor_;
  unsigned					framesN_;

  bool						runon_;
  bool						mean_var_found_;
};

typedef Inherit<MeanSubtractionFeature, VectorFloatFeatureStreamPtr> MeanSubtractionFeaturePtr;

/*@}*/

/**
* \defgroup FileMeanSubtractionFeature File Mean Subtraction Feature
*/
/*@{*/

// ----- definition for class `FileMeanSubtractionFeature' -----
//
class FileMeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src,
			     double devNormFactor = 0.0, const String& nm = "File Mean Subtraction");

  virtual ~FileMeanSubtractionFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  void read(const String& fileName, bool variance = false);

 private:
  static const float				variance_floor_;

  VectorFloatFeatureStreamPtr			src_;
  gsl_vector_float*				mean_;
  gsl_vector_float*				variance_;
  const double					devNormFactor_;
};

typedef Inherit<FileMeanSubtractionFeature, VectorFloatFeatureStreamPtr> FileMeanSubtractionFeaturePtr;

/*@}*/

/**
* \defgroup AdjacentFeature Adjacent Feature
*/
/*@{*/

// ----- definition for class `AdjacentFeature' -----
//
class AdjacentFeature : public VectorFloatFeatureStream {
 public:
  AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
		  const String& nm = "Adjacent");

  virtual ~AdjacentFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:
  void buffer_next_frame_(int frame_no);

  const unsigned				delta_;
  VectorFloatFeatureStreamPtr			single_;
  const unsigned				singleSize_;
  const unsigned				plen_;
  unsigned					framesPadded_;
};

typedef Inherit<AdjacentFeature, VectorFloatFeatureStreamPtr> AdjacentFeaturePtr;

/*@}*/

/**
* \defgroup LinearTransformFeature Linear Transform Feature
*/
/*@{*/

// ----- definition for class `LinearTransformFeature' -----
//
class LinearTransformFeature : public VectorFloatFeatureStream {
 public:
#if 0
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src,
			 gsl_matrix_float* mat = NULL, unsigned sz = 0, const String& nm = "Transform");
#else
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform");
#endif

  virtual ~LinearTransformFeature() { gsl_matrix_float_free(trans_); }

  virtual const gsl_vector_float* next(int frame_no = -5);

  gsl_matrix_float* matrix() const;

  void load(const String& fileName, bool old = false);

  void identity();

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 protected:
  VectorFloatFeatureStreamPtr			src_;
  gsl_matrix_float*				trans_;
};

typedef Inherit<LinearTransformFeature, VectorFloatFeatureStreamPtr> LinearTransformFeaturePtr;

/*@}*/

/**
* \defgroup StorageFeature Storage Feature
*/
/*@{*/

// ----- definition for class `StorageFeature' -----
//
class StorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  StorageFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Storage");

  virtual ~StorageFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

  void write(const String& fileName, bool plainText = false) const;
  void read(const String& fileName);
  int evaluate();

 private:
  VectorFloatFeatureStreamPtr			src_;
  _StorageVector				frames_;
};

typedef Inherit<StorageFeature, VectorFloatFeatureStreamPtr> StorageFeaturePtr;

/**
* \defgroup StaticStorageFeature Static Storage Feature
*/
/*@{*/

// ----- definition for class `StaticStorageFeature' -----
//
class StaticStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  StaticStorageFeature(unsigned dim, const String& nm = "Static Storage");

  virtual ~StaticStorageFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { VectorFloatFeatureStream::reset(); }

  //void write(const String& fileName) const;
  void read(const String& fileName);
  int evaluate();
  unsigned currentNFrames() const { return frame_no_; };

 private:
  //VectorFloatFeatureStreamPtr     src_;
  _StorageVector        frames_;
  int                  framesN_;
};

typedef Inherit<StaticStorageFeature, VectorFloatFeatureStreamPtr> StaticStorageFeaturePtr;

/*@}*/

/**
* \defgroup CircularStorageFeature Circular Storage Feature
*/
/*@{*/

// ----- definition for class `CircularStorageFeature' -----
//
class CircularStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN = 3, const String& nm = "CircularStorage");

  virtual ~CircularStorageFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:
  unsigned get_index_(int diff) const;

  VectorFloatFeatureStreamPtr			src_;
  const unsigned				framesN_;
  _StorageVector				frames_;
  unsigned					pointerX_;
};

typedef Inherit<CircularStorageFeature, VectorFloatFeatureStreamPtr> CircularStorageFeaturePtr;

/*@}*/

/**
* \defgroup FilterFeature Filter Feature
*/
/*@{*/

// ----- definition for class `FilterFeature' -----
//
class FilterFeature : public VectorFloatFeatureStream {
  class Buffer_ {
  public:
    Buffer_(unsigned len, unsigned nsamp)
      : len_(len), nsamp_(nsamp), offset_(int((nsamp_ - 1) / 2)),
      zero_(0), samples_(new gsl_vector_float*[nsamp_])
    {
      assert (nsamp_ % 2 == 1);
      for (unsigned i = 0; i < nsamp_; i++)
        samples_[i] = gsl_vector_float_calloc(len_);
    }
    ~Buffer_()
    {
      for (unsigned i = 0; i < nsamp_; i++)
        gsl_vector_float_free(samples_[i]);
      delete[] samples_;
    }

    const gsl_vector_float* sample(unsigned timeX) const {
      return samples_[index_(timeX)];
    }

    const double sample(int timeX, unsigned binX) const {
      unsigned idx = index_(timeX);
      const gsl_vector_float* vec = samples_[idx];
      return gsl_vector_float_get(vec, binX);
    }

    void nextSample(const gsl_vector_float* s = NULL) {
      zero_ = (zero_ + 1) % nsamp_;
      gsl_vector_float* nextBlock = samples_[(zero_ + offset_) % nsamp_];

      if (s == NULL) {
        gsl_vector_float_set_zero(nextBlock);
      } else {
        assert( s->size == len_ );
        gsl_vector_float_memcpy(nextBlock, s);
      }
    }

    void zero() {
      for (unsigned i = 0; i < nsamp_; i++)
        gsl_vector_float_set_zero(samples_[i]);
      zero_ = nsamp_ - offset_;
    }

    void print() const {
      for (int i = -offset_; i <= offset_; i++)
        printf("        %4d", i);
      printf("\n     --------------------------------------------------------------------------------\n");
      for (unsigned l = 0; l < len_; l++) {
        for (int i = -offset_; i <= offset_; i++)
          printf("  %10.4f", sample(i, l));
        printf("\n");
      }
    }

  private:
    unsigned index_(int idx) const {
      assert ( abs(idx) <= offset_);
      unsigned ret = (zero_ + nsamp_ + idx) % nsamp_;
      return ret;
    }

    const unsigned				len_;
    const unsigned				nsamp_;
    int						offset_;
    unsigned					zero_; // index of most recent sample
    gsl_vector_float**				samples_;
  };

 public:
  FilterFeature(const VectorFloatFeatureStreamPtr& src, gsl_vector* coeffA, const String& nm = "Filter");
  virtual ~FilterFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  void printBuffer() const { buffer_.print(); }

 private:
  void buffer_next_frame_(int frame_no);

  VectorFloatFeatureStreamPtr			src_;
  unsigned					lenA_;
  gsl_vector*					coeffA_;
  int						offset_;
  Buffer_					buffer_;
  unsigned					framesPadded_;
};

typedef Inherit<FilterFeature, VectorFloatFeatureStreamPtr> FilterFeaturePtr;

/*@}*/

/**
* \defgroup MergeFeature Merge Feature
*/
/*@{*/

// ----- definition for class `MergeFeature' -----
//
class MergeFeature : public VectorFloatFeatureStream {
  typedef list<VectorFloatFeatureStreamPtr>	FeatureList_;
  typedef FeatureList_::iterator		FeatureListIterator_;
  typedef FeatureList_::const_iterator		FeatureListConstIterator_;
 public:
  MergeFeature(VectorFloatFeatureStreamPtr& stat,
	       VectorFloatFeatureStreamPtr& delta,
	       VectorFloatFeatureStreamPtr& deltaDelta,
	       const String& nm = "Merge");

  virtual ~MergeFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

 private:
  FeatureList_					flist_;
};

typedef Inherit<MergeFeature, VectorFloatFeatureStreamPtr> MergeFeaturePtr;

/**
* \defgroup MultiModalFeature
*/
/*@{*/

// ----- definition for class `MultiModalFeature MergeFeature' -----
//
class MultiModalFeature : public VectorFloatFeatureStream {
  typedef list<VectorFloatFeatureStreamPtr>	FeatureList_;
  typedef FeatureList_::iterator		FeatureListIterator_;
  typedef FeatureList_::const_iterator		FeatureListConstIterator_;
 public:
  MultiModalFeature(unsigned nModality, unsigned totalVecSize, const String& nm = "Multi");

  ~MultiModalFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  void addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec=1 );

 private:
  unsigned     *samplePeriods_;  /* sample period (in nano sec.) */
  unsigned     minSamplePeriod_;
  unsigned     nModality_;
  unsigned     curr_vecsize_;
  FeatureList_ flist_;
};

typedef Inherit<MultiModalFeature, VectorFloatFeatureStreamPtr> MultiModalFeaturePtr;

/*@}*/

/**
* \defgroup FeatureSet Feature Set
*/
/*@{*/

// ----- definition for class `FeatureSet' -----
//
class FeatureSet {
  typedef List <VectorFloatFeatureStreamPtr>	List_;
 public:
  FeatureSet(const String& nm = "FeatureSet") :
    name_(nm), list_(nm) { }

  const String& name() const { return name_; }

  void add(VectorFloatFeatureStreamPtr& feat) { list_.add(feat->name(), feat); }
  VectorFloatFeatureStreamPtr& feature(const String& nm) { return list_[nm]; }

 private:
  const String		name_;
  List_			list_;
};

typedef refcount_ptr<FeatureSet>	FeatureSetPtr;

/*@}*/

void writeGSLMatrix(const String& fileName, const gsl_matrix* mat);


#ifdef JACK
#include <vector>
#include <jack/jack.h>
#include <jack/ringbuffer.h>

typedef struct {
  jack_port_t *port;
  jack_ringbuffer_t *buffer;
  unsigned buffersize;
  unsigned overrun;
  bool can_process;
} jack_channel_t;

/**
* \defgroup Jack Jack Object
*/
/*@{*/

class Jack {
 public:
  Jack(const String& nm);
  ~Jack();
  jack_channel_t* addPort(unsigned buffersize, const String& connection, const String& nm);
  void start(void) { can_capture = true; };
  unsigned getSampleRate() { return (unsigned)jack_get_sample_rate(client); }

 private:
  int process_callback (jack_nframes_t nframes);

  static int _process_callback(jack_nframes_t nframes, void *arg) {
    return static_cast<Jack *> (arg)->process_callback (nframes);
  }

  void shutdown_callback (void);
  static void _shutdown_callback(void *arg) {
    static_cast<Jack *> (arg)->shutdown_callback();
  }

  jack_client_t*				client;
  volatile bool					can_capture;
  volatile bool					can_process;
  vector<jack_channel_t*>			channel;
};

typedef refcount_ptr<Jack>			JackPtr;

/*@}*/

/**
* \defgroup JackFeature Jack Feature
*/
/*@{*/

// ----- definition for class `JackFeature' -----
//
class JackFeature;
typedef Inherit<JackFeature, VectorFloatFeatureStreamPtr> JackFeaturePtr;
class JackFeature : public VectorFloatFeatureStream {
 public:
  JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
              const String& connection, const String& nm);

  virtual ~JackFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { VectorFloatFeatureStream::reset(); }

 private:
  JackPtr					jack_;
  jack_channel_t*				channel;
};

#endif


/*@}*/

/**
* \defgroup ZeroCrossingRateHammingFeature Zero Crossing Rate Hamming Feature
*/
/*@{*/

// ----- definition for class `ZeroCrossingRateHammingFeature' -----
//
class ZeroCrossingRateHammingFeature : public VectorFloatFeatureStream {
 public:
  ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming");
  virtual ~ZeroCrossingRateHammingFeature() { delete[] window_; }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			samp_;
  unsigned					windowLen_;
  double*					window_;
};

typedef Inherit<ZeroCrossingRateHammingFeature, VectorFloatFeatureStreamPtr> ZeroCrossingRateHammingFeaturePtr;

/*@}*/

/**
* \defgroup YINPitchFeature YIN Pitch Feature
*/
/*@{*/

// ----- definition for class `YINPitchFeature' -----
//
class YINPitchFeature : public VectorFloatFeatureStream {
 public:
  YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch");
  virtual ~YINPitchFeature() { }

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { samp_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  float	_getPitch(const gsl_vector_float *input, gsl_vector_float *yin, float tol);
  VectorFloatFeatureStreamPtr			samp_;
  unsigned sr_;
  float	tr_;
};

typedef Inherit<YINPitchFeature, VectorFloatFeatureStreamPtr> YINPitchFeaturePtr;

/*@}*/

/**
* \defgroup SpikeFilter Spike Filter
*/
/*@{*/

// ----- definition for class `SpikeFilter' -----
//
class SpikeFilter : public VectorFloatFeatureStream {
 public:
  SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String& nm = "Spike Filter");
  virtual ~SpikeFilter();

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

private:
  VectorFloatFeatureStreamPtr		src_;
  const unsigned			adcN_;
  const unsigned			queueN_;
  float*				queue_;
  const unsigned			windowN_;
  float*				window_;
};

typedef Inherit<SpikeFilter, VectorFloatFeatureStreamPtr> SpikeFilterPtr;

/*@}*/

/**
* \defgroup SpikeFilter2 Spike Filter 2
*/
/*@{*/

// ----- definition for class `SpikeFilter2' -----
//
class SpikeFilter2 : public VectorFloatFeatureStream {
 public:
  SpikeFilter2(VectorFloatFeatureStreamPtr& src,
	       unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
	       const String& nm = "Spike Filter 2");

  virtual const gsl_vector_float* next(int frame_no = -5);

  virtual void reset();

  unsigned spikesN() const { return count_; }

private:
  VectorFloatFeatureStreamPtr		src_;
  const unsigned 			adcN_;
  const unsigned			width_;
  const float				maxslope_;
  const float				startslope_;
  const float				thresh_;
  float 				alpha_;
  float 				beta_;
  float					meanslope_;
  unsigned				count_;
  const unsigned			verbose_;
};

typedef Inherit<SpikeFilter2, VectorFloatFeatureStreamPtr> SpikeFilter2Ptr;

/*@}*/

namespace sndfile {
#include <sndfile.h>

/**
* \defgroup SoundFile Sound File
*/
/*@{*/

// ----- definition for class `SoundFile' -----
//
class SoundFile {
 public:
  SoundFile(const String& fn,
            int mode = sndfile::SFM_RDWR,
            int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_16,
            int samplerate = 16000,
            int channels = 1,
            bool normalize = false);
  ~SoundFile() { sf_close(sndfile_); }
  sf_count_t frames() const { return sfinfo_.frames; }
  int samplerate() const { return sfinfo_.samplerate; }
  int channels() const { return sfinfo_.channels; }
  int format() const { return sfinfo_.format; }
  int sections() const { return sfinfo_.sections; }
  int seekable() const { return sfinfo_.seekable; }
  sf_count_t readf(float *ptr, sf_count_t frames)
    { return sf_readf_float(sndfile_, ptr, frames); }
  sf_count_t writef(float *ptr, sf_count_t frames)
    { return sf_writef_float(sndfile_, ptr, frames); }
  sf_count_t read(float *ptr, sf_count_t items)
    { return sf_read_float(sndfile_, ptr, items); }
  sf_count_t write(float *ptr, sf_count_t items)
    { return sf_write_float(sndfile_, ptr, items); }
  sf_count_t seek(sf_count_t frames, int whence = SEEK_SET)
    { return sf_seek(sndfile_, frames, whence); }
 private:
  SNDFILE* sndfile_;
  SF_INFO sfinfo_;
};
}
typedef refcount_ptr<sndfile::SoundFile>	SoundFilePtr;

/*@}*/

/**
* \defgroup DirectSampleFeature Direct Sample Feature
*/
/*@{*/

// ----- definition for class `DirectSampleFeature' -----
//
class DirectSampleFeature;
typedef Inherit<DirectSampleFeature, VectorFloatFeatureStreamPtr> DirectSampleFeaturePtr;
class DirectSampleFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleFeature(const SoundFilePtr &sndfile,
                      unsigned blockLen = 320,
                      unsigned start = 0,
                      unsigned end = (unsigned)-1,
                      const String& nm = "DirectSample");
  virtual ~DirectSampleFeature() {}
  virtual const gsl_vector_float* next(int frame_no = -5);
  int sampleRate() const { return sndfile_->samplerate(); }
  int channels() const { return sndfile_->channels(); }
  void setRegion(unsigned start = 0, unsigned end = (unsigned)-1) {
    start_ = start;
    end_ = end;
  }
  virtual void reset() {
    sndfile_->seek(start_, SEEK_SET);
    cur_ = 0;
    VectorFloatFeatureStream::reset();
  }
 private:
  SoundFilePtr sndfile_;
  unsigned blockLen_;
  unsigned start_;
  unsigned end_;
  unsigned cur_;
};

/*@}*/

/**
* \defgroup DirectSampleOutputFeature Direct Sample Output Feature
*/
/*@{*/

// ----- definition for class `DirectSampleOutputFeature' -----
// 
class DirectSampleOutputFeature;
typedef Inherit<DirectSampleOutputFeature, VectorFloatFeatureStreamPtr> DirectSampleOutputFeaturePtr;
class DirectSampleOutputFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
			    const SoundFilePtr &sndfile,
			    const String& nm = "DirectSampleOutput");
  virtual ~DirectSampleOutputFeature() {}
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset() { src_->reset(); sndfile_->seek(0, SEEK_SET); VectorFloatFeatureStream::reset(); }
  
 private:
  VectorFloatFeatureStreamPtr src_;
  SoundFilePtr sndfile_;
  unsigned blockLen_;
};

/*@}*/

/**
* \defgroup ChannelExtractionFeature Channel Extraction Feature
*/
/*@{*/

// ----- definition for class `ChannelExtractionFeature' -----
//
class ChannelExtractionFeature;
typedef Inherit<ChannelExtractionFeature, VectorFloatFeatureStreamPtr> ChannelExtractionFeaturePtr;
class ChannelExtractionFeature : public VectorFloatFeatureStream {
 public:
  ChannelExtractionFeature(const VectorFloatFeatureStreamPtr& src,
                           unsigned chX = 0,
                           unsigned chN = 1,
                           const String& nm = "ChannelExtraction")
    : VectorFloatFeatureStream(src->size()/chN, nm), src_(src), chX_(chX), chN_(chN)
    {
      assert(chX < chN);
      assert((src->size() % chN) == 0);
    }
  virtual ~ChannelExtractionFeature() {}
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }
 private:
  VectorFloatFeatureStreamPtr src_;
  unsigned chX_;
  unsigned chN_;
};

/*@}*/

/**
* \defgroup SignalInterferenceFeature Signal Interference Feature
*/
/*@{*/

// ----- definition for class SignalInterferenceFeature -----
//
class SignalInterferenceFeature;
typedef Inherit<SignalInterferenceFeature, VectorFloatFeatureStreamPtr> SignalInterferenceFeaturePtr;
class SignalInterferenceFeature : public VectorFloatFeatureStream{
public:
  SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference, double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "Signal Interference");

  virtual ~SignalInterferenceFeature() {};
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset() { signal_->reset(); interference_->reset(); VectorFloatFeatureStream::reset(); }

private:
  VectorFloatFeatureStreamPtr   	signal_;
  VectorFloatFeatureStreamPtr		interference_;
  const double				level_;
};

/*@}*/

/**
* \defgroup AmplificationFeature Amplification Feature
*/
/*@{*/

// ----- definition for class `AmplificationFeature' -----
//
class AmplificationFeature;
typedef Inherit<AmplificationFeature, VectorFloatFeatureStreamPtr> AmplificationFeaturePtr;
class AmplificationFeature : public VectorFloatFeatureStream {
 public:
 AmplificationFeature(const VectorFloatFeatureStreamPtr& src,
                      double amplify = 1.0,
                      const String& nm = "Amplification")
   : VectorFloatFeatureStream(src->size(), nm), src_(src), amplify_(amplify)
  {}
  virtual ~AmplificationFeature() {}
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset() { src_->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr src_;
  double amplify_;
};

// ----- definition for class `WriteSoundFile' -----
//
class WriteSoundFile {
public:
  WriteSoundFile(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32);
  ~WriteSoundFile();
  int write( gsl_vector *vector );
  int writeInt( gsl_vector *vector );
  int writeShort( gsl_vector *vector );
  int writeFloat( gsl_vector *vector );

private:
  sndfile::SNDFILE* sndfile_;
  sndfile::SF_INFO sfinfo_;
};

typedef refcount_ptr<WriteSoundFile> WriteSoundFilePtr;

#endif
