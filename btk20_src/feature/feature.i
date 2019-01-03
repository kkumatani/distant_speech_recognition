/**
 * @file feature.i
 * @brief Speech recognition front end.
 * @author John McDonough, Tobias Gehrig, Kenichi Kumatani, Friedrich Faubel
 */

%module(package="btk20") feature

%init {
  // NumPy needs to set up callback functions
  import_array();
}

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include "feature/feature.h"
#include "feature/lpc.h"
using namespace sndfile;
%}

%include btk.h
%include jexception.i
%include typedefs.i
%include vector.i
%include matrix.i

%import stream/stream.i

#ifdef AUTODOC
%section "Feature", before
#endif

enum
{/* Major formats. */
  SF_FORMAT_WAV  = 0x010000,    /* Microsoft WAV format (little endian). */
  SF_FORMAT_AIFF  = 0x020000,   /* Apple/SGI AIFF format (big endian). */
  SF_FORMAT_AU  = 0x030000,     /* Sun/NeXT AU format (big endian). */
  SF_FORMAT_RAW  = 0x040000,    /* RAW PCM data. */
  SF_FORMAT_PAF  = 0x050000,    /* Ensoniq PARIS file format. */
  SF_FORMAT_SVX  = 0x060000,    /* Amiga IFF / SVX8 / SV16 format. */
  SF_FORMAT_NIST  = 0x070000,   /* NIST Sphere format. */
  SF_FORMAT_VOC  = 0x080000,    /* VOC files. */
  SF_FORMAT_IRCAM  = 0x0A0000,  /* Berkeley/IRCAM/CARL */
  SF_FORMAT_W64  = 0x0B0000,    /* Sonic Foundry's 64 bit RIFF/WAV */
  SF_FORMAT_MAT4  = 0x0C0000,   /* Matlab (tm) V4.2 / GNU Octave 2.0 */
  SF_FORMAT_MAT5  = 0x0D0000,   /* Matlab (tm) V5.0 / GNU Octave 2.1 */
  SF_FORMAT_PVF  = 0x0E0000,    /* Portable Voice Format */
  SF_FORMAT_XI  = 0x0F0000,     /* Fasttracker 2 Extended Instrument */
  SF_FORMAT_HTK  = 0x100000,    /* HMM Tool Kit format */
  SF_FORMAT_SDS  = 0x110000,    /* Midi Sample Dump Standard */
  SF_FORMAT_AVR  = 0x120000,    /* Audio Visual Research */
  SF_FORMAT_WAVEX  = 0x130000,    /* MS WAVE with WAVEFORMATEX */

  /* Subtypes from here on. */

  SF_FORMAT_PCM_S8    = 0x0001,    /* Signed 8 bit data */
  SF_FORMAT_PCM_16    = 0x0002,    /* Signed 16 bit data */
  SF_FORMAT_PCM_24    = 0x0003,    /* Signed 24 bit data */
  SF_FORMAT_PCM_32    = 0x0004,    /* Signed 32 bit data */

  SF_FORMAT_PCM_U8    = 0x0005,    /* Unsigned 8 bit data (WAV and RAW only) */

  SF_FORMAT_FLOAT  = 0x0006,    /* 32 bit float data */
  SF_FORMAT_DOUBLE    = 0x0007, /* 64 bit float data */
  // SF_FORMAT_PCM_SHORTEN    = 0x0008,    /* Shorten. */

  SF_FORMAT_ULAW  = 0x0010,    /* U-Law encoded. */
  SF_FORMAT_ALAW  = 0x0011,    /* A-Law encoded. */
  SF_FORMAT_IMA_ADPCM    = 0x0012,    /* IMA ADPCM. */
  SF_FORMAT_MS_ADPCM    = 0x0013,    /* Microsoft ADPCM. */

  SF_FORMAT_GSM610    = 0x0020,    /* GSM 6.10 encoding. */
  SF_FORMAT_VOX_ADPCM    = 0x0021,    /* OKI / Dialogix ADPCM */

  SF_FORMAT_G721_32    = 0x0030,    /* 32kbs G721 ADPCM encoding. */
  SF_FORMAT_G723_24    = 0x0031,    /* 24kbs G723 ADPCM encoding. */
  SF_FORMAT_G723_40    = 0x0032,    /* 40kbs G723 ADPCM encoding. */

  SF_FORMAT_DWVW_12    = 0x0040,     /* 12 bit Delta Width Variable Word encoding. */
  SF_FORMAT_DWVW_16    = 0x0041,     /* 16 bit Delta Width Variable Word encoding. */
  SF_FORMAT_DWVW_24    = 0x0042,     /* 24 bit Delta Width Variable Word encoding. */
  SF_FORMAT_DWVW_N    = 0x0043,     /* N bit Delta Width Variable Word encoding. */

  SF_FORMAT_DPCM_8    = 0x0050,    /* 8 bit differential PCM (XI only) */
  SF_FORMAT_DPCM_16    = 0x0051,    /* 16 bit differential PCM (XI only) */

  /* Endian-ness options. */

  SF_ENDIAN_FILE  = 0x00000000,  /* Default file endian-ness. */
  SF_ENDIAN_LITTLE    = 0x10000000,  /* Force little endian-ness. */
  SF_ENDIAN_BIG  = 0x20000000,  /* Force big endian-ness. */
  SF_ENDIAN_CPU  = 0x30000000,  /* Force CPU endian-ness. */

  SF_FORMAT_SUBMASK    = 0x0000FFFF,
  SF_FORMAT_TYPEMASK    = 0x0FFF0000,
  SF_FORMAT_ENDMASK    = 0x30000000
};

enum
{/* True and false */
  SF_FALSE  = 0,
  SF_TRUE    = 1,

  /* Modes for opening files. */
  SFM_READ  = 0x10,
  SFM_WRITE  = 0x20,
  SFM_RDWR  = 0x30
};


// ----- definition for class `FileFeature' -----
// 
%ignore FileFeature;
class FileFeature : public VectorFloatFeatureStream {
public:
  FileFeature(unsigned sz, const String nm = "File");

  void bload(const String fileName, bool old = false) { bload(fileName, old); }

  unsigned size() const;

  const gsl_vector_float* next() const;

  void copy(gsl_matrix_float* matrix);
};

class FileFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FileFeaturePtr(unsigned sz, const String nm = "File") {
      return new FileFeaturePtr(new FileFeature(sz, nm));
    }

    FileFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FileFeature* operator->();
};


// ----- definition of Conversion24bit2Short -----
//
class Conversion24bit2Short : public VectorShortFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  Conversion24bit2Short(VectorCharFeatureStreamPtr& src,
                        const String& nm = "Conversion from 24 bit integer to Short");
  virtual const gsl_vector_short* next(int frame_no = -5);
  virtual void reset() { _src->reset(); VectorShortFeatureStream::reset(); }
};

class Conversion24bit2ShortPtr : public VectorShortFeatureStreamPtr {
  %feature("kwargs") Conversion24bit2ShortPtr;
 public:
  %extend {
   Conversion24bit2ShortPtr(VectorCharFeatureStreamPtr& src,
                            const String& nm = "Conversion from 24 bit integer to Short") {
      return new Conversion24bit2ShortPtr(new Conversion24bit2Short(src, nm));
   }
    Conversion24bit2ShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  Conversion24bit2ShortPtr* operator->();
};


// ----- definition of Conversion24bit2Float -----
//
class Conversion24bit2Float : public VectorFloatFeatureStream {
  %feature("kwargs") reset;
  %feature("kwargs") next;
 public:
  Conversion24bit2Float(VectorCharFeatureStreamPtr& src,
                        const String& nm = "Conversion from 24 bit integer to Float");
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class Conversion24bit2FloatPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") Conversion24bit2FloatPtr;
 public:
  %extend {
    Conversion24bit2FloatPtr(VectorCharFeatureStreamPtr& src,
                             const String& nm = "Conversion from 24 bit integer to Float") {
      return new Conversion24bit2FloatPtr(new Conversion24bit2Float(src, nm));
   }
    Conversion24bit2FloatPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  Conversion24bit2FloatPtr* operator->();
};

// ----- definition for class `SampleFeature' -----
//
%ignore SampleFeature;
class SampleFeature : public VectorFloatFeatureStream {
  %feature("kwargs") read;
  %feature("kwargs") write;
  %feature("kwargs") cut;
  %feature("kwargs") reset;
  %feature("kwargs") next;
  %feature("kwargs") frame_no;
  %feature("kwargs") exit;
  %feature("kwargs") randomize;
  %feature("kwargs") data;
  %feature("kwargs") samplesN;
public:
  SampleFeature(const String fn = "", unsigned blockLen = 320,
                unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample");
  unsigned read(const String& fn, int format = 0, int samplerate = 16000,
                int chX = 1, int chN = 1, int cfrom = 0, int to = -1,
                int outsamplerate=-1, float norm = 0.0);
  void write(const String& fn, int format = SF_FORMAT_NIST|SF_FORMAT_PCM_16, int sampleRate = -1);
  void cut(unsigned cfrom, unsigned cto);
  virtual void reset();
  virtual const gsl_vector_float* next(int frame_no = -5);
  int frame_no() const;
  void exit();
  void randomize(int startX, int endX, double sigma2);
  const gsl_vector_float* data();
  unsigned samplesN() const;

  const gsl_vector* dataDouble();
  void zeroMean();
  void addWhiteNoise( float snr );
  void setSamples(const gsl_vector* samples, unsigned sampleRate);
  void copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to);
  int getSampleRate() const;
  int getChanN() const;
};

class SampleFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SampleFeaturePtr;
 public:
  %extend {
    SampleFeaturePtr(const String fn = "", unsigned block_len = 320,
                     unsigned shift_len = 160, bool pad_zeros = false, const String nm = "Sample") {
      return new SampleFeaturePtr(new SampleFeature(fn, block_len, shift_len, pad_zeros, nm));
    }

    SampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SampleFeature* operator->();
};

// ----- definition for class `SampleFeatureRunon' -----
//
%ignore SampleFeatureRunon;
class SampleFeatureRunon : public SampleFeature {
  %feature("kwargs") frame_no;
  %feature("kwargs") frameN;
public:
  SampleFeatureRunon(const String fn = "", unsigned blockLen = 320,
                     unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample");

  virtual int frame_no() const;
  virtual int frameN() const;
};

class SampleFeatureRunonPtr : public SampleFeaturePtr {
  %feature("kwargs") SampleFeatureRunonPtr;
 public:
  %extend {
    SampleFeatureRunonPtr(const String fn = "", unsigned blockLen = 320,
                          unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample") {
      return new SampleFeatureRunonPtr(new SampleFeatureRunon(fn, blockLen, shiftLen, padZeros, nm));
    }

    SampleFeatureRunonPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SampleFeatureRunon* operator->();
};

// ----- definition for class `IterativeSingleChannelSampleFeature' -----
//
%ignore IterativeSingleChannelSampleFeature;
class IterativeSingleChannelSampleFeature : public VectorFloatFeatureStream {
  %feature("kwargs") read;
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") samplesN;
public:
  IterativeSingleChannelSampleFeature(unsigned blockLen = 320, const String& nm = "IterativeSingleChannelSampleFeature");

  void read(const String& fileName, int format = 0, int samplerate = 44100, int cfrom = 0, int cto = -1 );
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  unsigned samplesN() const;
};

class IterativeSingleChannelSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") IterativeSingleChannelSampleFeaturePtr;
 public:
  %extend {
    IterativeSingleChannelSampleFeaturePtr(unsigned block_len = 320, const String& nm = "IterativeSingleChannelSampleFeature") {
      return new IterativeSingleChannelSampleFeaturePtr(new IterativeSingleChannelSampleFeature( block_len, nm ));
    }

    IterativeSingleChannelSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IterativeSingleChannelSampleFeature* operator->();
};

// ----- definition for class `IterativeSampleFeature' -----
//
%ignore IterativeSampleFeature;
class IterativeSampleFeature : public VectorFloatFeatureStream {
  %feature("kwargs") read;
  %feature("kwargs") next;
  %feature("kwargs") samplesN;
public:
  IterativeSampleFeature(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample");
  void read(const String& fileName, int format = SF_FORMAT_NIST|SF_FORMAT_PCM_32, int samplerate = 44100, int chN = 64, int cfrom = 0, int cto = -1 );
  virtual const gsl_vector_float* next(int frame_no = -5);
  unsigned samplesN() const;

  void changeFirstChannelID(unsigned firstChanX);
};

class IterativeSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") IterativeSampleFeaturePtr;
 public:
  %extend {
    IterativeSampleFeaturePtr(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample") {
      return new IterativeSampleFeaturePtr(new IterativeSampleFeature(chX, blockLen, firstChanX,nm));
    }

    IterativeSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IterativeSampleFeature* operator->();
};


// ----- definition for class `BlockSizeConversionFeature' -----
//
%ignore BlockSizeConversionFeature;
class BlockSizeConversionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
                             unsigned blockLen = 320,
                             unsigned shiftLen = 160, const String& nm = "BlockSizeConversion");
  const gsl_vector_float* next(int frame_no = -5) const;
};

class BlockSizeConversionFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") BlockSizeConversionFeaturePtr;
 public:
  %extend {
    BlockSizeConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                                  unsigned blockLen = 320,
                                  unsigned shiftLen = 160, const String& nm = "BlockSizeConversion") {
      return new BlockSizeConversionFeaturePtr(new BlockSizeConversionFeature(src, blockLen, shiftLen));
    }

    BlockSizeConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockSizeConversionFeature* operator->();
};


// ----- definition for class `BlockSizeConversionFeatureShort' -----
//
%ignore BlockSizeConversionFeatureShort;
class BlockSizeConversionFeatureShort : public VectorShortFeatureStream {
  %feature("kwargs") next;
public:
  BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
                                  unsigned blockLen = 320,
                                  unsigned shiftLen = 160, const String& nm = "BlockSizeConversion");
  const gsl_vector_short* next() const;
};

class BlockSizeConversionFeatureShortPtr : public VectorShortFeatureStreamPtr {
  %feature("kwargs") BlockSizeConversionFeatureShortPtr;
 public:
  %extend {
    BlockSizeConversionFeatureShortPtr(VectorShortFeatureStreamPtr& src,
                                       unsigned blockLen = 320,
                                       unsigned shiftLen = 160, const String& nm = "BlockSizeConversion") {
      return new BlockSizeConversionFeatureShortPtr(new BlockSizeConversionFeatureShort(src, blockLen, shiftLen));
    }

    BlockSizeConversionFeatureShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockSizeConversionFeatureShort* operator->();
};


#ifdef SMARTFLOW

// ----- definition for class `SmartFlowFeature' -----
//
%ignore SmartFlowFeature;
class SmartFlowFeature : public VectorShortFeatureStream {
  %feature("kwargs") next;
public:
  SmartFlowFeature(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
                   unsigned shiftLen = 160, const String& nm = "SmartFlowFeature");
  const gsl_vector_short* next() const;
};

class SmartFlowFeaturePtr : public VectorShortFeatureStreamPtr {
  %feature("kwargs") SmartFlowFeaturePtr;
 public:
  %extend {
    SmartFlowFeaturePtr(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
                        unsigned shiftLen = 160, const String& nm = "SmartFlowFeature") {
      return new SmartFlowFeaturePtr(new SmartFlowFeature(sfflow, blockLen, shiftLen, nm));
    }

    SmartFlowFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SmartFlowFeature* operator->();
};

#endif


// ----- definition for class `PreemphasisFeature' -----
//
%ignore PreemphasisFeature;
class PreemphasisFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") next_speaker;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") nextSpeaker;
#endif
public:
  PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis");
  const gsl_vector_float* next() const;
  virtual void reset();
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif
};

class PreemphasisFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") PreemphasisFeaturePtr;
 public:
  %extend {
    PreemphasisFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis") {
      return new PreemphasisFeaturePtr(new PreemphasisFeature(samp, mu, nm));
    }

    PreemphasisFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PreemphasisFeature* operator->();
};


// ----- definition for class `HammingFeatureShort' -----
//
%ignore HammingFeatureShort;
class HammingFeatureShort : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm = "HammingShort");
  const gsl_vector_float* next() const;
};

class HammingFeatureShortPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") HammingFeatureShortPtr;
 public:
  %extend {
    HammingFeatureShortPtr(const VectorShortFeatureStreamPtr& samp, const String& nm = "HammingShort") {
      return new HammingFeatureShortPtr(new HammingFeatureShort(samp, nm));
    }

    HammingFeatureShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  HammingFeatureShort* operator->();
};


// ----- definition for class `HammingFeature' -----
//
%ignore HammingFeature;
class HammingFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming");
  const gsl_vector_float* next() const;
};

class HammingFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") HammingFeaturePtr;
 public:
  %extend {
    HammingFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming") {
      return new HammingFeaturePtr(new HammingFeature(samp, nm));
    }

    HammingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  HammingFeature* operator->();
};


// ----- definition for class `FFTFeature' -----
//
%ignore FFTFeature;
class FFTFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") fftLen;
  %feature("kwargs") nBlocks;
  %feature("kwargs") subsamplerate;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") subSampRate;
#endif
public:
  FFTFeature(const VectorFloatFeatureStreamPtr samp, unsigned fftLen = 512, const String& nm = "FFT");
  const gsl_vector_complex* next(int frame_no = -5) const;

  unsigned fftLen()    const;
  unsigned windowLen() const;
  unsigned nBlocks()     const;
  unsigned subsamplerate() const;

#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() const;
#endif
};

class FFTFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") FFTFeaturePtr;
 public:
  %extend {
    FFTFeaturePtr(const VectorFloatFeatureStreamPtr samp, unsigned fft_len = 512, const String& nm = "FFT") {
      return new FFTFeaturePtr(new FFTFeature(samp, fft_len, nm));
    }

    FFTFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FFTFeature* operator->();
};

// ----- definition for class `SpectralPowerFloatFeature' -----
//
%ignore SpectralPowerFloatFeature;
class SpectralPowerFloatFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power");
  const gsl_vector_float* next() const;
};

class SpectralPowerFloatFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SpectralPowerFloatFeaturePtr;
 public:
  %extend {
    SpectralPowerFloatFeaturePtr(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power") {
      return new SpectralPowerFloatFeaturePtr(new SpectralPowerFloatFeature(fft, powN, nm));
    }

    SpectralPowerFloatFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralPowerFloatFeature* operator->();
};


// ----- definition for class `SpectralPowerFeature' -----
//
%ignore SpectralPowerFeature;
class SpectralPowerFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power");
  const gsl_vector* next() const;
};

class SpectralPowerFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") SpectralPowerFeaturePtr;
 public:
  %extend {
    SpectralPowerFeaturePtr(const VectorComplexFeatureStreamPtr& fft, unsigned pow_num = 0, const String nm = "Power") {
      return new SpectralPowerFeaturePtr(new SpectralPowerFeature(fft, pow_num, nm));
    }

    SpectralPowerFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralPowerFeature* operator->();
};


// ----- definition for class `SignalPowerFeature' -----
//
%ignore SignalPowerFeature;
class SignalPowerFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  SignalPowerFeature(const VectorFloatFeatureStreamPtr& samp, const String nm = "SignalPower");
  const gsl_vector_float* next() const;
};

class SignalPowerFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SignalPowerFeaturePtr;
 public:
  %extend {
    SignalPowerFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String nm = "SignalPower") {
      return new SignalPowerFeaturePtr(new SignalPowerFeature(samp, nm));
    }

    SignalPowerFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SignalPowerFeature* operator->();
};


// ----- definition for class `ALogFeature' -----
//
%ignore ALogFeature;
class ALogFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") next_speaker;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") nextSpeaker;
#endif
public:
  ALogFeature(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
              bool runon = false, const String nm = "ALogPower");
  const gsl_vector_float* next() const;
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif
};

class ALogFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") ALogFeaturePtr;
 public:
  %extend {
    ALogFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
                   bool runon = false, const String nm = "ALog Power") {
      return new ALogFeaturePtr(new ALogFeature(samp, m, a, runon, nm));
    }

    ALogFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ALogFeature* operator->();
};


// ----- definition for class `NormalizeFeature' -----
//
%ignore NormalizeFeature;
class NormalizeFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") next_speaker;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") nextSpeaker;
#endif
public:
  NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
                   bool runon = false, const String nm = "Normalize");
  const gsl_vector_float* next() const;
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif
};

class NormalizeFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") NormalizeFeaturePtr;
 public:
  %extend {
    NormalizeFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
                        bool runon = false, const String nm = "Normalize") {
      return new NormalizeFeaturePtr(new NormalizeFeature(samp, min, max, runon, nm));
    }

    NormalizeFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizeFeature* operator->();
};


// ----- definition for class `ThresholdFeature' -----
//
%ignore ThresholdFeature;
class ThresholdFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
                   const String& mode = "upper", const String& nm = "Threshold");
  const gsl_vector_float* next() const;
};

class ThresholdFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") ThresholdFeaturePtr;
 public:
  %extend {
    ThresholdFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
                        const String& mode = "upper", const String& nm = "Threshold") {
      return new ThresholdFeaturePtr(new ThresholdFeature(samp, value, thresh, mode, nm));
    }

    ThresholdFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ThresholdFeature* operator->();
};


// ----- definition for class `SpectralResamplingFeature' -----
//
%ignore SpectralResamplingFeature;
class SpectralResamplingFeature : public VectorFeatureStream {
  %feature("kwargs") next;
 public:
  SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio, unsigned len = 0,
                            const String& nm = "Resampling");
  virtual ~SpectralResamplingFeature();
  virtual const gsl_vector* next(int frame_no = -5);
};

class SpectralResamplingFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") SpectralResamplingFeaturePtr;
 public:
  %extend {
    SpectralResamplingFeaturePtr(const VectorFeatureStreamPtr& src, double ratio, unsigned len = 0,
                                 const String& nm = "Resampling") {
      return new SpectralResamplingFeaturePtr(new SpectralResamplingFeature(src, ratio, len, nm));
    }

    SpectralResamplingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralResamplingFeature* operator->();
};


#ifdef SRCONV

// ----- definition for class `SamplerateConversionFeature' -----
//
%ignore SamplerateConversionFeature;
class SamplerateConversionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
 public:
  SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src,
                              unsigned sourcerate = 22050, unsigned destrate = 16000,
                              unsigned len = 0, const String& method = "fastest",
                              const String& nm = "SamplerateConversion");
  virtual ~SamplerateConversionFeature() { }
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class SamplerateConversionFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SamplerateConversionFeaturePtr;
 public:
  %extend {
    SamplerateConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                                   unsigned sourcerate = 22050, unsigned destrate = 16000,
                                   unsigned len = 0, const String& method = "fastest",
                                   const String& nm = "SamplerateConversion") {
      return new SamplerateConversionFeaturePtr(new SamplerateConversionFeature(src, sourcerate, destrate, len, method, nm));
    }

    SamplerateConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SamplerateConversionFeature* operator->();
};

#endif


// ----- definition for class `VTLNFeature' -----
//
%ignore VTLNFeature;
class VTLNFeature : public VectorFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") matrix;
  %feature("kwargs") warp;
public:
  VTLNFeature(const VectorFeatureStreamPtr& pow,
              unsigned coeffN = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
              const String& nm = "VTLN");
  virtual const gsl_vector* next(int frame_no = -5) const;
  void matrix(gsl_matrix* matrix) const;
  // specify the warp factor
  void warp(double w);
};

class VTLNFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") VTLNFeaturePtr;
 public:
  %extend {
    VTLNFeaturePtr(const VectorFeatureStreamPtr& pow,
                   unsigned coeff_num = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
                   const String& nm = "VTLN") {
      return new VTLNFeaturePtr(new VTLNFeature(pow, coeff_num, ratio, edge, version, nm));
    }

    VTLNFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VTLNFeature* operator->();
};


// ----- definition for class `MelFeature' -----
//
%ignore MelFeature;
class MelFeature : public VectorFeatureStream {
  %feature("kwargs") read;
  %feature("kwargs") next;
  %feature("kwargs") matrix;
public:
  MelFeature(const VectorFeatureStreamPtr mag, int powN = 0,
             float rate = 16000.0, float low = 0.0, float up = 0.0,
             int filterN = 30, int version = 1, const String& nm = "MelFFT");
  const gsl_vector* next() const;
  void read(const String& fileName);
  void matrix(gsl_matrix* matrix) const;
};

class MelFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") MelFeaturePtr;
 public:
  %extend {
    MelFeaturePtr(const VectorFeatureStreamPtr mag, int pow_num = 0,
                  float rate = 16000.0, float low = 0.0, float up = 0.0,
                  int filter_num = 30, int version = 1, const String& nm = "MelFFT") {
      return new MelFeaturePtr(new MelFeature(mag, pow_num, rate, low, up, filter_num, version, nm));
    }

    MelFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MelFeature* operator->();
};


// ----- definition for class `SphinxMelFeature' -----
//
%ignore SphinxMelFeature;
class SphinxMelFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
                   float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
                   unsigned filterN = 30, const String& nm = "Sphinx Mel Filter Bank");
  const gsl_vector* next() const;
};

class SphinxMelFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") SphinxMelFeaturePtr;
 public:
  %extend {
    SphinxMelFeaturePtr(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
                        float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
                        unsigned filterN = 30, const String& nm = "Sphinx Mel Filter Bank") {
      return new SphinxMelFeaturePtr(new SphinxMelFeature(mag, fftN, powerN, sampleRate, lowerF, upperF, filterN, nm));
    }

    SphinxMelFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphinxMelFeature* operator->();
};


// ----- definition for class `LogFeature' -----
//
%ignore LogFeature;
class LogFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  LogFeature(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
             bool sphinxFlooring = false, const String& nm = "LogMel");
  const gsl_vector_float* next() const;
};

class LogFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") LogFeaturePtr;
 public:
  %extend {
    LogFeaturePtr(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
                  bool sphinxFlooring = false, const String& nm = "LogMel") {
      return new LogFeaturePtr(new LogFeature(mel, m, a, sphinxFlooring, nm));
    }

    LogFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LogFeature* operator->();
};

// ----- definition for class `FloatToDoubleConversionFeature' -----
//
%ignore FloatToDoubleConversionFeature;
class FloatToDoubleConversionFeature : public VectorFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  FloatToDoubleConversionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Float toDoubleConversion");
  virtual ~FloatToDoubleConversionFeature();
  virtual const gsl_vector* next(int frame_no = -5);
  virtual void reset();
};

class FloatToDoubleConversionFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") FloatToDoubleConversionFeaturePtr;
 public:
  %extend {
    FloatToDoubleConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Float toDoubleConversion") {
      return new FloatToDoubleConversionFeaturePtr(new FloatToDoubleConversionFeature(src, nm));
    }

    FloatToDoubleConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FloatToDoubleConversionFeaturePtr* operator->();
};


// ----- definition for class `CepstralFeature' -----
//
%ignore CepstralFeature;
// type:
//   0 = 
//   1 = Type 2 DCT
//   2 = Sphinx Legacy
class CepstralFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") matrix;
public:
  CepstralFeature(const VectorFloatFeatureStreamPtr mel, unsigned ncep = 13, int type = 1, const String nm = "Cepstral");
  const gsl_vector_float* next() const;
  gsl_matrix* matrix() const;
};

class CepstralFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") CepstralFeaturePtr;
 public:
  %extend {
    CepstralFeaturePtr(const VectorFloatFeatureStreamPtr mel, unsigned ncep = 13, int type = 1, const String nm = "Cepstral") {
      return new CepstralFeaturePtr(new CepstralFeature(mel, ncep, type, nm));
    }

    CepstralFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CepstralFeature* operator->();
};


// ----- definition for class `WarpMVDRFeature' -----
//
%ignore WarpMVDRFeature;
class WarpMVDRFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  WarpMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, const String& nm = "MVDR");
  const gsl_vector_float* next() const;
};

class WarpMVDRFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") WarpMVDRFeaturePtr ;
 public:
  %extend {
    WarpMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, const String& nm = "MVDR") {
      return new WarpMVDRFeaturePtr(new WarpMVDRFeature(src, order, correlate, warp, nm));
    }

    WarpMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpMVDRFeature* operator->();
};


// ----- definition for class `BurgMVDRFeature' -----
//
%ignore BurgMVDRFeature;
class BurgMVDRFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  BurgMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, const String& nm = "MVDR");
  const gsl_vector_float* next() const;
};

class BurgMVDRFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") BurgMVDRFeaturePtr;
 public:
  %extend {
    BurgMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, const String& nm = "MVDR") {
      return new BurgMVDRFeaturePtr(new BurgMVDRFeature(src, order, correlate, warp, nm));
    }

    BurgMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BurgMVDRFeature* operator->();
};


// ----- definition for class `WarpedTwiceFeature' -----
//
%ignore WarpedTwiceMVDRFeature;
class WarpedTwiceMVDRFeature : public VectorFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  WarpedTwiceMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, bool warpFactorFixed=false, float sensibility = 0.1, const String& nm = "WTMVDR");
  virtual ~WarpedTwiceMVDRFeature();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void reset();
};

class WarpedTwiceMVDRFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") WarpedTwiceMVDRFeaturePtr;
 public:
  %extend {
    WarpedTwiceMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, bool warpFactorFixed=false, float sensibility = 0.1, const String& nm = "WTMVDR"){
      return new WarpedTwiceMVDRFeaturePtr(new WarpedTwiceMVDRFeature( src, order, correlate, warp, warpFactorFixed, sensibility, nm));
    }

    WarpedTwiceMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpedTwiceMVDRFeature* operator->();
};


// ----- definition for class `WarpLPCFeature' -----
//
%ignore WarpLPCFeature;
class WarpLPCFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  WarpLPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm = "LPC");
  const gsl_vector* next() const;
};

class WarpLPCFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") WarpLPCFeaturePtr;
 public:
  %extend {
    WarpLPCFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm = "LPC") {
      return new WarpLPCFeaturePtr(new WarpLPCFeature(src, order, correlate, warp, nm));
    }

    WarpLPCFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpLPCFeature* operator->();
};


// ----- definition for class `BurgLPCFeature' -----
//
%ignore BurgLPCFeature;
class BurgLPCFeature : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  BurgLPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm = "LPC");
  const gsl_vector* next() const;
};

class BurgLPCFeaturePtr : public VectorFeatureStreamPtr {
  %feature("kwargs") BurgLPCFeaturePtr;
 public:
  %extend {
    BurgLPCFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, const String& nm = "LPC") {
      return new BurgLPCFeaturePtr(new BurgLPCFeature(src, order, correlate, warp, nm));
    }

    BurgLPCFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BurgLPCFeature* operator->();
};


// ----- definition for class `SpectralSmoothing' -----
//
%ignore SpectralSmoothing;
class SpectralSmoothing : public VectorFeatureStream {
  %feature("kwargs") next;
public:
  SpectralSmoothing(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom, const String& nm = "Spectral Smoothing");
  const gsl_vector* next() const;
};

class SpectralSmoothingPtr : public VectorFeatureStreamPtr {
  %feature("kwargs") SpectralSmoothingPtr;
 public:
  %extend {
    SpectralSmoothingPtr(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom, const String& nm = "Spectral Smoothing") {
      return new SpectralSmoothingPtr(new SpectralSmoothing(adjustTo, adjustFrom, nm));
    }

    SpectralSmoothingPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralSmoothing* operator->();
};


// ----- definition for class `MeanSubtractionFeature' -----
//
%ignore MeanSubtractionFeature;
class MeanSubtractionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") mean;
  %feature("kwargs") write;
  %feature("kwargs") next_speaker;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") nextSpeaker;
#endif
 public:
  MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL, double devNormFactor = 0.0, bool runon = false, const String& nm = "Mean Subtraction");
  virtual ~MeanSubtractionFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  const gsl_vector_float* mean() const;
  void write(const String& fileName, bool variance = false) const;
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif
};

class MeanSubtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") MeanSubtractionFeaturePtr;
 public:
  %extend {
    MeanSubtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL, double devNormFactor = 0.0, bool runon = false, const String nm = "Mean Subtraction") {
      return new MeanSubtractionFeaturePtr(new MeanSubtractionFeature(src, weight, devNormFactor, runon, nm));
    }

    MeanSubtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MeanSubtractionFeature* operator->();
};


// ----- definition for class `FileMeanSubtractionFeature' -----
//
%ignore FileMeanSubtractionFeature;
class FileMeanSubtractionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") read;
 public:
  FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src,
                             double devNormFactor = 0.0, const String& nm = "MeanSubtraction");
  virtual ~FileMeanSubtractionFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  void read(const String& fileName, bool variance = false);
};

class FileMeanSubtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") FileMeanSubtractionFeaturePtr;
 public:
  %extend {
    FileMeanSubtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                                  double devNormFactor = 0.0, const String nm = "MeanSubtraction") {
      return new FileMeanSubtractionFeaturePtr(new FileMeanSubtractionFeature(src, devNormFactor, nm));
    }

    FileMeanSubtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FileMeanSubtractionFeature* operator->();
};


// ----- definition for class `AdjacentFeature' -----
//
%ignore AdjacentFeature;
class AdjacentFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
                  const String& nm = "Adjacent");
  virtual ~AdjacentFeature();
  const gsl_vector_float* next() const;
};

class AdjacentFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") AdjacentFeaturePtr;
 public:
  %extend {
    AdjacentFeaturePtr(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
                       const String& nm = "Adjacent") {
      return new AdjacentFeaturePtr(new AdjacentFeature(single, delta, nm));
    }

    AdjacentFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  AdjacentFeature* operator->();
};


// ----- definition for class `LinearTransformFeature' -----
//
%ignore LinearTransformFeature;
class LinearTransformFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") matrix;
  %feature("kwargs") load;
  %feature("kwargs") identity;
 public:
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform");
  virtual ~LinearTransformFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  gsl_matrix_float* matrix() const;
  void load(const String& fileName, bool old = false);
  void identity();
};

class LinearTransformFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") inearTransformFeaturePtr;
 public:
  %extend {
    LinearTransformFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform") {

      cout << "Allocating 'LinearTransformFeaturePtr'" << endl;
      return new LinearTransformFeaturePtr(new LinearTransformFeature(src, sz, nm));
    }

    LinearTransformFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LinearTransformFeature* operator->();
};


// ----- definition for class `StorageFeature' -----
//
%ignore StorageFeature;
class StorageFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") write;
  %feature("kwargs") read;
  %feature("kwargs") evaluate;
  %feature("kwargs") reset;
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  StorageFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Storage");
  virtual ~StorageFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  void write(const String& fileName, bool plainText = false) const;
  void read(const String& fileName);
  int  evaluate();
  void reset();
};

class StorageFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") StorageFeaturePtr;
 public:
  %extend {
    StorageFeaturePtr(const VectorFloatFeatureStreamPtr& src = NULL, const String& nm = "Storage") {
      return new StorageFeaturePtr(new StorageFeature(src, nm));
    }

    StorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  StorageFeature* operator->();
};

// ----- definition for class `StaticStorageFeature' -----
//
%ignore StaticStorageFeature;
class StaticStorageFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") read;
  %feature("kwargs") evaluate;
  %feature("kwargs") reset;
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  StaticStorageFeature(unsigned dim, const String& nm = "Storage");
  virtual ~StaticStorageFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  //void write(const String& fileName) const;
  void read(const String& fileName);
  int  evaluate();
  void reset();

  unsigned currentNFrames();
};

class StaticStorageFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") StaticStorageFeaturePtr;
 public:
  %extend {
    StaticStorageFeaturePtr(unsigned dim, const String& nm = "Storage") {
      return new StaticStorageFeaturePtr(new StaticStorageFeature(dim, nm));
    }

    StaticStorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  StaticStorageFeature* operator->();
};

// ----- definition for class `CircularStorageFeature' -----
//
%ignore CircularStorageFeature;
class CircularStorageFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN = 3,
                         const String& nm = "Storage");
  virtual ~CircularStorageFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class CircularStorageFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") CircularStorageFeaturePtr;
 public:
  %extend {
    CircularStorageFeaturePtr(const VectorFloatFeatureStreamPtr& src, int framesN = 3,
                              const String& nm = "Storage") {
      return new CircularStorageFeaturePtr(new CircularStorageFeature(src, framesN, nm));
    }

    CircularStorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CircularStorageFeature* operator->();
};


// ----- definition for class `FilterFeature' -----
//
%ignore FilterFeature;
class FilterFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
 public:
  FilterFeature(const VectorFloatFeatureStreamPtr& src, gsl_vector* coeffA,
                const String& nm = "Filter");
  virtual ~FilterFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class FilterFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") FilterFeaturePtr;
 public:
  %extend {
    FilterFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                     gsl_vector* coeffA, const String& nm = "Filter") {
      return new FilterFeaturePtr(new FilterFeature(src, coeffA, nm));
    }

    FilterFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FilterFeature* operator->();
};


// ----- definition for class `MergeFeature' -----
//
%ignore MergeFeature;
class MergeFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
 public:
  MergeFeature(VectorFloatFeatureStreamPtr& stat,
               VectorFloatFeatureStreamPtr& delta,
               VectorFloatFeatureStreamPtr& deltaDelta,
               const String& nm = "Merge");
  virtual ~MergeFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class MergeFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") MergeFeaturePtr;
 public:
  %extend {
    MergeFeaturePtr(VectorFloatFeatureStreamPtr& stat,
                    VectorFloatFeatureStreamPtr& delta,
                    VectorFloatFeatureStreamPtr& deltaDelta,
                    const String& nm = "Merge") {
      return new MergeFeaturePtr(new MergeFeature(stat, delta, deltaDelta, nm));
    }

    MergeFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MergeFeature* operator->();
};

// ----- definition for class `MultiModalFeature' -----
//
%ignore MultiModalFeature;
class MultiModalFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
 public:
  MultiModalFeature(unsigned nModality, unsigned totalVecSize, const String& nm = "MultiModal");
  virtual ~MultiModalFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);

  void addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec=1 );
};

class MultiModalFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") MultiModalFeaturePtr;
 public:
  %extend {
    MultiModalFeaturePtr(unsigned nModality, unsigned totalVecSize,
                         const String& nm = "MultiModal") {
      return new MultiModalFeaturePtr(new MultiModalFeature(nModality, totalVecSize, nm));
    }

    MultiModalFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiModalFeature* operator->();
};

// ----- definition for class `FeatureSet' -----
//
%ignore FeatureSet;
class FeatureSet {
  %feature("kwargs") name;
  %feature("kwargs") add;
  %feature("kwargs") feature;
public:
  FeatureSet(const String nm = "FeatureSet");
  const String name() const;
  void add(VectorFloatFeatureStreamPtr feat);
  VectorFloatFeatureStreamPtr feature(const String nm);
};

class FeatureSetPtr {
  %feature("kwargs") FeatureSetPtr;
 public:
  %extend {
    FeatureSetPtr(const String nm = "FeatureSet") {
      return new FeatureSetPtr(new FeatureSet(nm));
    }

    // return a codebook
    VectorFloatFeatureStreamPtr __getitem__(const String name) {
      return (*self)->feature(name);
    }
  }

  FeatureSet* operator->();
};

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

%ignore Jack;
class Jack
{
  %feature("kwargs") start;

  %feature("kwargs") addPort;
  %feature("kwargs") getSampleRate;
 public:
  Jack(const String& nm);
  ~Jack();
  void start(void) { can_capture = true; };

  jack_channel_t* addPort(unsigned buffersize, const String& connection, const String& nm);
  unsigned getSampleRate();
};

class JackPtr {
  %feature("kwargs") JackPtr;
 public:
  %extend {
    JackPtr(const String& nm) {
      return new JackPtr(new Jack(nm));
    }
  }

  Jack* operator->();
};

%ignore JackFeature;
class JackFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
              const String& connection, const String& nm);
  virtual ~JackFeature() { };
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
};

class JackFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") JackFeaturePtr;
 public:
  %extend {
    JackFeaturePtr(JackPtr& jack, unsigned blockLen, unsigned buffersize,
                   const String& connection, const String& nm) {
      return new JackFeaturePtr(new JackFeature(jack, blockLen, buffersize, connection, nm));
    }

    JackFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  JackFeature* operator->();
};

#endif


// ----- definition for class `ZeroCrossingRateHammingFeature' -----
//
%ignore ZeroCrossingRateHammingFeature;
class ZeroCrossingRateHammingFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming");
  const gsl_vector_float* next() const;
};

class ZeroCrossingRateHammingFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") ZeroCrossingRateHammingFeaturePtr;
 public:
  %extend {
    ZeroCrossingRateHammingFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming") {
      return new ZeroCrossingRateHammingFeaturePtr(new ZeroCrossingRateHammingFeature(samp, nm));
    }

    ZeroCrossingRateHammingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ZeroCrossingRateHammingFeature* operator->();
};

// ----- definition for class `YINPitchFeature' -----
//
%ignore YINPitchFeature;
class YINPitchFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch");
  const gsl_vector_float* next() const;
};

class YINPitchFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") YINPitchFeaturePtr;
 public:
  %extend {
    YINPitchFeaturePtr(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch") {
      return new YINPitchFeaturePtr(new YINPitchFeature(samp, samplerate, threshold, nm));
    }

    YINPitchFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  YINPitchFeature* operator->();
};


// ----- definition for class `SpikeFilter' -----
//
%ignore SpikeFilter;
class SpikeFilter : public VectorFloatFeatureStream {
  %feature("kwargs") size;
  %feature("kwargs") next;
public:
  SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String& nm = "SpikeFilter");
  unsigned size() const;
  const gsl_vector_float* next() const;
};

class SpikeFilterPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SpikeFilterPtr;
 public:
  %extend {
    SpikeFilterPtr(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String nm = "SpikeFilter") {
      return new SpikeFilterPtr(new SpikeFilter(src, tapN, nm));
    }

    SpikeFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpikeFilter* operator->();
};


// ----- definition for class `SpikeFilter2' -----
//
%ignore SpikeFilter2;
class SpikeFilter2 : public VectorFloatFeatureStream {
  %feature("kwargs") size;
  %feature("kwargs") next;
public:
  SpikeFilter2(VectorFloatFeatureStreamPtr& src,
               unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
               const String& nm = "Spike Filter2");
  unsigned size() const;
  const gsl_vector_float* next() const;

  unsigned spikesN() const { return _count; }
};

class SpikeFilter2Ptr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SpikeFilter2Ptr;
 public:
  %extend {
    SpikeFilter2Ptr(VectorFloatFeatureStreamPtr& src,
                    unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
                    const String& nm = "Spike Filter2") {
      return new SpikeFilter2Ptr(new SpikeFilter2(src, width, maxslope, startslope, thresh, alpha, verbose, nm));
    }

    SpikeFilter2Ptr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpikeFilter2* operator->();
};


// ----- definition for class `SoundFile' -----
//
%ignore SoundFile;
class SoundFile {
  %feature("kwargs") readf;
  %feature("kwargs") writef;
  %feature("kwargs") read;
  %feature("kwargs") write;
  %feature("kwargs") seek;
 public:
  SoundFile(const String& fn,
            int mode,
            int format = 0,
            int samplerate = 16000,
            int channels = 1,
            bool normalize = false);
  ~SoundFile();
  sf_count_t frames() const;
  int samplerate() const;
  int channels() const;
  int format() const;
  int sections() const;
  int seekable() const;
  sf_count_t readf(float *ptr, sf_count_t frames);
  sf_count_t writef(float *ptr, sf_count_t frames);
  sf_count_t read(float *ptr, sf_count_t items);
  sf_count_t write(float *ptr, sf_count_t items);
  sf_count_t seek(sf_count_t frames, int whence = SEEK_SET);
};

class SoundFilePtr {
  %feature("kwargs")  SoundFilePtr;
 public:
  %extend {
    SoundFilePtr(const String& fn,
                 int mode,
                 int format = 0,
                 int samplerate = 16000,
                 int channels = 1,
                 bool normalize = false) {
      return new SoundFilePtr(new SoundFile(fn,
                                            mode,
                                            format,
                                            samplerate,
                                            channels,
                                            normalize));
    }
  }

  SoundFile* operator->();
};

// ----- definition for class `DirectSampleFeature' -----
//
%ignore DirectSampleFeature;
class DirectSampleFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") channels;
 public:
  DirectSampleFeature(const SoundFilePtr &sndfile,
                      unsigned blockLen = 320,
                      unsigned start = 0,
                      unsigned end = (unsigned)-1,
                      const String& nm = "DirectSample");
  ~DirectSampleFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  int channels() const;

  int sampleRate() const;
  void setRegion(unsigned start = 0, unsigned end = (unsigned)-1);
};

class DirectSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") DirectSampleFeaturePtr;
 public:
  %extend {
    DirectSampleFeaturePtr(const SoundFilePtr &sndfile,
                           unsigned blockLen = 320,
                           unsigned start = 0,
                           unsigned end = (unsigned)-1,
                           const String& nm = "DirectSample") {
      return new DirectSampleFeaturePtr(new DirectSampleFeature(sndfile, blockLen, start, end, nm));
    }

    DirectSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DirectSampleFeature* operator->();
};

// ----- definition for class `DirectSampleOutputFeature' -----
//
%ignore DirectSampleOutputFeature;
class DirectSampleOutputFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
                            const SoundFilePtr &sndfile,
                            const String& nm = "DirectSampleOutput");
  ~DirectSampleOutputFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
};

class DirectSampleOutputFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") DirectSampleOutputFeaturePtr;
 public:
  %extend {
    DirectSampleOutputFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                                 const SoundFilePtr &sndfile,
                                 const String& nm = "DirectSampleOutput") {
      return new DirectSampleOutputFeaturePtr(new DirectSampleOutputFeature(src, sndfile, nm));
    }

    DirectSampleOutputFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DirectSampleOutputFeature* operator->();
};

// ----- definition for class `ChannelExtractionFeature' -----
//
%ignore ChannelExtractionFeature;
class ChannelExtractionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
 public:
  ChannelExtractionFeature(const VectorFloatFeatureStreamPtr& src,
                           unsigned chX = 0,
                           unsigned chN = 1,
                           const String& nm = "ChannelExtraction");
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class ChannelExtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %feature("kwargs") ChannelExtractionFeaturePtr;
  %extend {
    ChannelExtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                                unsigned chX = 0,
                                unsigned chN = 1,
                                const String& nm = "ChannelExtraction") {
      return new ChannelExtractionFeaturePtr(new ChannelExtractionFeature(src, chX, chN, nm));
    }

    ChannelExtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ChannelExtractionFeature* operator->();
};


// ----- definition for class 'SampleInterferenceFeature -----
//
%ignore SignalInterferenceFeature;
class SignalInterferenceFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference, double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "SignalInterference");
  virtual const gsl_vector_float* next(int frame_no = -5);
};

class SignalInterferenceFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SignalInterferenceFeaturePtr;
public:
  %extend {
    SignalInterferenceFeaturePtr(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference, double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "SignalInterference") {
      return new SignalInterferenceFeaturePtr(new SignalInterferenceFeature(signal, interference, dBInterference, blockLen, nm));
    }
    SignalInterferenceFeaturePtr __iter__() {
      (*self)->reset(); return *self;
    }
  }
  SignalInterferenceFeature* operator->();
};


// ----- definition for class `AmplificationFeature' -----
//
%ignore AmplificationFeature;
class AmplificationFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  AmplificationFeature(const VectorFloatFeatureStreamPtr& src,
                       double amplify = 1.0,
                       const String& nm = "Amplification");
  virtual ~AmplificationFeature();
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
};

class AmplificationFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") AmplificationFeaturePtr;
 public:
  %extend {
    AmplificationFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                            double amplify = 1.0,
                            const String& nm = "Amplification") {
      return new AmplificationFeaturePtr(new AmplificationFeature(src,
                                                                  amplify,
                                                                  nm));
    }

    AmplificationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  AmplificationFeature* operator->();
};

// ----- definition for class `LPCSpectrumEstimator' -----
//
%ignore LPCSpectrumEstimator;
class LPCSpectrumEstimator : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  LPCSpectrumEstimator(const VectorFloatFeatureStreamPtr& source, unsigned order, unsigned fftLen, const String& nm = "LPC Spectrum Estimator");
  virtual ~LPCSpectrumEstimator();
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();

  const gsl_vector_float* getLPCs() const;
  const gsl_vector_float* getAutoCorrelationVector();
  float getPredictionError();
};

class LPCSpectrumEstimatorPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") LPCSpectrumEstimatorPtr;
 public:
  %extend {
    LPCSpectrumEstimatorPtr(const VectorFloatFeatureStreamPtr& source, unsigned order, unsigned fftLen, const String& nm = "LPC Spectrum Estimator") {
      return new LPCSpectrumEstimatorPtr(new LPCSpectrumEstimator(source, order, fftLen, nm));
    }

    LPCSpectrumEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LPCSpectrumEstimator* operator->();
};


// ----- definition for class `CepstralSpectrumEstimator' -----
//
%ignore CepstralSpectrumEstimator;
class CepstralSpectrumEstimator : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  CepstralSpectrumEstimator(const VectorComplexFeatureStreamPtr& source, unsigned order, unsigned fftLen, double logPadding = 1.0, const String& nm = "CepstralSpectrumEstimator");
  virtual ~CepstralSpectrumEstimator();
  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
};

class CepstralSpectrumEstimatorPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") CepstralSpectrumEstimatorPtr;
 public:
  %extend {
    CepstralSpectrumEstimatorPtr(const VectorComplexFeatureStreamPtr& source, unsigned order, unsigned fftLen, double logPadding = 1.0, const String& nm = "CepstralSpectrumEstimator") {
      return new CepstralSpectrumEstimatorPtr(new CepstralSpectrumEstimator(source, order, fftLen, logPadding, nm));
    }

    CepstralSpectrumEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CepstralSpectrumEstimator* operator->();
};


// ----- definition for class `SEMNB' -----
//
%ignore SEMNB;
class SEMNB {
  %feature("kwargs") reset;
 public:
  SEMNB( unsigned order, unsigned fftLen, const String& nm= "SEMNB");
  ~SEMNB();
  void reset();
  gsl_vector* calcDerivativeOfDeviation(LPCSpectrumEstimatorPtr &lpcSEPtr );
  const gsl_vector* getLPEnvelope();
};

class SEMNBPtr {
  %feature("kwargs") SEMNBPtr;
 public:
  %extend {
    SEMNBPtr( unsigned order, unsigned fftLen, const String& nm= "SEMNBPtr")
    {
      return new SEMNBPtr(new SEMNB( order, fftLen,  nm ));
    }
  }

  SEMNB* operator->();
};

// ----- definition for class `WriteSoundFile' -----
//
%ignore WriteSoundFile;
class WriteSoundFile {
  %feature("kwargs") write;
public:
  WriteSoundFile(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32);
  ~WriteSoundFile();
  int write( gsl_vector *vector );

  int writeInt( gsl_vector *vector );
  int writeShort( gsl_vector *vector );
  int writeFloat( gsl_vector *vector );
private:
  sndfile::SNDFILE* _sndfile;
  sndfile::SF_INFO _sfinfo;
};

class WriteSoundFilePtr {
  %feature("kwargs") WriteSoundFilePtr;
public:
  %extend {
    WriteSoundFilePtr(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32 ){
      return new WriteSoundFilePtr(new WriteSoundFile( fn, sampleRate, nChan, format));
    }
  }
  WriteSoundFile * operator->();
};
