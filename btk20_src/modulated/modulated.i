/**
 * @file modulated.i
 * @brief Cosine modulated analysis and synthesis filter banks.
 * @author John McDonough and Kenichi Kumatani
 */

%module(package="btk20") modulated

%{
#include "modulated/modulated.h"
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "modulated/prototype_design.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h
%include jexception.i
%include typedefs.i
%include complex.i
%include matrix.i
%include vector.i

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%pythoncode %{
import btk20
from btk20 import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

// ----- definition for class `NormalFFTAnalysisBank' -----
//
%ignore NormalFFTAnalysisBank;
class NormalFFTAnalysisBank : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") fftLen;
public:
  NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			unsigned fftLen,  unsigned r = 1, unsigned windowType = 1,
			const String& nm = "NormalFFTAnalysisBank");
  ~NormalFFTAnalysisBank();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  unsigned fftLen() const;
};

class NormalFFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") NormalFFTAnalysisBankPtr;
 public:
  %extend {
    NormalFFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
			     unsigned fftLen,  unsigned r = 1, unsigned window_type = 1,
			     const String& nm = "NormalFFTAnalysisBank") {
      return new NormalFFTAnalysisBankPtr(new NormalFFTAnalysisBank( samp, fftLen, r, window_type, nm ));
    }

     NormalFFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

   NormalFFTAnalysisBank* operator->();
};

// ----- definition for class `OverSampledDFTAnalysisBank' -----
//
%ignore OverSampledDFTAnalysisBank;
class OverSampledDFTAnalysisBank : public VectorComplexFeatureStream {
  %feature("kwargs") polyphase;
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") is_end;
  %feature("kwargs") fftLen;
  %feature("kwargs") nBlocks;
  %feature("kwargs") subsamplerate;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") subSampRate;
#endif
 public:
  OverSampledDFTAnalysisBank(VectorShortFeatureStreamPtr& samp,
			     gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r, unsigned delayCompensationType =0,
			     const String& nm = "OverSampledDFTAnalysisBank");
  ~OverSampledDFTAnalysisBank();
  double polyphase(unsigned m, unsigned n) const;
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  bool is_end();
  unsigned fftLen() const;
  unsigned nBlocks() const;
  unsigned subsamplerate() const;

#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() const;
#endif
};

class OverSampledDFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") OverSampledDFTAnalysisBankPtr;
 public:
  %extend {
    OverSampledDFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
                                  gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r = 0, unsigned delay_compensation_type = 0,
                                  const String& nm = "OverSampledDFTAnalysisBankFloat") {
      return new OverSampledDFTAnalysisBankPtr(new OverSampledDFTAnalysisBank(samp, prototype, M, m, r, delay_compensation_type, nm));
    }

    OverSampledDFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverSampledDFTAnalysisBank* operator->();
};


// ----- definition for class `OverSampledDFTSynthesisBank' -----
// 
%ignore OverSampledDFTSynthesisBank;
class OverSampledDFTSynthesisBank : public VectorFloatFeatureStream {
  %feature("kwargs") polyphase;
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") is_end;
  %feature("kwargs") fftLen;
  %feature("kwargs") nBlocks;
  %feature("kwargs") input_source_vector;
  %feature("kwargs") no_stream_feature;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") inputSourceVector;
  %feature("kwargs") doNotUseStreamFeature;
#endif
 public:
  OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& subband,
                              gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
                              unsigned delayCompensationType = 0, int gainFactor = 1,
                              const String& nm = "OverSampledDFTSynthesisBank");
  ~OverSampledDFTSynthesisBank();
  double polyphase(unsigned m, unsigned n) const;
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();

  void input_source_vector(const gsl_vector_complex* block);
  void no_stream_feature(bool flag=true);
#ifdef ENABLE_LEGACY_BTK_API
  void inputSourceVector(const gsl_vector_complex* block);
  void doNotUseStreamFeature(bool flag=true);
#endif
};

class OverSampledDFTSynthesisBankPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") OverSampledDFTSynthesisBankPtr;
public:
  %extend {
    OverSampledDFTSynthesisBankPtr(VectorComplexFeatureStreamPtr& samp,
				   gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
				   unsigned delay_compensation_type = 0, int gain_factor = 1,
				   const String& nm = "OverSampledDFTSynthesisBank") {
      return new OverSampledDFTSynthesisBankPtr(new OverSampledDFTSynthesisBank(samp, prototype, M, m, r, delay_compensation_type, gain_factor, nm));
    }

    OverSampledDFTSynthesisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverSampledDFTSynthesisBank* operator->();
};

// ----- definition for class `PerfectReconstructionFFTAnalysisBank' -----
//
%ignore PerfectReconstructionFFTAnalysisBank;
class PerfectReconstructionFFTAnalysisBank : public VectorComplexFeatureStream {
  %feature("kwargs") polyphase;
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") is_end;
  %feature("kwargs") fftLen;
  %feature("kwargs") nBlocks;
  %feature("kwargs") subsamplerate;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") subSampRate;
#endif
 public:
  PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
				       gsl_vector* prototype, unsigned M = 256, unsigned m = 3,
				       const String& nm = "PerfectReconstructionFFTAnalysisBank");
  ~PerfectReconstructionFFTAnalysisBank();
  double polyphase(unsigned m, unsigned n) const;
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  unsigned fftLen() const;
  unsigned nBlocks() const;
  unsigned subsamplerate() const;

#ifdef ENABLE_LEGACY_BTK_API
  unsigned subSampRate() const;
#endif
};

class PerfectReconstructionFFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") PerfectReconstructionFFTAnalysisBankPtr;
 public:
  %extend {
    PerfectReconstructionFFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
					    gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r = 0,
					    const String& nm = "PerfectReconstructionFFTAnalysisBankFloat") {
      return new PerfectReconstructionFFTAnalysisBankPtr(new PerfectReconstructionFFTAnalysisBank(samp, prototype, M, m, r, nm));
    }

    PerfectReconstructionFFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PerfectReconstructionFFTAnalysisBank* operator->();
};


// ----- definition for class `PerfectReconstructionFFTSynthesisBank' -----
// 
%ignore PerfectReconstructionFFTSynthesisBank;
class PerfectReconstructionFFTSynthesisBank : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") polyphase;
 public:
  PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBF");
  ~PerfectReconstructionFFTSynthesisBank();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();
  double polyphase(unsigned m, unsigned n) const;
};

class PerfectReconstructionFFTSynthesisBankPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") PerfectReconstructionFFTSynthesisBankPtr;
public:
  %extend {
    PerfectReconstructionFFTSynthesisBankPtr(VectorComplexFeatureStreamPtr& samp,
					     gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					     const String& nm = "PerfectReconstructionFFTSynthesisBank") {
      return new PerfectReconstructionFFTSynthesisBankPtr(new PerfectReconstructionFFTSynthesisBank(samp, prototype, M, m, r, nm));
    }

    PerfectReconstructionFFTSynthesisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PerfectReconstructionFFTSynthesisBank* operator->();
};

// ----- definition for class `DelayFeature' -----
//
%ignore DelayFeature;
class DelayFeature : public VectorComplexFeatureStream {
  %feature("kwargs") set_time_delay;
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  DelayFeature( const VectorComplexFeatureStreamPtr& samp, float time_delay=0.0, const String& nm = "DelayFeature");
  ~DelayFeature();
  void set_time_delay(double time_delay);
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
};

class DelayFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") DelayFeaturePtr;
public:
  %extend {
    DelayFeaturePtr( const VectorComplexFeatureStreamPtr& samp, float time_delay=0.0, const String& nm = "DelayFeature" ) {
      return new DelayFeaturePtr(new DelayFeature(samp, time_delay, nm));
    }

    DelayFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DelayFeature* operator->();
};


// ----- definition for class `CosineModulatedPrototypeDesign' -----
// 
class CosineModulatedPrototypeDesign {
  %feature("kwargs") fcn;
  %feature("kwargs") grad;
  %feature("kwargs") M;
  %feature("kwargs") N;
  %feature("kwargs") m;
  %feature("kwargs") J;
  %feature("kwargs") proto;
 public:
  CosineModulatedPrototypeDesign(int M = 256, int N = 3072, double fs = 1.0);
  ~CosineModulatedPrototypeDesign();
  void fcn(const double* x, double* f);
  void grad(const double* x, double* g);
  int M() const;
  int N() const;
  int m() const;
  int J() const;
  // return (one-half) of the prototype filter
  const gsl_vector* proto();
};

%feature("kwargs") design_f;
%feature("kwargs") design_df;
%feature("kwargs") design_fdf;
double design_f(const gsl_vector* v, void* params);
void   design_df(const gsl_vector* v, void* params, gsl_vector* df);
void   design_fdf(const gsl_vector* v, void* params, double* f, gsl_vector* df);

%feature("kwargs") write_gsl_format;
void write_gsl_format(const String& fileName, const gsl_vector* prototype);


// ----- definition for class `AnalysisOversampledDFTDesign' -----
//
%ignore AnalysisOversampledDFTDesign;
class AnalysisOversampledDFTDesign {
  %feature("kwargs") design;
  %feature("kwargs") save;
  %feature("kwargs") calcError;
public:
  AnalysisOversampledDFTDesign(unsigned M = 256, unsigned m = 4, unsigned r = 1, double wp = 1.0, int tau_h = -1 );
  ~AnalysisOversampledDFTDesign();
  // design prototype
  // tolerance for double precision: 2.2204e-16
  // tolerance for single precision: 1.1921e-07
  const gsl_vector* design(double tolerance = 2.2204e-16);
  void save(const String& fileName);
  // calculate distortion measures
  const gsl_vector* calcError(bool doPrint = true);
};

class AnalysisOversampledDFTDesignPtr {
  %feature("kwargs") AnalysisOversampledDFTDesignPtr;
public:
  %extend {
    AnalysisOversampledDFTDesignPtr(unsigned M = 256, unsigned m = 4, unsigned r = 1, double wp = 1.0, int tau_h = -1 ) {
      return new AnalysisOversampledDFTDesignPtr(new AnalysisOversampledDFTDesign(M, m, r, wp, tau_h ));
    }
  }

  AnalysisOversampledDFTDesign* operator->();
};


// ----- definition for class `SynthesisOversampledDFTDesign' -----
//
%ignore SynthesisOversampledDFTDesign;
class SynthesisOversampledDFTDesign {
  %feature("kwargs") design;
  %feature("kwargs") save;
  %feature("kwargs") calcError;
public:
  SynthesisOversampledDFTDesign(const gsl_vector* h, unsigned M = 256, unsigned m = 4, unsigned r = 1, double v = 1.0, double wp = 1.0, int tau_T = -1 );
  ~SynthesisOversampledDFTDesign();
  // design prototype
  // tolerance for double precision: 2.2204e-16
  // tolerance for single precision: 1.1921e-07
  const gsl_vector* design(double tolerance = 2.2204e-16);
  void save(const String& fileName);
  // calculate distortion measures
  const gsl_vector* calcError(bool doPrint = true);
};

class SynthesisOversampledDFTDesignPtr {
  %feature("kwargs") SynthesisOversampledDFTDesignPtr;
public:
  %extend {
    SynthesisOversampledDFTDesignPtr(const gsl_vector* h, unsigned M = 256, unsigned m = 4, unsigned r = 1, double v = 1.0, double wp = 1.0, int tau_T = -1 ) {
      return new SynthesisOversampledDFTDesignPtr(new SynthesisOversampledDFTDesign(h, M, m, r, v, wp, tau_T));
    }
  }

  SynthesisOversampledDFTDesign* operator->();
};

// Use tools/filterbank/DesignNyquistFilterBank.m for de Haan and Nyquist(M) filter prototype design. 
// The following functions will be obsoleted.
//
// ----- definition for class `AnalysisNyquistMDesign' -----
//
%ignore AnalysisNyquistMDesign;
class AnalysisNyquistMDesign : public AnalysisOversampledDFTDesign {
public:
  AnalysisNyquistMDesign(unsigned M = 512, unsigned m = 2, unsigned r = 1, double wp = 1.0);
  ~AnalysisNyquistMDesign();
};

class AnalysisNyquistMDesignPtr : public AnalysisOversampledDFTDesignPtr{
  %feature("kwargs") AnalysisNyquistMDesignPtr;
public:
  %extend {
    AnalysisNyquistMDesignPtr(unsigned M = 512, unsigned m = 2, unsigned r = 1, double wp = 1.0) {
      return new AnalysisNyquistMDesignPtr(new AnalysisNyquistMDesign(M, m, r, wp));
    }
  }

  AnalysisNyquistMDesign* operator->();
};


// ----- definition for class `SynthesisNyquistMDesign' -----
//
%ignore SynthesisNyquistMDesign;
class SynthesisNyquistMDesign : public SynthesisOversampledDFTDesign {
public:
  SynthesisNyquistMDesign(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
			  double wp = 1.0);
  ~SynthesisNyquistMDesign();
};

class SynthesisNyquistMDesignPtr : public SynthesisOversampledDFTDesignPtr {
  %feature("kwargs") SynthesisNyquistMDesignPtr;
public:
  %extend {
    SynthesisNyquistMDesignPtr(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
			       double wp = 1.0) {
      return new SynthesisNyquistMDesignPtr(new SynthesisNyquistMDesign(h, M, m, r, wp));
    }
  }

  SynthesisNyquistMDesign* operator->();
};


// ----- definition for class `SynthesisNyquistMDesignCompositeResponse' -----
//
%ignore SynthesisNyquistMDesignCompositeResponse;
class SynthesisNyquistMDesignCompositeResponse : public SynthesisNyquistMDesign {
public:
  SynthesisNyquistMDesignCompositeResponse(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
					   double wp = 1.0);
  ~SynthesisNyquistMDesignCompositeResponse();
};

class SynthesisNyquistMDesignCompositeResponsePtr : public SynthesisNyquistMDesignPtr {
  %feature("kwargs") SynthesisNyquistMDesignCompositeResponsePtr;
public:
  %extend {
    SynthesisNyquistMDesignCompositeResponsePtr(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
						double wp = 1.0) {
      return new SynthesisNyquistMDesignCompositeResponsePtr(new SynthesisNyquistMDesignCompositeResponse(h, M, m, r, wp));
    }
  }

  SynthesisNyquistMDesignCompositeResponse* operator->();
};

%feature("kwargs") get_window;
gsl_vector* get_window( unsigned winType, unsigned winLen );

%rename(__str__) print;
%ignore *::print();
