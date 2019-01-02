/**
 * @file sad.i
 * @brief Voice activity detection.
 * @author Kenichi Kumatani and John McDonough
 */

%module(package="btk20") sad

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include "feature/feature.h"
#include <numpy/arrayobject.h>
#include "sad/sad.h"
#include "sad/sad_feature.h"
#include "sad/ica.h"
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


// ----- definition for class `NeuralNetVAD' -----
//
class NeuralNetVAD {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") read;
public:
  NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				  unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1,
				  const String& neuralNetFile = "");
  ~NeuralNetVAD();

  bool next(int frame_no = -5) = 0;
  void reset();
  void read(const String& neuralNetFile);
};

class NeuralNetVADPtr {
  %feature("kwargs") NeuralNetVADPtr;
 public:
  %extend {
    NeuralNetVADPtr(VectorFloatFeatureStreamPtr& cep,
				       unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1,
				       const String& neuralNetFile = "") {
      return new NeuralNetVADPtr(new NeuralNetVAD(cep, context, hiddenUnitsN, outputUnitsN, threshold, neuralNetFile));
    }

    NeuralNetVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NeuralNetVAD* operator->();
};


// ----- definition for class `VAD' -----
//
%ignore VAD;
class VAD {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") frame;
public:
  VAD(VectorFloatFeatureStreamPtr& samp)
    : _samp(samp) { }
  ~VAD();

  virtual bool next() = 0;
  virtual void reset() = 0;
  const gsl_vector_complex* frame() const;
  virtual void next_speaker() = 0;
};

class VADPtr {
  %feature("kwargs") VADPtr;
 public:
  %extend {
    VADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VAD* operator->();
};


// ----- definition for class `SimpleEnergyVAD' -----
//
%ignore SimpleEnergyVAD;
class SimpleEnergyVAD : public VAD {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
                  double threshold, double gamma = 0.995);
  ~SimpleEnergyVAD();

  bool next();
  virtual void reset();
  virtual void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
#endif
};

class SimpleEnergyVADPtr : public VADPtr {
  %feature("kwargs") SimpleEnergyVADPtr;
 public:
  %extend {
    SimpleEnergyVADPtr(VectorComplexFeatureStreamPtr& samp,
                       double threshold, double gamma = 0.98) {
      return new SimpleEnergyVADPtr(new SimpleEnergyVAD(samp, threshold, gamma));
    }

    SimpleEnergyVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SimpleEnergyVAD* operator->();
};


// ----- definition for class `SimpleLikelihoodRatioVAD' -----
//
%ignore SimpleLikelihoodRatioVAD;
class SimpleLikelihoodRatioVAD : public VAD {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_variance;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setVariance;
#endif
public:
  SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp, double threshold = 0.0, double alpha = 0.99);
  ~SimpleLikelihoodRatioVAD();

  bool next();
  virtual void reset();
  virtual void next_speaker();
  void set_variance(const gsl_vector* variance);

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  void setVariance(const gsl_vector* variance);
#endif
};

class SimpleLikelihoodRatioVADPtr : public VADPtr {
  %feature("kwargs") SimpleLikelihoodRatioVADPtr;
 public:
  %extend {
    SimpleLikelihoodRatioVADPtr(VectorComplexFeatureStreamPtr& samp,
                                double threshold = 0.0, double alpha = 0.99) {
      return new SimpleLikelihoodRatioVADPtr(new SimpleLikelihoodRatioVAD(samp, threshold, alpha));
    }

    SimpleLikelihoodRatioVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SimpleLikelihoodRatioVAD* operator->();
};


// ----- definition for class `EnergyVADFeature' -----
// 
%ignore EnergyVADFeature;
class EnergyVADFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 30, unsigned energiesN = 200, const String& nm = "Energy VAD");

  virtual const gsl_vector_float* next() const;
  void reset();
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
#endif
};

class EnergyVADFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") EnergyVADFeaturePtr;
 public:
  %extend {
    EnergyVADFeaturePtr(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 40, unsigned energiesN = 200, const String& nm = "Hamming") {
      return new EnergyVADFeaturePtr(new EnergyVADFeature(source, threshold, bufferLength, energiesN, nm));
    }

    EnergyVADFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyVADFeature* operator->();
};


// ----- definition for abstract base class `VADMetric' -----
//
%ignore VADMetric;
class VADMetric :  public Countable {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") score;
  %feature("kwargs") next_speaker;
public:
  VADMetric();
  ~VADMetric();

  virtual double next(int frame_no = -5) = 0;
  virtual void reset() = 0;
  double score();
  virtual void next_speaker() = 0;

#ifdef  _LOG_SAD_
  bool openLogFile( const String & logfilename );
  int  writeLog( const char *format, ... );
  void closeLogFile();
  void initScore();
  void setScore( double score);
  double getAverageScore();
#endif /* _LOG_SAD_ */
};

class VADMetricPtr {
  %feature("kwargs") VADMetricPtr;
 public:
  %extend {
    VADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VADMetric* operator->();
};

// ----- definition for class `EnergyVADMetric' -----
//
%ignore EnergyVADMetric;
class EnergyVADMetric : public VADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") energy_percentile;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") energyPercentile;
#endif
public:
  EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4, unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric");
  ~EnergyVADMetric();

  virtual double next(int frame_no = -5);
  virtual void   reset();
  virtual void next_speaker();
  double energy_percentile(double percentile = 50.0) const;

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  double energyPercentile(double percentile = 50.0) const;
#endif
};

class EnergyVADMetricPtr : public VADMetricPtr {
  %feature("kwargs") EnergyVADMetricPtr;
 public:
  %extend {
    EnergyVADMetricPtr(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4,
		       unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric")
    {
      return new EnergyVADMetricPtr(new EnergyVADMetric(source, initialEnergy, threshold, headN, tailN, energiesN, nm));
    }

    EnergyVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyVADMetric* operator->();
};


// ----- definition for class `MultiChannelVADMetric' -----
//
%ignore FloatMultiChannelVADMetric;
class FloatMultiChannelVADMetric : public VADMetric {
  %feature("kwargs") set_channel;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setChannel;
#endif
public:
  FloatMultiChannelVADMetric();
  ~FloatMultiChannelVADMetric();
  void set_channel(VectorFloatFeatureStreamPtr& chan);

#ifdef ENABLE_LEGACY_BTK_API__
  void setChannel(VectorFloatFeatureStreamPtr& chan);
#endif
};

class FloatMultiChannelVADMetricPtr : public VADMetricPtr {
  %feature("kwargs") FloatMultiChannelVADMetricPtr;
 public:
  %extend {
    FloatMultiChannelVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FloatMultiChannelVADMetricPtr* operator->();
};

// ----- definition for class `MultiChannelVADMetric' -----
//
%ignore ComplexMultiChannelVADMetric;
class ComplexMultiChannelVADMetric : public VADMetric {
  %feature("kwargs") set_channel;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setChannel;
#endif
public:
  ComplexMultiChannelVADMetric();
  ~ComplexMultiChannelVADMetric();
  void set_channel(VectorComplexFeatureStreamPtr& chan);

#ifdef ENABLE_LEGACY_BTK_API__
  void setChannel(VectorComplexFeatureStreamPtr& chan);
#endif
};

class ComplexMultiChannelVADMetricPtr : public VADMetricPtr {
  %feature("kwargs") ComplexMultiChannelVADMetricPtr;
 public:
  %extend {
    ComplexMultiChannelVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ComplexMultiChannelVADMetricPtr* operator->();
};

// ----- definition for class `PowerSpectrumVADMetric' -----
//
%ignore PowerSpectrumVADMetric;
class PowerSpectrumVADMetric : public FloatMultiChannelVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") get_metrics;
  %feature("kwargs") set_E0;
  %feature("kwargs") clear_channel;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") getMetrics;
  %feature("kwargs") setE0;
  %feature("kwargs") clearChannel;
#endif
public:
  PowerSpectrumVADMetric(unsigned fftLen,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");
  ~PowerSpectrumVADMetric();
  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  gsl_vector *get_metrics() const;
  void set_E0( double E0 );
  void clear_channel();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  gsl_vector *getMetrics();
  void setE0( double E0 );
  void clearChannel();
#endif
};

class PowerSpectrumVADMetricPtr : public FloatMultiChannelVADMetricPtr {
  %feature("kwargs") PowerSpectrumVADMetricPtr;
 public:
  %extend {
    PowerSpectrumVADMetricPtr(unsigned fftLen,
			      double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			      const String& nm = "Power Spectrum VAD Metric")
    {
      return new PowerSpectrumVADMetricPtr(new PowerSpectrumVADMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    PowerSpectrumVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PowerSpectrumVADMetric* operator->();
};

// ----- definition for class `NormalizedEnergyMetric' -----
//
%ignore NormalizedEnergyMetric;
class NormalizedEnergyMetric : public PowerSpectrumVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_E0;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setE0;
#endif
public:
  NormalizedEnergyMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "NormalizedEnergyMetric");
  ~NormalizedEnergyMetric();
  virtual double next(int frame_no = -5);
  virtual void reset();
  void set_E0( double E0 );

#ifdef ENABLE_LEGACY_BTK_API__
  void setE0( double E0 );
#endif
};

class NormalizedEnergyMetricPtr : public PowerSpectrumVADMetricPtr {
  %feature("kwargs") NormalizedEnergyMetricPtr;
 public:
  %extend {
    NormalizedEnergyMetricPtr(unsigned fftLen,
		     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		     const String& nm = "NormalizedEnergyMetric")
    {
      return new NormalizedEnergyMetricPtr(new NormalizedEnergyMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    NormalizedEnergyMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedEnergyMetric* operator->();
};

// ----- definition for class `CCCVADMetric' -----
//
%ignore CCCVADMetric;
class CCCVADMetric : public ComplexMultiChannelVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_NCand;
  %feature("kwargs") set_threshold;
  %feature("kwargs") get_metrics;
  %feature("kwargs") clear_channel;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setNCand;
  %feature("kwargs") setThreshold;
  %feature("kwargs") getMetrics;
  %feature("kwargs") clearChannel;
#endif
public:
  CCCVADMetric(unsigned fftLen, unsigned nCand,
	       double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
	       const String& nm = "CCC VAD Metric");
  ~CCCVADMetric();
  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  void set_NCand(unsigned nCand);
  void set_threshold(double threshold);
  gsl_vector *get_metrics() const;
  void clear_channel();

#ifdef ENABLE_LEGACY_BTK_API__
  void setNCand(unsigned nCand);
  void setThreshold(double threshold);
  void nextSpeaker();
  gsl_vector *getMetrics();
  void clearChannel();
#endif
};

class CCCVADMetricPtr : public ComplexMultiChannelVADMetricPtr {
  %feature("kwargs") CCCVADMetricPtr;
 public:
  %extend {
    CCCVADMetricPtr(unsigned fftLen, unsigned nCand,
                    double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
                    const String& nm = "CCC VAD Metric")
      {
        return new CCCVADMetricPtr(new CCCVADMetric(fftLen, nCand, sampleRate, lowCutoff, highCutoff, nm));
      }

    CCCVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CCCVADMetric* operator->();
};

// ----- definition for class `TSPSVADMetric' -----
//
%ignore TSPSVADMetric;
class TSPSVADMetric : public PowerSpectrumVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") set_E0;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") setE0;
#endif
public:
  TSPSVADMetric(unsigned fftLen,
                double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
                const String& nm = "TSPS VAD Metric");

  ~TSPSVADMetric();
  virtual double next(int frame_no = -5);
  virtual void reset();
  void set_E0( double E0 );

#ifdef ENABLE_LEGACY_BTK_API__
  void setE0( double E0 );
#endif
};

class TSPSVADMetricPtr : public PowerSpectrumVADMetricPtr {
  %feature("kwargs") TSPSVADMetricPtr;
 public:
  %extend {
    TSPSVADMetricPtr(unsigned fftLen,
		     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		     const String& nm = "TSPS VAD Metric")
    {
      return new TSPSVADMetricPtr(new TSPSVADMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    TSPSVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  TSPSVADMetric* operator->();
};

// ----- definition for class `NegentropyVADMetric' -----
//
%ignore NegentropyVADMetric;
class NegentropyVADMetric : public VADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") calc_negentropy;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") calcNegentropy;
#endif
public:
  NegentropyVADMetric(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
		      const String& shapeFactorFileName = "", double threshold = 0.5,
		      double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		      const String& nm = "Negentropy VAD Metric");
  ~NegentropyVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double calc_negentropy(int frame_no);

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  double calcNegentropy(int frame_no);
#endif
};

class NegentropyVADMetricPtr : public VADMetricPtr {
  %feature("kwargs") NegentropyVADMetricPtr;
 public:
  %extend {
    NegentropyVADMetricPtr(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
			   const String& shapeFactorFileName = "", double threshold = 0.5,
			   double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			   const String& nm = "Negentropy VAD Metric")
    {
      return new NegentropyVADMetricPtr(new NegentropyVADMetric(source, spectralEstimator, shapeFactorFileName, threshold,
								sampleRate, lowCutoff, highCutoff, nm));
    }

    NegentropyVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NegentropyVADMetric* operator->();
};


// ----- definition for class `MutualInformationVADMetric' -----
//
%ignore MutualInformationVADMetric;
class MutualInformationVADMetric : public NegentropyVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") calc_mutual_information;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") calcMutualInformation;
#endif
public:
  MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			     const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			     const String& shapeFactorFileName = "", double twiddle = -1.0, double threshold = 1.3, double beta = 0.95,
			     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			     const String& nm = "Mutual Information VAD Metric");
  ~MutualInformationVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double calc_mutual_information(int frame_no);

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  double calcMutualInformation(int frame_no);
#endif
};

class MutualInformationVADMetricPtr : public NegentropyVADMetricPtr {
  %feature("kwargs") MutualInformationVADMetricPtr;
 public:
  %extend {
    MutualInformationVADMetricPtr(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
				  const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
				  const String& shapeFactorFileName = "", double twiddle = -1.0, double threshold = 1.3, double beta = 0.95,
				  double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
				  const String& nm = "Mutual Information VAD Metric")
    {
      return new MutualInformationVADMetricPtr(new MutualInformationVADMetric(source1, source2, spectralEstimator1, spectralEstimator2,
									      shapeFactorFileName, twiddle, threshold, beta,
									      sampleRate, lowCutoff, highCutoff, nm));
    }

    MutualInformationVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MutualInformationVADMetric* operator->();
};


// ----- definition for class `LikelihoodRatioVADMetric' -----
//
%ignore LikelihoodRatioVADMetric;
class LikelihoodRatioVADMetric : public NegentropyVADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") calc_likelihood_ratio;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") calcLikelihoodRatio;
#endif
public:
  LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName = "", double threshold = 1.0,
			   double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			   const String& nm = "Mutual Information VAD Metric");

  ~LikelihoodRatioVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double calc_likelihood_ratio(int frame_no);

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
  double calcLikelihoodRatio(int frame_no);
#endif
};

class LikelihoodRatioVADMetricPtr : public NegentropyVADMetricPtr {
  %feature("kwargs") LikelihoodRatioVADMetricPtr;
 public:
  %extend {
    LikelihoodRatioVADMetricPtr(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
				const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
				const String& shapeFactorFileName = "", double threshold = 0.0,
				double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
				const String& nm = "Mutual Information VAD Metric")
    {
      return new LikelihoodRatioVADMetricPtr(new LikelihoodRatioVADMetric(source1, source2, spectralEstimator1, spectralEstimator2,
									  shapeFactorFileName, threshold, sampleRate, lowCutoff, highCutoff, nm));
    }

    LikelihoodRatioVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LikelihoodRatioVADMetric* operator->();
};


// ----- definition for class `LowFullBandEnergyRatioVADMetric' -----
//
class LowFullBandEnergyRatioVADMetric : public VADMetric {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric");
  ~LowFullBandEnergyRatioVADMetric();
  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
#endif
};

class LowFullBandEnergyRatioVADMetricPtr : public VADMetricPtr {
  %feature("kwargs") LowFullBandEnergyRatioVADMetricPtr;
 public:
  %extend {
    LowFullBandEnergyRatioVADMetricPtr(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric")
    {
      return new LowFullBandEnergyRatioVADMetricPtr(new LowFullBandEnergyRatioVADMetric(source, lowpass, threshold, nm));
    }

    LowFullBandEnergyRatioVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LowFullBandEnergyRatioVADMetric* operator->();
};


// ----- definition for class `HangoverVADFeature' -----
// 
%ignore HangoverVADFeature;
class HangoverVADFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
		     unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");

  virtual const gsl_vector_float* next() const;
  void reset();
  virtual void next_speaker();
  int prefixN() const;

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker();
#endif
};

class HangoverVADFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") HangoverVADFeaturePtr;
 public:
  %extend {
    HangoverVADFeaturePtr(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
			  unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature")
    {
      return new HangoverVADFeaturePtr(new HangoverVADFeature(source, metric, threshold, headN, tailN, nm));
    }

    HangoverVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverVADFeature* operator->();
};


// ----- definition for class `HangoverMIVADFeature' -----
//
%ignore HangoverMIVADFeature;
class HangoverMIVADFeature : public HangoverVADFeature {
  %feature("kwargs") decision_metric;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") decisionMetric;
#endif
public:
  HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		       const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		       double energyThreshold = 0.5, double mutualInformationThreshold = 0.5, double powerThreshold = 0.5,
		       unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover MIVAD Feature");

  int decision_metric() const;

#ifdef ENABLE_LEGACY_BTK_API__
  int decisionMetric() const;
#endif
};

class HangoverMIVADFeaturePtr : public HangoverVADFeaturePtr {
  %feature("kwargs") HangoverMIVADFeaturePtr;
 public:
  %extend {
    HangoverMIVADFeaturePtr(const VectorFloatFeatureStreamPtr& source,
                            const VADMetricPtr& energyMetric,
                            const VADMetricPtr& mutualInformationMetric,
                            const VADMetricPtr& powerMetric,
                            double energyThreshold = 0.5,
                            double mutualInformationThreshold = 0.5,
                            double powerThreshold = 0.5,
                            unsigned headN = 4,
                            unsigned tailN = 10,
                            const String& nm = "Hangover MIVAD Feature")
    {
      return new HangoverMIVADFeaturePtr(new HangoverMIVADFeature(source, energyMetric, mutualInformationMetric, powerMetric, energyThreshold, mutualInformationThreshold, powerThreshold, headN, tailN, nm));
    }

    HangoverMIVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverMIVADFeature* operator->();
};


// ----- definition for class `HangoverMultiStageVADFeature' -----
//
%ignore HangoverMultiStageVADFeature;
class HangoverMultiStageVADFeature : public HangoverVADFeature {
  %feature("kwargs") decision_metric;
  %feature("kwargs") set_metric;
#ifdef ENABLE_LEGACY_BTK_API__
  %feature("kwargs") decisionMetric;
  %feature("kwargs") setMetric;
#endif
public:
  HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
                               const VADMetricPtr& energyMetric,
                               double energyThreshold = 0.5,
                               unsigned headN = 4,
                               unsigned tailN = 10,
                               const String& nm = "HangoverMultiStageVADFeature");

  int decision_metric() const;
  void set_metric( const VADMetricPtr& metricPtr, double threshold );

#ifdef ENABLE_LEGACY_BTK_API__
  int decisionMetric() const;
  void setMetric( const VADMetricPtr& metricPtr, double threshold );
#endif

#ifdef _LOG_SAD_
  void initScores();
  gsl_vector *getScores();
#endif /* _LOG_SAD_ */
};

class HangoverMultiStageVADFeaturePtr : public HangoverVADFeaturePtr {
  %feature("kwargs") HangoverMultiStageVADFeaturePtr;
 public:
  %extend {
    HangoverMultiStageVADFeaturePtr(const VectorFloatFeatureStreamPtr& source,
                                    const VADMetricPtr& energyMetric,
                                    double energyThreshold = 0.5,
                                    unsigned headN = 4,
                                    unsigned tailN = 10,
                                    const String& nm = "HangoverMultiStageVADFeature")
    {
      return new HangoverMultiStageVADFeaturePtr(new HangoverMultiStageVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm));
    }

    HangoverMultiStageVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverMultiStageVADFeature* operator->();
};


// ----- definition for class `BrightnessFeature' -----
// 
%ignore BrightnessFeature;
class BrightnessFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness");

  const gsl_vector_float* next() const;
};

class BrightnessFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") BrightnessFeaturePtr;
 public:
  %extend {
    BrightnessFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness") {
      return new BrightnessFeaturePtr(new BrightnessFeature(src, sampleRate, nm));
    }

    BrightnessFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BrightnessFeature* operator->();
};


// ----- definition for class `EnergyDiffusionFeature' -----
// 
%ignore EnergyDiffusionFeature;
class EnergyDiffusionFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Energy Diffusion");

  const gsl_vector_float* next() const;
};

class EnergyDiffusionFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") EnergyDiffusionFeaturePtr;
 public:
  %extend {
    EnergyDiffusionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Energy Diffusion") {
      return new EnergyDiffusionFeaturePtr(new EnergyDiffusionFeature(src, nm));
    }

    EnergyDiffusionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyDiffusionFeature* operator->();
};


// ----- definition for class `BandEnergyRatioFeature' -----
// 
%ignore BandEnergyRatioFeature;
class BandEnergyRatioFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class BandEnergyRatioFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") BandEnergyRatioFeaturePtr;
 public:
  %extend {
    BandEnergyRatioFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio") {
      return new BandEnergyRatioFeaturePtr(new BandEnergyRatioFeature(src, sampleRate, threshF, nm));
    }

    BandEnergyRatioFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BandEnergyRatioFeature* operator->();
};


// ----- definition for class `NormalizedFluxFeature' -----
// 
%ignore NormalizedFluxFeature;
class NormalizedFluxFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class NormalizedFluxFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") NormalizedFluxFeaturePtr;
 public:
  %extend {
    NormalizedFluxFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio") {
      return new NormalizedFluxFeaturePtr(new NormalizedFluxFeature(src, sampleRate, threshF, nm));
    }

    NormalizedFluxFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedFluxFeature* operator->();
};


// ----- definition for class `NegativeEntropyFeature' -----
// 
%ignore NegativeEntropyFeature;
class NegativeEntropyFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy");

  const gsl_vector_float* next() const;
};

class NegativeEntropyFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") NegativeEntropyFeaturePtr;
 public:
  %extend {
    NegativeEntropyFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy") {
      return new NegativeEntropyFeaturePtr(new NegativeEntropyFeature(src, nm));
    }

    NegativeEntropyFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NegativeEntropyFeature* operator->();
};


// ----- definition for class `SignificantSubbandsFeature' -----
// 
%ignore SignificantSubbandsFeature;
class SignificantSubbandsFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Significant Subbands");

  const gsl_vector_float* next() const;
};

class SignificantSubbandsFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") SignificantSubbandsFeaturePtr;
 public:
  %extend {
    SignificantSubbandsFeaturePtr(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Significant Subbands") {
      return new SignificantSubbandsFeaturePtr(new SignificantSubbandsFeature(src, thresh, nm));
    }

    SignificantSubbandsFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SignificantSubbandsFeature* operator->();
};


// ----- definition for class `NormalizedBandwidthFeature' -----
// 
%ignore NormalizedBandwidthFeature;
class NormalizedBandwidthFeature : public VectorFloatFeatureStream {
  %feature("kwargs") next;
public:
  NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class NormalizedBandwidthFeaturePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") NormalizedBandwidthFeaturePtr;
 public:
  %extend {
    NormalizedBandwidthFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio") {
      return new NormalizedBandwidthFeaturePtr(new NormalizedBandwidthFeature(src, sampleRate, thresh, nm));
    }

    NormalizedBandwidthFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedBandwidthFeature* operator->();
};


// ----- definition for class `PCA' -----
// 
%ignore PCA;
class PCA {
  %feature("kwargs") pca_svd;
  %feature("kwargs") pca_eigen;
public:
  PCA(unsigned dimN);
  ~PCA();

  void pca_svd(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);
  void pca_eigen(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);
};

class PCAPtr {
  %feature("kwargs") PCAPtr;
 public:
  %extend {
    PCAPtr(unsigned dimN) {
      return new PCAPtr(new PCA(dimN));
    }
  }

  PCA* operator->();
};


// ----- definition for class `FastICA' -----
// 
%ignore FastICA;
class FastICA {
  %feature("kwargs") deflation;
public:
  FastICA(unsigned dimN, unsigned maxIterN);
  ~FastICA();

  void deflation(gsl_matrix* data, gsl_matrix* B, gsl_matrix* A, gsl_matrix* W, gsl_matrix* M,
                 gsl_matrix* neg, double eps, int maxIterN);
};

class FastICAPtr {
  %feature("kwargs") FastICAPtr;
 public:
  %extend {
    FastICAPtr(unsigned dimN, unsigned maxIterN) {
      return new FastICAPtr(new FastICA(dimN, maxIterN));
    }
  }

  FastICA* operator->();
};
