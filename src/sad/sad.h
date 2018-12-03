/**
 * @file sad.h
 * @brief Voice activity detection.
 * @author Kenichi Kumatani and John McDonough
 */

#ifndef SAD_H
#define SAD_H

#include <stdio.h>
#include <assert.h>
#define _LOG_SAD_
#ifdef _LOG_SAD_
#include <stdarg.h>
#endif /* _LOG_SAD_ */

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fit.h>

#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "sad/neural_spnsp_incl.h"


// ----- definition for abstract base class `NeuralNetVAD' -----
//
class NeuralNetVAD {
public:
  NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				  unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1, const String& neuralNetFile = "");
  ~NeuralNetVAD();

  bool next(int frame_no = -5);
  void reset();
  void read(const String& neuralNetFile);

private:
  void shift_down_();
  void increment_() { frame_no_++; }
  void update_buffer_(int frame_no);

  const VectorFloatFeatureStreamPtr			cep_;
  const int						frame_reset_no_;
  const unsigned					cepLen_;
  bool							is_speech_;
  int							frame_no_;
  unsigned						framesPadded_;
  unsigned						context_;
  unsigned						hiddenUnitsN_;
  unsigned						outputUnitsN_;
  float							threshold_;
  MLP*							mlp_;
  float**						frame_;
};

typedef refcount_ptr<NeuralNetVAD> NeuralNetVADPtr;


// ----- definition for abstract base class `VAD' -----
//
class VAD {
 public:
  VAD(VectorComplexFeatureStreamPtr& samp);
  ~VAD();

  virtual bool next(int frame_no = -5) = 0;
  virtual void reset() { frame_no_ = frame_reset_no_; }
  const gsl_vector_complex* frame() const { return frame_; }
  virtual void next_speaker() = 0;

 protected:
  void increment_() { frame_no_++; }

  const VectorComplexFeatureStreamPtr			samp_;
  const int						frame_reset_no_;
  const unsigned					fftLen_;
  bool							is_speech_;
  int							frame_no_;
  gsl_vector_complex*					frame_;
};

typedef refcount_ptr<VAD> VADPtr;


// ----- definition for class `EnergyVAD' -----
//
class SimpleEnergyVAD : public VAD {
 public:
  SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
			       double threshold, double gamma = 0.995);
  ~SimpleEnergyVAD();

  virtual bool next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
#endif

 private:
  const double						threshold_;
  const double						gamma_;
  double						spectral_energy_;
};

typedef Inherit<SimpleEnergyVAD, VADPtr> SimpleEnergyVADPtr;


// ----- definition for class `SimpleLikelihoodRatioVAD' -----
//
class SimpleLikelihoodRatioVAD : public VAD {
 public:
  SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp,
					double threshold, double alpha);
  ~SimpleLikelihoodRatioVAD();

  virtual bool next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  void set_variance(const gsl_vector* variance);

#ifdef ENABLE_LEGACY_BTK_API__
  void setVariance(const gsl_vector* variance);
  void nextSpeaker(){ next_speaker(); }
#endif

 private:
  double calc_Ak_(double vk, double gammak, double Rk);

  bool							variance_set_;
  gsl_vector*						noise_variance_;
  gsl_vector*						prev_Ak_;
  gsl_vector_complex*					prev_frame_;
  double						threshold_;
  double						alpha_;
};

typedef Inherit<SimpleLikelihoodRatioVAD, VADPtr> SimpleLikelihoodRatioVADPtr;

// ----- definition for class `EnergyVADFeature' -----
//
class EnergyVADFeature : public VectorFloatFeatureStream {
 public:
  EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 30, unsigned energiesN = 200, const String& nm = "Energy VAD");
  virtual ~EnergyVADFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
#endif

 private:
  static int comparator_(const void* elem1, const void* elem2);
  virtual bool above_threshold_(const gsl_vector_float* vector);

  VectorFloatFeatureStreamPtr			source_;
  bool						recognizing_;

  gsl_vector_float**				buffer_;
  const unsigned				bufferLen_;
  int						bufferX_;
  unsigned					bufferedN_;

  unsigned					abovethresholdN_;
  unsigned					belowThresholdN_;

  unsigned					energiesN_;
  double*					energies_;
  double*					sorted_energies_;
  const unsigned				medianX_;
};

typedef Inherit<EnergyVADFeature, VectorFloatFeatureStreamPtr> EnergyVADFeaturePtr;


// ----- definition for abstract base class `VADMetric' -----
//
class VADMetric :  public Countable {
public:
  VADMetric();
  ~VADMetric();

  virtual double next(int frame_no = -5) = 0;
  virtual void reset() = 0;
  double score(){ return cur_score_;}
  virtual void next_speaker() = 0;

#ifdef  _LOG_SAD_
  bool openLogFile( const String & logfilename );
  int  writeLog( const char *format, ... );
  void closeLogFile();
  void initScore(){ _score=0.0; _scoreX=0; }
  void setScore(double score){ _score=score; _scoreX++; }
  double getAverageScore(){ if(_scoreX==0){return 0;} return(_score/_scoreX); }

  int   frame_no_;
#endif /* _LOG_SAD_ */

protected:
  double                                        cur_score_;

#ifdef  _LOG_SAD_
private:
  FILE *logfp_;
  double _score;
  unsigned _scoreX;
#endif /* _LOG_SAD_ */
};

typedef refcountable_ptr<VADMetric> VADMetricPtr;


// ----- definition for class `EnergyVADMetric' -----
//
class EnergyVADMetric : public VADMetric {
public:
  EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4,
		  unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric");
  
  ~EnergyVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double energy_percentile(double percentile = 50.0) const;

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
  double energyPercentile(double percentile = 50.0){ return energy_percentile(percentile); }
#endif

private:
  static int comparator_(const void* elem1, const void* elem2);
  virtual bool above_threshold_(const gsl_vector_float* vector);

  VectorFloatFeatureStreamPtr			source_;
  const double					initial_energy_;

  const unsigned				headN_;
  const unsigned				tailN_;
  bool						recognizing_;

  unsigned					abovethresholdN_;
  unsigned					belowThresholdN_;

  unsigned					energiesN_;
  double*					energies_;
  double*					sorted_energies_;
  const unsigned				medianX_;
};

typedef Inherit<EnergyVADMetric, VADMetricPtr> EnergyVADMetricPtr;

// ----- definition for class `MultiChannelVADMetric' -----
//
template <typename ChannelType>
class MultiChannelVADMetric : public VADMetric {
 public:
  MultiChannelVADMetric(unsigned fftLen,
			double sampleRate, double lowCutoff, double highCutoff, const String& nm);
  ~MultiChannelVADMetric();

  void set_channel(ChannelType& chan);

#ifdef ENABLE_LEGACY_BTK_API__
  void setChannel(ChannelType& chan){ set_channel(chan); }
#endif

protected:
  unsigned set_lowX_(double lowCutoff) const;
  unsigned set_highX_(double highCutoff) const;
  unsigned set_binN_() const;

  typedef list<ChannelType>      	ChannelList_;
  ChannelList_				_channelList;

  const unsigned			fftLen_;
  const unsigned			fftLen2_;
  const double				samplerate_;
  const unsigned			lowX_;
  const unsigned			highX_;
  const unsigned			binN_;
  FILE                                 *logfp_;
};

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm);
template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm);
template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::~MultiChannelVADMetric();
template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::~MultiChannelVADMetric();
template<> void MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::set_channel(VectorFloatFeatureStreamPtr& chan);
template<> void MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::set_channel(VectorComplexFeatureStreamPtr& chan);

typedef MultiChannelVADMetric<VectorFloatFeatureStreamPtr>   FloatMultiChannelVADMetric;
typedef MultiChannelVADMetric<VectorComplexFeatureStreamPtr> ComplexMultiChannelVADMetric;
typedef refcountable_ptr<FloatMultiChannelVADMetric>         FloatMultiChannelVADMetricPtr;
typedef refcountable_ptr<ComplexMultiChannelVADMetric>       ComplexMultiChannelVADMetricPtr;

// ----- definition for class `PowerSpectrumVADMetric' -----
//
/**
   @brief detect voice activity based on the energy comparison
   @usage
   1. construct the object with PowerSpectrumVADMetric().
   2. set the channel data with set_channel().
   3. call next().
   @note the first channel is associated with the target speaker.
 */
class PowerSpectrumVADMetric : public FloatMultiChannelVADMetric {
public:
  PowerSpectrumVADMetric(unsigned fftLen,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");
  PowerSpectrumVADMetric(VectorFloatFeatureStreamPtr& source1, VectorFloatFeatureStreamPtr& source2,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");

  ~PowerSpectrumVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  gsl_vector *get_metrics() const { return powerList_;}
  void set_E0( double E0 ){ E0_ = E0;}
  void clear_channel();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
  gsl_vector *getMetrics(){ return get_metrics(); }
  void setE0( double E0 ){ set_E0(E0); }
  void clearChannel(){ clear_channel(); }
#endif

protected:
  typedef ChannelList_::iterator		ChannelIterator_;
  gsl_vector                                   *powerList_;
  double                                        E0_;
};

typedef Inherit<PowerSpectrumVADMetric, FloatMultiChannelVADMetricPtr> PowerSpectrumVADMetricPtr;

// ----- definition for class `NormalizedEnergyMetric' -----
//

/**
   @class

   @usage
   @note
*/

class NormalizedEnergyMetric : public  PowerSpectrumVADMetric {
  static double initial_energy_;
public:
  NormalizedEnergyMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "TSPS VAD Metric");

  ~NormalizedEnergyMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
};

typedef Inherit<NormalizedEnergyMetric, PowerSpectrumVADMetricPtr> NormalizedEnergyMetricPtr;

// ----- definition for class `CCCVADMetric' -----
//

/**
   @class compute cross-correlation coefficients (CCC) as a function of time delays,
          and average up the n-best values.
   @usage
   @note
*/

class CCCVADMetric : public ComplexMultiChannelVADMetric {
public:
  CCCVADMetric(unsigned fftLen, unsigned nCand,
	       double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
	       const String& nm = "CCC VAD Metric");
  ~CCCVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  void set_NCand(unsigned nCand);
  void set_threshold(double threshold){ threshold_ = threshold;}
  gsl_vector *get_metrics() const { return ccList_;}
  void clear_channel();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
  void setNCand(unsigned nCand){ set_NCand(nCand); }
  void setThreshold(double threshold){ set_threshold(threshold); }
  gsl_vector *getMetrics(){ return get_metrics(); }
  void clearChannel(){ clear_channel(); }
#endif

protected:
  typedef ChannelList_::iterator		ChannelIterator_;
  unsigned                                      nCand_;
  gsl_vector                                   *ccList_;
  gsl_vector_int                               *sample_delays_;
  double                                        threshold_;
  double                                       *pack_cross_spectrum_;
};

typedef Inherit<CCCVADMetric, ComplexMultiChannelVADMetricPtr> CCCVADMetricPtr;

// ----- definition for class `TSPSVADMetric' -----
//

/**
   @class

   @usage
   @note
*/

class TSPSVADMetric : public  PowerSpectrumVADMetric {
  static double initial_energy_;
public:
  TSPSVADMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "TSPS VAD Metric");
  ~TSPSVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
};

typedef Inherit<TSPSVADMetric,PowerSpectrumVADMetricPtr> TSPSVADMetricPtr;

// ----- definition for class `NegentropyVADMetric' -----
//
class NegentropyVADMetric : public VADMetric {

protected:
  class ComplexGeneralizedGaussian_ : public Countable {
  public:
    ComplexGeneralizedGaussian_(double shapeFactor = 2.0);
    double logLhood(gsl_complex X, double scaleFactor) const;
    double shapeFactor() const { return shape_factor_; }
    double Bc() const { return Bc_; }
    double normalization() const { return normalization_; }

  protected:
    virtual double calc_Bc_() const;
    virtual double calc_normalization_() const;

    const double				shape_factor_;
    /* const */ double				Bc_;
    /* const */ double				normalization_;
  };

  typedef refcountable_ptr<ComplexGeneralizedGaussian_> ComplexGeneralizedGaussianPtr_;

  typedef list<ComplexGeneralizedGaussianPtr_>	GaussianList_;
  typedef GaussianList_::iterator		GaussianListIterator_;
  typedef GaussianList_::const_iterator		GaussianListConstIterator_;

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
  void nextSpeaker(){ next_speaker(); }
  double calcNegentropy(int frame_no){ return calc_negentropy(frame_no); }
#endif

protected:
  virtual bool above_threshold_(int frame_no);
  unsigned set_lowX_(double lowCutoff) const;
  unsigned set_highX_(double highCutoff) const;
  unsigned set_binN_() const;

  VectorComplexFeatureStreamPtr			source_;
  VectorFloatFeatureStreamPtr			spectral_estimator_;

  GaussianList_					generalized_gaussians_;
  ComplexGeneralizedGaussianPtr_		gaussian_;

  const double					threshold_;

  const unsigned				fftLen_;
  const unsigned				fftLen2_;

  const double					samplerate_;
  const unsigned				lowX_;
  const unsigned				highX_;
  const unsigned				binN_;
};

typedef Inherit<NegentropyVADMetric, VADMetricPtr> NegentropyVADMetricPtr;


// ----- definition for class `MutualInformationVADMetric' -----
//
class MutualInformationVADMetric : public NegentropyVADMetric {
 public:
  MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1,
                             const VectorComplexFeatureStreamPtr& source2,
                             const VectorFloatFeatureStreamPtr& spectralEstimator1,
                             const VectorFloatFeatureStreamPtr& spectralEstimator2,
                             const String& shapeFactorFileName = "",
                             double twiddle = -1.0,
                             double threshold = 1.3,
                             double beta = 0.95,
                             double sampleRate = 16000.0,
                             double lowCutoff = 187.0,
                             double highCutoff = 1000.0,
                             const String& nm = "Mutual Information VAD Metric");
  ~MutualInformationVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double calc_mutual_information(int frame_no);

#ifdef ENABLE_LEGACY_BTK_API__
  double calcMutualInformation(int frame_no){ return calc_mutual_information(frame_no); }
#endif

 protected:
  class JointComplexGeneralizedGaussian_ : public NegentropyVADMetric::ComplexGeneralizedGaussian_ {
 public:
   JointComplexGeneralizedGaussian_(const NegentropyVADMetric::ComplexGeneralizedGaussianPtr_& ggaussian);
   ~JointComplexGeneralizedGaussian_();

   double logLhood(gsl_complex X1, gsl_complex X2, double scaleFactor1, double scaleFactor2, gsl_complex rho12) const;

 private:
   static const double      sqrt_two_;
   static const gsl_complex complex_one_;
   static const gsl_complex complex_zero_;

   virtual double calc_Bc_() const;
   virtual double calc_normalization_() const;

   double lngamma_ratio_(double f) const;
   double lngamma_ratio_joint_(double f) const;
   double match_(double f) const;

   double match_score_marginal_(double f) const;
   double match_score_joint_(double fJ) const;

   static const double				tolerance_;

   gsl_vector_complex*				X_;
   gsl_vector_complex*				scratch_;
   gsl_matrix_complex*				SigmaX_inverse_;
 };

 protected:
 typedef refcountable_ptr<JointComplexGeneralizedGaussian_> JointComplexGeneralizedGaussianPtr_;

 typedef list<JointComplexGeneralizedGaussianPtr_>	JointGaussianList_;
 typedef JointGaussianList_::iterator			JointGaussianListIterator_;
 typedef JointGaussianList_::const_iterator		JointGaussianListConstIterator_;

 typedef vector<gsl_complex>				CrossCorrelationVector_;

 static const double				epsilon_;

 virtual bool above_threshold_(int frame_no);
 double calc_fixed_threshold_();
 double calc_total_threshold_() const;
 void initialize_pdfs_();

 VectorComplexFeatureStreamPtr			source2_;
 VectorFloatFeatureStreamPtr			spectral_estimator2_;

 JointGaussianList_				joint_generalized_gaussians_;
 CrossCorrelationVector_			ccs_; // cross correlations
 const double					fixed_threshold_;
 const double					twiddle_;
 const double					threshold_;
 const double					beta_;
};

typedef Inherit<MutualInformationVADMetric, NegentropyVADMetricPtr> MutualInformationVADMetricPtr;


// ----- definition for class `LikelihoodRatioVADMetric' -----
//
class LikelihoodRatioVADMetric : public NegentropyVADMetric {
 public:
  LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName = "", double threshold = 0.0,
			   double sampleRate = 16000.0, double lowCutoff = 187.0, double highCutoff = 1000.0,
			   const String& nm = "Likelihood VAD Metric");
  ~LikelihoodRatioVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  double calc_likelihood_ratio(int frame_no);

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
  double calcLikelihoodRatio(int frame_no){ return calc_likelihood_ratio(frame_no); }
#endif

 private:
  const VectorComplexFeatureStreamPtr			source2_;
  const VectorFloatFeatureStreamPtr			spectral_estimator2_;
};

typedef Inherit<LikelihoodRatioVADMetric, NegentropyVADMetricPtr> LikelihoodRatioVADMetricPtr;


// ----- definition for class `LowFullBandEnergyRatioVADMetric' -----
//
class LowFullBandEnergyRatioVADMetric : public VADMetric {
public:
  LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric");
  ~LowFullBandEnergyRatioVADMetric();

  virtual double next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
#endif

private:
  virtual bool above_threshold_(int frame_no);
  void calc_auto_correlation_vector_(int frame_no);
  void calc_covariance_matrix_();
  double calc_lower_band_energy_();

  VectorFloatFeatureStreamPtr			source_;
  const unsigned				_lagsN;
  gsl_vector*					_lowpass;
  gsl_vector*					scratch_;
  double*					_autocorrelation;
  gsl_matrix*					_covariance;
};

typedef Inherit<LowFullBandEnergyRatioVADMetric, VADMetricPtr> LowFullBandEnergyRatioVADMetricPtr;


// ----- definition for class `HangoverVADFeature' -----
//
class HangoverVADFeature : public VectorFloatFeatureStream {
 public:
  HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
		     unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");
  virtual ~HangoverVADFeature();

  virtual const gsl_vector_float* next(int frame_no = -5);
  virtual void reset();
  virtual void next_speaker();
  int prefixN() const { return prefixN_ - headN_; }

#ifdef ENABLE_LEGACY_BTK_API__
  void nextSpeaker(){ next_speaker(); }
#endif

 protected:
  typedef pair<VADMetricPtr, double>		MetricPair_;
  typedef vector<MetricPair_>			MetricList_;
  typedef MetricList_::iterator			MetricListIterator_;
  typedef MetricList_::const_iterator		MetricListConstIterator_;

  static const unsigned EnergyVADMetricX		= 0;
  static const unsigned MutualInformationVADMetricX	= 1;
  static const unsigned LikelihoodRatioVADMetricX	= 2;

  static int comparator_(const void* elem1, const void* elem2);
  virtual bool above_threshold_(int frame_no);

  VectorFloatFeatureStreamPtr			source_;
  bool						recognizing_;

  gsl_vector_float**				buffer_;
  const unsigned				headN_;
  const unsigned				tailN_;
  int						bufferX_;
  unsigned					bufferedN_;

  unsigned					abovethresholdN_;
  unsigned					belowThresholdN_;
  unsigned					prefixN_;

  MetricList_					metricList_;
};

typedef Inherit<HangoverVADFeature, VectorFloatFeatureStreamPtr> HangoverVADFeaturePtr;


// ----- definition for class `HangoverMIVADFeature' -----
//
class HangoverMIVADFeature : public HangoverVADFeature {
  static const unsigned EnergyVADMetricX		= 0;
  static const unsigned MutualInformationVADMetricX	= 1;
  static const unsigned PowerVADMetricX			= 2;

 public:
  HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		       const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		       double energythreshold = 0.5, double mutualInformationThreshold = 0.5, double powerThreshold = 0.5,
		       unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");

  int decision_metric() const { return decision_metric_; }

#ifdef ENABLE_LEGACY_BTK_API__
  int decisionMetric() { return decision_metric(); }
#endif

protected:
  virtual bool above_threshold_(int frame_no);

  int						decision_metric_;
};

typedef Inherit<HangoverMIVADFeature, HangoverVADFeaturePtr> HangoverMIVADFeaturePtr;


// ----- definition for class `HangoverMultiStageVADFeature' -----
//
class HangoverMultiStageVADFeature : public HangoverVADFeature {
 public:
  HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
			       const VADMetricPtr& energyMetric, double energythreshold = 0.5, 
			       unsigned headN = 4, unsigned tailN = 10, const String& nm = "HangoverMultiStageVADFeature");
  ~HangoverMultiStageVADFeature();
  int decision_metric() const { return decision_metric_; }
  void set_metric( const VADMetricPtr& metricPtr, double threshold ){
    metricList_.push_back( MetricPair_( metricPtr, threshold ) );
  }

#ifdef ENABLE_LEGACY_BTK_API__
  int decisionMetric() { return decision_metric(); }
  void setMetric( const VADMetricPtr& metricPtr, double threshold ){ set_metric(metricPtr, threshold); }
#endif

#ifdef _LOG_SAD_
  void initScores();
  gsl_vector *getScores();
#endif /* _LOG_SAD_ */

protected:
  virtual bool above_threshold_(int frame_no);
  int          decision_metric_;
#ifdef _LOG_SAD_
  gsl_vector *_scores;
#endif /* _LOG_SAD_ */
};

typedef Inherit<HangoverMultiStageVADFeature, HangoverVADFeaturePtr> HangoverMultiStageVADFeaturePtr;

#endif
