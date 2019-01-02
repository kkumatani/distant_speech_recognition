/**
 * @file sad.cc
 * @brief Voice activity detection.
 * @author Kenichi Kumatani and John McDonough
 */

#include "sad/sad.h"
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>

// ----- methods for class `NeuralNetVAD' -----
//
NeuralNetVAD::
NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				unsigned context, unsigned hiddenUnitsN, unsigned outputUnitsN, float threshold,
				const String& neuralNetFile)
  : cep_(cep), frame_reset_no_(-1), cepLen_(cep_->size()),
    context_(context), hiddenUnitsN_(hiddenUnitsN), outputUnitsN_(outputUnitsN), threshold_(threshold),
    mlp_(Mlp_Param_Mem_Alloc(cepLen_, context_, hiddenUnitsN_, outputUnitsN_)),
    frame_(new float*[2*context_+1])
{
  for (unsigned rowX = 0; rowX < 2*context_+1; rowX++)
    frame_[rowX] = new float[cepLen_];

  if (neuralNetFile != "")
    Read_Mlp_Param(neuralNetFile.c_str(), mlp_, cepLen_, context_);
}

NeuralNetVAD::~NeuralNetVAD()
{
  Free_Mlp_Param_Mem(mlp_);

  for (unsigned rowX = 0; rowX < 2*context_+1; rowX++)
    delete[] frame_[rowX];
  delete[] frame_;
}

void NeuralNetVAD::reset()
{
  framesPadded_ = 0; frame_no_ = frame_reset_no_;  cep_->reset();
}

void NeuralNetVAD::read(const String& neuralNetFile)
{
  printf("Reading neural net from file \'%s\'.\n", neuralNetFile.c_str());  fflush(stdout);
  Read_Mlp_Param(neuralNetFile.c_str(), mlp_, cepLen_, context_);
}

void NeuralNetVAD::shift_down_()
{
  float* tmp = frame_[0];
  for (unsigned rowX = 0; rowX < 2 * context_; rowX++)
    frame_[rowX] = frame_[rowX+1];
  frame_[2*context_] = tmp;
}

void NeuralNetVAD::update_buffer_(int frame_no)
{
  shift_down_();

  if (framesPadded_ == 0) {			// normal processing
    try {

      const gsl_vector_float* cep = cep_->next(frame_no);
      memcpy(frame_[2*context_], cep->data, cepLen_ * sizeof(float));

    } catch  (jiterator_error& e) {
      memcpy(frame_[2*context_], frame_[2*context_ - 1], cepLen_ * sizeof(float));

      // printf("Padding frame %d.\n", framesPadded_);

      framesPadded_++;
    }

  } else if (framesPadded_ < context_) {	// repeat last frame

    memcpy(frame_[2*context_], frame_[2*context_ - 1], cepLen_ * sizeof(float));

    // printf("Padding frame %d.\n", framesPadded_);

    framesPadded_++;

  } else {					// end of utterance

    throw jiterator_error("end of samples!");

  }
}

// return true if current frame is speech
bool NeuralNetVAD::next(int frame_no)
{
  if (frame_no == frame_no_) return is_speech_;

  if (frame_no >= 0 && frame_no != frame_no_ + 1)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frame_no - 1, frame_no_);

  // "prime" the buffer
  if (frame_no_ == frame_reset_no_) {
    for (unsigned itnX = 0; itnX < context_; itnX++)
      update_buffer_(0);
    for (unsigned itnX = 0; itnX < context_; itnX++)
      update_buffer_(itnX);
  }

  increment_();
  update_buffer_(frame_no_ + context_);

  int nsp_flag;
  Neural_Spnsp_Det(frame_, cepLen_, context_, mlp_, threshold_, &nsp_flag);
  is_speech_ = (nsp_flag ? false : true);

  return is_speech_;
}


// ----- methods for class `VAD' -----
//
VAD::VAD(VectorComplexFeatureStreamPtr& samp)
  : samp_(samp), frame_reset_no_(-1), fftLen_(samp_->size()),
    frame_(gsl_vector_complex_alloc(fftLen_))
{
}

VAD::~VAD()
{
  gsl_vector_complex_free(frame_);
}


// ----- methods for class `SimpleEnergyVAD' -----
//
SimpleEnergyVAD::
SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
		double threshold, double gamma)
  : VAD(samp),
    threshold_(threshold), gamma_(gamma), spectral_energy_(0.0) { }

SimpleEnergyVAD::
~SimpleEnergyVAD() { }

void SimpleEnergyVAD::next_speaker()
{
  spectral_energy_ = 0.0;  reset();
}

// return true if current frame is speech
bool SimpleEnergyVAD::next(int frame_no)
{
  if (frame_no == frame_no_) return is_speech_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_complex* samp = samp_->next(frame_no);
  gsl_vector_complex_memcpy(frame_, samp);

  double currentEnergy = 0.0;
  for (unsigned k = 0; k < fftLen_; k++)
    currentEnergy += gsl_complex_abs2(gsl_vector_complex_get(samp, k));

  spectral_energy_ = gamma_ * spectral_energy_ + (1.0 - gamma_) * currentEnergy;

  is_speech_ = (currentEnergy / spectral_energy_) > threshold_;

  increment_();
  return is_speech_;
}

void SimpleEnergyVAD::reset()
{
  VAD::reset();  samp_->reset();
}


// ----- methods for class `SimpleLikelihoodRatioVAD' -----
//
SimpleLikelihoodRatioVAD::
SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp,
				      double threshold, double alpha)
  : VAD(samp),
    variance_set_(false),
    noise_variance_(gsl_vector_alloc(samp->size())),
    prev_Ak_(gsl_vector_alloc(samp->size())),
    prev_frame_(gsl_vector_complex_alloc(samp->size())),
    threshold_(threshold), alpha_(alpha)
{
  gsl_vector_complex_set_zero(prev_frame_);
}

SimpleLikelihoodRatioVAD::~SimpleLikelihoodRatioVAD()
{
  gsl_vector_free(noise_variance_);
  gsl_vector_free(prev_Ak_);
  gsl_vector_complex_free(prev_frame_);
}

void SimpleLikelihoodRatioVAD::next_speaker()
{
  variance_set_ = false;  reset();
}

void SimpleLikelihoodRatioVAD::set_variance(const gsl_vector* variance)
{
  // initialize 'Ak[n-1]' to the noise floor
  if (variance_set_ == false)
    for (unsigned k = 0; k < fftLen_; k++)
      gsl_vector_set(prev_Ak_, k, sqrt(gsl_vector_get(variance, k)));

  if (variance->size != samp_->size())
    throw jdimension_error("Variance and sample sizes (%d vs. %d) do not match.",
			   variance->size, samp_->size());

  gsl_vector_memcpy(noise_variance_, variance);
  variance_set_ = true;
}

double SimpleLikelihoodRatioVAD::calc_Ak_(double vk, double gammak, double Rk)
{
  return (sqrt(M_PI) / 2.0) * (sqrt(vk) / gammak) * gsl_sf_hyperg_1F1(-0.5, 1.0, -vk) * Rk;
}

// return true if current frame is speech
bool SimpleLikelihoodRatioVAD::next(int frame_no)
{
  if (variance_set_ == false)
    throw jconsistency_error("Must set noise variance before calling 'next()'.");

  if (frame_no == frame_no_) return is_speech_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_complex* samp = samp_->next(frame_no);
  gsl_vector_complex_memcpy(frame_, samp);

  double logLR = 0.0;
  for (unsigned k = 0; k < fftLen_; k++) {
    double Rk       = gsl_complex_abs(gsl_vector_complex_get(samp, k));
    double lambdaNk = gsl_vector_get(noise_variance_, k);
    double gammak   = Rk * Rk / lambdaNk;
    double prevAk   = gsl_vector_get(prev_Ak_, k);
    double xik      = alpha_ * (prevAk * prevAk / lambdaNk) + (1.0 - alpha_) * max(gammak - 1.0, 0.0);
    double vk       = (xik / (1.0 + xik)) * gammak;
    double Ak       = calc_Ak_(vk, gammak, Rk);

    gsl_vector_set(prev_Ak_, k, Ak);

    logLR += -log(1.0 + xik) + (gammak * xik / (1.0 + xik));
  }

  gsl_vector_complex_memcpy(prev_frame_, samp);

  is_speech_ = (logLR / fftLen_) > threshold_;

  increment_();
  return is_speech_;
}

void SimpleLikelihoodRatioVAD::reset()
{
  samp_->reset();
}

// -----------------------------------------------------------------
//
//  Implementation Notes:
//
//  This speech activity detector is based on [1], which makes
//  extensive references to the minimum mean square estimation
//  techniques reported in [2].
//
// References:
//
// [1] J. Sohn, N. S. Kim, W. Sung, "A statistical model-based
//     voice activity detection," IEEE Signal Processing Letters,
//     6(1), January, 1999.

// [2] Y. Ephraim, D. Malah, "Speech enhancement using a minimum
//     mean-square error short-time spectral amplitude estimator,"
//     IEEE Trans. Acoust. Speech Signal Proc., ASSP-32(6),
//     December, 1984.
//
// -----------------------------------------------------------------

// ----- Methods for class 'EnergyVADFeature' -----
//
EnergyVADFeature::
EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold, unsigned bufferLength, unsigned energiesN, const String& nm)
  : VectorFloatFeatureStream(source->size(), nm),
    source_(source), recognizing_(false),
    buffer_(NULL),
    bufferLen_(bufferLength), bufferX_(0), bufferedN_(bufferLen_),
    energiesN_(energiesN), energies_(new double[energiesN_]), medianX_(unsigned(threshold * energiesN_))
{
  buffer_ = new gsl_vector_float*[bufferLen_];
  for (unsigned n = 0; n < bufferLen_; n++)
    buffer_[n] = gsl_vector_float_calloc(source_->size());

  energies_       = new double[energiesN_];
  sorted_energies_ = new double[energiesN_];

  for (unsigned n = 0; n < energiesN_; n++)
    energies_[n] = HUGE;

  // printf("Median index %d\n", medianX_);
}

EnergyVADFeature::~EnergyVADFeature()
{
  for (unsigned n = 0; n < bufferLen_; n++)
    gsl_vector_float_free(buffer_[n]);

  delete[] buffer_;  delete[] energies_;  delete[] sorted_energies_;
}

void EnergyVADFeature::reset()
{
  /* source_->reset(); */ VectorFloatFeatureStream::reset();

  bufferedN_   = bufferX_ = abovethresholdN_ = belowThresholdN_ = 0;
  recognizing_ = false;
}

void EnergyVADFeature::next_speaker()
{
  for (unsigned n = 0; n < energiesN_; n++)
    energies_[n] = HUGE;
}

int EnergyVADFeature::comparator_(const void* elem1, const void* elem2)
{
  double* e1 = (double*) elem1;
  double* e2 = (double*) elem2;
  
  if (*e1 == *e2) return 0;
  if (*e1 <  *e2) return -1;
  return 1;
}

bool EnergyVADFeature::above_threshold_(const gsl_vector_float* vector)
{
  double sum = 0.0;
  for (unsigned n = 0; n < vector->size; n++) {
    double val = gsl_vector_float_get(vector, n);
    sum += val * val;
  }

  memcpy(sorted_energies_, energies_, energiesN_ * sizeof(double));
  qsort(sorted_energies_, energiesN_, sizeof(double), comparator_);

  if (recognizing_ == false && abovethresholdN_ == 0) {
    memmove(energies_, energies_ + 1, (energiesN_ - 1) * sizeof(double));
    energies_[energiesN_ - 1] = sum;
  }

  // printf("Threshold = %10.2f\n", sorted_energies_[medianX_]);

  return ((sum > sorted_energies_[medianX_]) ? true : false);
}

const gsl_vector_float* EnergyVADFeature::next(int frame_no)
{
  if (recognizing_) {

    // use up the buffered blocks
    if (bufferedN_ > 0) {
      const gsl_vector_float* vector = buffer_[bufferX_];
      bufferX_ = (bufferX_ + 1) % bufferLen_;
      bufferedN_--;
      return vector;
    }

    // buffer is empty; take blocks directly from source
    const gsl_vector_float* vector = source_->next();
    if (above_threshold_(vector)) {
      belowThresholdN_ = 0;
    } else {
      if (belowThresholdN_ == bufferLen_)
	throw jiterator_error("end of samples!");
      belowThresholdN_++;
    }
    return vector;

  } else {

    // buffer sample blocks until sufficient blocks have energy above the threshold
    while (true) {
      const gsl_vector_float* vector = source_->next();
      gsl_vector_float_memcpy(buffer_[bufferX_], vector);
      bufferX_ = (bufferX_ + 1) % bufferLen_;
      bufferedN_ = min(bufferLen_, bufferedN_ + 1);

      if (above_threshold_(vector)) {
	if (abovethresholdN_ == bufferLen_) {
	  recognizing_ = true;
	  vector = buffer_[bufferX_];
	  bufferX_ = (bufferX_ + 1) % bufferLen_;
	  bufferedN_--;
	  return vector;
	}
	abovethresholdN_++;
      } else {
	abovethresholdN_ = 0;
      }
    }
  }
}

// ----- definition for abstract base class `VADMetric' -----
//
VADMetric::
VADMetric()
{
#ifdef  _LOG_SAD_
  logfp_ = NULL;
  initScore();
#endif /* _LOG_SAD_ */
}

VADMetric::~VADMetric()
{
#ifdef  _LOG_SAD_
  closeLogFile();
#endif /* _LOG_SAD_ */
}

#ifdef  _LOG_SAD_
bool VADMetric::openLogFile( const String & logfilename )
{
  if( NULL != logfp_ ){
    printf("closing the previous log file\n");
    fclose(logfp_);
    return false;
  }
  logfp_ = fopen( logfilename.c_str(), "w" );
  return true;
}

int VADMetric::writeLog( const char *format, ... )
{
  if( NULL != logfp_ ){
    int ret;
    va_list args;

    va_start(args, format);
    ret = vfprintf( logfp_, format, args);
    va_end (args);

    return ret;
  }
  return 0;
}

void VADMetric::closeLogFile()
{
  if( NULL != logfp_ )
    fclose(logfp_);
}
#endif /* _LOG_SAD_ */


// ----- Methods for class 'EnergyVADMetric' -----
//
EnergyVADMetric::
EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy, double threshold, unsigned headN,
		unsigned tailN, unsigned energiesN, const String& nm)
  : source_(source), initial_energy_(initialEnergy), headN_(headN), tailN_(tailN), recognizing_(false),
    abovethresholdN_(0), belowThresholdN_(0), energiesN_(energiesN), energies_(new double[energiesN_]),
    medianX_(unsigned(threshold * energiesN_))
{
  energies_       = new double[energiesN_];
  sorted_energies_ = new double[energiesN_];

  for (unsigned n = 0; n < energiesN_; n++)
    energies_[n] = initial_energy_;

  // printf("Median index %d\n", medianX_);
}

EnergyVADMetric::~EnergyVADMetric()
{
  delete[] energies_;  delete[] sorted_energies_;
}

void EnergyVADMetric::reset()
{
  abovethresholdN_ = belowThresholdN_ = 0;
  recognizing_ = false;
}

void EnergyVADMetric::next_speaker()
{
  abovethresholdN_ = belowThresholdN_ = 0;
  recognizing_ = false;

  for (unsigned n = 0; n < energiesN_; n++)
    energies_[n] = initial_energy_;
}

int EnergyVADMetric::comparator_(const void* elem1, const void* elem2)
{
  double* e1 = (double*) elem1;
  double* e2 = (double*) elem2;

  if (*e1 == *e2) return 0;
  if (*e1 <  *e2) return -1;
  return 1;
}

bool EnergyVADMetric::above_threshold_(const gsl_vector_float* vector)
{
  double sum = 0.0;
  for (unsigned n = 0; n < vector->size; n++) {
    double val = gsl_vector_float_get(vector, n);
    sum += val * val;
  }

  memcpy(sorted_energies_, energies_, energiesN_ * sizeof(double));
  qsort(sorted_energies_, energiesN_, sizeof(double), comparator_);

  if (recognizing_ == false && abovethresholdN_ == 0) {
    memmove(energies_, energies_ + 1, (energiesN_ - 1) * sizeof(double));
    energies_[energiesN_ - 1] = sum;
  }

  // printf("Threshold = %12.4e\n", sorted_energies_[medianX_]);
  cur_score_ = sum;
#ifdef _LOG_SAD_
  writeLog( "%d %e\n", frame_no_, sum );
  setScore( sum );
#endif

  return ((sum > sorted_energies_[medianX_]) ? true : false);
}

double EnergyVADMetric::energy_percentile(double percentile) const
{
  if ( percentile < 0.0 ||  percentile > 100.0)
    throw jdimension_error("Percentile %g is out of range [0.0, 100.0].", percentile);

  memcpy(sorted_energies_, energies_, energiesN_ * sizeof(double));
  qsort(sorted_energies_, energiesN_, sizeof(double), comparator_);

  return sorted_energies_[int((percentile / 100.0) * energiesN_)] / energiesN_;
}

double EnergyVADMetric::next(int frame_no)
{
#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  const gsl_vector_float* vector = source_->next(frame_no);
  if (recognizing_) {

    if (above_threshold_(vector)) {
      belowThresholdN_ = 0;
      return 1.0;
    } else {
      belowThresholdN_++;
      if (belowThresholdN_ == tailN_) {
	recognizing_ = false;  abovethresholdN_ = 0;
      }
      return 0.0;
    }

  } else {

    if (above_threshold_(vector)) {
      abovethresholdN_++;
      if (abovethresholdN_ == headN_) {
	recognizing_ = true;  belowThresholdN_ = 0;
      }
      return 1.0;
    } else {
      abovethresholdN_ = 0;
      return 0.0;
    }
  }
}

// ----- definition for class `MultiChannelVADMetric' -----
//

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : fftLen_(fftLen), fftLen2_(fftLen / 2), samplerate_(sampleRate),
    lowX_(set_lowX_(lowCutoff)),
    highX_(set_highX_(highCutoff)),
    binN_(set_binN_()),
    logfp_(NULL)
{
}

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::~MultiChannelVADMetric()
{
  if( NULL != logfp_ )
    fclose( logfp_ );
}

template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : fftLen_(fftLen), fftLen2_(fftLen / 2), samplerate_(sampleRate),
    lowX_(set_lowX_(lowCutoff)),
    highX_(set_highX_(highCutoff)),
    binN_(set_binN_())
{
}

template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::~MultiChannelVADMetric()
{}

template<> void MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::set_channel(VectorFloatFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

template<> void MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::set_channel(VectorComplexFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::set_lowX_(double lowCutoff) const
{
  if (lowCutoff < 0.0) return 0;

  if (lowCutoff >= samplerate_ / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", lowCutoff);

  unsigned binX = (unsigned) ((lowCutoff / samplerate_) * fftLen_);

  printf("Setting lowest bin to %d\n", binX);

  return binX;
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::set_highX_(double highCutoff) const
{
  if (highCutoff < 0.0) return fftLen2_;
  
  if (highCutoff >= samplerate_ / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", highCutoff);

  unsigned binX = (unsigned) ((highCutoff / samplerate_) * fftLen_ + 0.5);

  printf("Setting highest bin to %d\n", binX);

  return binX;
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::set_binN_() const
{
  if (lowX_ > 0)
    return 2 * (highX_ - lowX_ + 1);

  return 2 * (highX_ - lowX_) + 1;
}

// ----- Methods for class 'PowerSpectrumVADMetric' -----
//
PowerSpectrumVADMetric::PowerSpectrumVADMetric(unsigned fftLen,
					       double sampleRate, double lowCutoff, double highCutoff,
					       const String& nm)
  : FloatMultiChannelVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm)
{
  powerList_ = NULL;
  E0_ = 1;
}

PowerSpectrumVADMetric::PowerSpectrumVADMetric(VectorFloatFeatureStreamPtr& source1, VectorFloatFeatureStreamPtr& source2,
					       double sampleRate, double lowCutoff, double highCutoff,
					       const String& nm)
  : FloatMultiChannelVADMetric( source1->size(), sampleRate, lowCutoff, highCutoff, nm)
{
  set_channel( source1 );
  set_channel( source2 );
  powerList_ = NULL;
  E0_ = 1;
}

PowerSpectrumVADMetric::~PowerSpectrumVADMetric()
{
  if( NULL != powerList_ )
    gsl_vector_free( powerList_ );
  powerList_ = NULL;
}

double PowerSpectrumVADMetric::next(int frame_no)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalpower = 0;
  double power_ratio = 0;

#ifdef  _LOG_SAD_ 
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  if( NULL == powerList_ )
    powerList_ = gsl_vector_alloc( chanN );
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frame_no);
    for (unsigned fbinX = lowX_; fbinX <= highX_; fbinX++) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= fftLen_;
    totalpower += power_n;
    gsl_vector_set( powerList_, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
  }

  power_ratio = gsl_vector_get( powerList_,0) / totalpower;

  if( power_ratio > (E0_/chanN) )
    return 1.0;
  else
    return -1.0;

  return 0.0;
}

void PowerSpectrumVADMetric::reset()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void PowerSpectrumVADMetric::next_speaker()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void PowerSpectrumVADMetric::clear_channel()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
  if( NULL != powerList_ )
    gsl_vector_free( powerList_ );
  powerList_ = NULL;
}

// ----- definition for class `NormalizedEnergyMetric' -----
//
NormalizedEnergyMetric::NormalizedEnergyMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
PowerSpectrumVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm )
{
  E0_ = 1.0;
}

NormalizedEnergyMetric::~NormalizedEnergyMetric()
{}

/**
   @brief compuate the ratio of each channel's energy to the total energy at each frame.
   @return 1.0 if the voice detected. Otherwise, return 0.0.
 */
double NormalizedEnergyMetric::next(int frame_no)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalenergy = 0;
  double energy_ratio = 0;

#ifdef  _LOG_SAD_ 
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  if( NULL == powerList_ )
    powerList_ = gsl_vector_alloc( chanN );
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frame_no);
    for (unsigned fbinX = lowX_; fbinX <= highX_; fbinX++) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= fftLen_;
    totalenergy += sqrt( power_n );
    gsl_vector_set( powerList_, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
    //fprintf(stderr,"Energy Chan %d : %e\n", chanX, power_n  );
  }

  energy_ratio = sqrt( gsl_vector_get( powerList_,0) ) / totalenergy;
  cur_score_ = energy_ratio;

  //fprintf(stderr,"ER %e  %e\n", energy_ratio, E0_/chanN  );
#ifdef  _LOG_SAD_
  writeLog( "%d %e\n", frame_no_, energy_ratio );
  setScore( energy_ratio );
#endif /* _LOG_SAD_ */

  if ( energy_ratio > (E0_/chanN) )
    return 1.0;
  else
    return -1.0;

  return 0.0;
}

void NormalizedEnergyMetric::reset()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

// ----- definition for class `CCCVADMetric' -----
//

/**
   @brief constructor
   @param unsigned fftLen[in]
   @param unsigned nCand[in] the number of candidates
   @param double sampleRate[in]
   @param double lowCutoff[in]
   @param double highCutoff[in]
*/
CCCVADMetric::CCCVADMetric(unsigned fftLen, unsigned nCand, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
  ComplexMultiChannelVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm)
{
  nCand_ = nCand;
  ccList_ = gsl_vector_alloc( nCand_ );
  sample_delays_ = gsl_vector_int_alloc( nCand_ );
  pack_cross_spectrum_ = new double[2*fftLen_];
  threshold_ = 0.1;
}

CCCVADMetric::~CCCVADMetric()
{
  gsl_vector_free( ccList_ );
  gsl_vector_int_free( sample_delays_ );
  delete [] pack_cross_spectrum_;
}

void CCCVADMetric::set_NCand(unsigned nCand)
{
  gsl_vector_free( ccList_ );
  gsl_vector_int_free( sample_delays_ );

  nCand_ = nCand;
  ccList_ = gsl_vector_alloc( nCand_ );
  sample_delays_ = gsl_vector_int_alloc( nCand_ );
}

double CCCVADMetric::next(int frame_no)
{
#define myREAL(z,i) ((z)[2*(i)])
#define myIMAG(z,i) ((z)[2*(i)+1])
  size_t stride = 1;
  const gsl_vector_complex* refSpectrum;
  const gsl_vector_complex* spectrum;
  double totalCCMetric = 0.0;

#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  for(unsigned fbinX=0;fbinX<2*fftLen_;fbinX++)
    pack_cross_spectrum_[fbinX] = 0.0;

  ChannelIterator_ itr = _channelList.begin();
  refSpectrum = (*itr)->next(frame_no);
  itr++;
  for (unsigned chanX = 1; itr != _channelList.end(); itr++,chanX++) {
    spectrum = (*itr)->next(frame_no);
    for (unsigned fbinX = lowX_; fbinX <= highX_; fbinX++) {
      gsl_complex val1, val2, cc;

      val1 = gsl_vector_complex_get( refSpectrum, fbinX );
      val2 = gsl_vector_complex_get( spectrum, fbinX );
      cc   = gsl_complex_mul( gsl_complex_conjugate( val1 ), val2 );
      cc   = gsl_complex_div_real( cc,  gsl_complex_abs( cc ) );
      myREAL( pack_cross_spectrum_, fbinX ) = GSL_REAL(cc);
      myIMAG( pack_cross_spectrum_, fbinX ) = GSL_IMAG(cc);
      if( fbinX > 0 ){
	myREAL( pack_cross_spectrum_, (fftLen_-fbinX)*stride ) =  GSL_REAL(cc);
	myIMAG( pack_cross_spectrum_, (fftLen_-fbinX)*stride ) = -GSL_IMAG(cc);
      }
    }
    gsl_fft_complex_radix2_inverse( pack_cross_spectrum_, stride, fftLen_ );// with scaling
    {/* detect _nHeldMaxCC peaks */
      /* ccList_[0] > ccList_[1] > ccList_[2] ... */

      gsl_vector_int_set( sample_delays_, 0, 0 );
      gsl_vector_set( ccList_, 0, myREAL( pack_cross_spectrum_, 0 ) );
      for(unsigned i=1;i<nCand_;i++){
	gsl_vector_int_set( sample_delays_, i, -10 );
	gsl_vector_set( ccList_, i, -1e10 );
      }
      for(unsigned fbinX=1;fbinX<fftLen_;fbinX++){
	double cc = myREAL( pack_cross_spectrum_, fbinX );
	
	if( cc > gsl_vector_get( ccList_, nCand_-1 ) ){
	  for(unsigned i=0;i<nCand_;i++){
	    if( cc > gsl_vector_get( ccList_, i ) ){
	      for(unsigned j=nCand_-1;j>i;j--){
		gsl_vector_int_set( sample_delays_, j, gsl_vector_int_get( sample_delays_, j-1 ) );
		gsl_vector_set(     ccList_,       j, gsl_vector_get( ccList_, j-1 ) );
	      }
	    }
	    gsl_vector_int_set( sample_delays_, i, fbinX);
	    gsl_vector_set(     ccList_,       i, cc);
	    break;
	  }
	}
      }

      double ccMetric = 0.0;
      //set time delays to _vector
      for(unsigned i=0;i<nCand_;i++){
	unsigned sampleDelay = gsl_vector_int_get( sample_delays_, i );
	double   cc = gsl_vector_get( ccList_, i );
	float timeDelay;
	
	if( sampleDelay < fftLen_/2 ){
	  timeDelay = sampleDelay * 1.0 / samplerate_;
	}
	else{
	  timeDelay = - ( fftLen_ - sampleDelay ) * 1.0 / samplerate_;
	}
	//fprintf(stderr,"Chan %d : %d : SD = %d : TD = %e : CC = %e\n", chanX, i, sampleDelay, timeDelay, cc );
	ccMetric += cc;
      }
      ccMetric /= nCand_;
      totalCCMetric += ccMetric;
    }
  }
  totalCCMetric = totalCCMetric / ( _channelList.size() - 1 );
  cur_score_ = totalCCMetric;
#ifdef  _LOG_SAD_
  writeLog( "%d %e\n", frame_no_, totalCCMetric );
  setScore( totalCCMetric );
#endif /* _LOG_SAD_ */

  if( totalCCMetric < threshold_ )
    return 1.0;
  else
    return -1.0;
  return 0.0;

#undef myREAL
#undef myIMAG
}

void CCCVADMetric::reset()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void CCCVADMetric::next_speaker()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void CCCVADMetric::clear_channel()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
}

// ----- definition for class `TSPSVADMetric' -----
//
TSPSVADMetric::TSPSVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
PowerSpectrumVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm )
{
  E0_ = 5000;
}

TSPSVADMetric::~TSPSVADMetric()
{}

double TSPSVADMetric::next(int frame_no)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalpower = 0;
  double TSPS = 0;
  double tgtPower;

#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  if( NULL == powerList_ )
    powerList_ = gsl_vector_alloc( chanN );

  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frame_no);
    for (unsigned fbinX = lowX_; fbinX <= highX_; fbinX++) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= fftLen_;
    totalpower += power_n;
    gsl_vector_set( powerList_, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
  }

  tgtPower = gsl_vector_get( powerList_, 0 );
  TSPS = log(tgtPower/(totalpower-tgtPower)) - log(E0_/totalpower);
#ifdef  _LOG_SAD_
  writeLog( "%d %e\n", frame_no_, TSPS );
  setScore( TSPS );
#endif /* _LOG_SAD_ */

  if( TSPS > 0 )
    return 1.0;
  else
    return -1.0;

  return 0.0;
}

void TSPSVADMetric::reset()
{
  for (ChannelIterator_ itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

// ----- Methods for class 'NegentropyVADMetric::ComplexGeneralizedGaussian_' -----
//
NegentropyVADMetric::ComplexGeneralizedGaussian_::
ComplexGeneralizedGaussian_(double shapeFactor)
  : shape_factor_(shapeFactor), Bc_(calc_Bc_()), normalization_(calc_normalization_()) { }

double NegentropyVADMetric::ComplexGeneralizedGaussian_::calc_Bc_() const
{
  double lg1 = gsl_sf_lngamma(2.0 / shape_factor_);
  double lg2 = gsl_sf_lngamma(4.0 / shape_factor_);

  return exp((lg1 - lg2) / 2.0);
}

double NegentropyVADMetric::ComplexGeneralizedGaussian_::calc_normalization_() const
{
  return log(shape_factor_ / (2 * M_PI * Bc_ * Bc_ * gsl_sf_gamma(2.0 / shape_factor_)));
}

double NegentropyVADMetric::ComplexGeneralizedGaussian_::logLhood(gsl_complex X, double scaleFactor) const
{
  return normalization_ - pow(gsl_complex_abs(X) / (scaleFactor * Bc_), shape_factor_) - 2.0 * log(scaleFactor);
}


// ----- Methods for class 'NegentropyVADMetric' -----
//
NegentropyVADMetric::
NegentropyVADMetric(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
		    const String& shapeFactorFileName, double threshold, double sampleRate, double lowCutoff, double highCutoff,
		    const String& nm)
  : source_(source),
    spectral_estimator_(spectralEstimator), gaussian_(new ComplexGeneralizedGaussian_()),
    threshold_(threshold), fftLen_(source_->size()), fftLen2_(fftLen_ / 2),
    samplerate_(sampleRate), lowX_(set_lowX_(lowCutoff)), highX_(set_highX_(highCutoff)), binN_(set_binN_())
{
  size_t n      = 0;
  char*  buffer = NULL;
  static char fileName[256];

  // initialize the GG pdfs
  for (unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++) {

    double factor = 2.0;	// default is Gaussian pdf
    if (shapeFactorFileName != "") {
      sprintf(fileName, "%s/_M-%04d", shapeFactorFileName.c_str(), fbinX);
      FILE* fp = btk_fopen(fileName, "r");
      getline(&buffer, &n, fp);
      static char* token[2];
      token[0] = strtok(buffer, " ");
      token[1] = strtok(NULL, " ");

      factor = strtod(token[1], NULL);
      btk_fclose(fileName, fp);
    }

    // printf("Bin %d has shape factor %8.4f\n", fbinX, factor);  fflush(stdout);

    generalized_gaussians_.push_back(ComplexGeneralizedGaussianPtr_(new ComplexGeneralizedGaussian_(factor)));
  }

  if (generalized_gaussians_.size() != fftLen2_ + 1)
    throw jdimension_error("Numbers of spectral bins and shape factors do not match (%d vs. %d)", generalized_gaussians_.size(), fftLen2_ + 1);
}

NegentropyVADMetric::~NegentropyVADMetric() { }

double NegentropyVADMetric::calc_negentropy(int frame_no)
{
  const gsl_vector_complex* sample   = source_->next(frame_no);
  const gsl_vector_float*   envelope = spectral_estimator_->next(frame_no);

  unsigned fbinX = 0;
  double logLikelihoodRatio = 0.0;
  for (GaussianListConstIterator_ itr = generalized_gaussians_.begin(); itr != generalized_gaussians_.end(); itr++) {
    ComplexGeneralizedGaussianPtr_ gg(*itr);
    gsl_complex X      = gsl_vector_complex_get(sample, fbinX);
    double      sigmaH = sqrt(gsl_vector_float_get(envelope, fbinX));
    double	lr     = gg->logLhood(X, sigmaH) - gaussian_->logLhood(X, sigmaH);

    if (fbinX >= lowX_ && fbinX <= highX_) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1))
	logLikelihoodRatio += lr;
      else
	logLikelihoodRatio += 2.0 * lr;
    }

    fbinX++;
  }

  logLikelihoodRatio /= binN_;

  printf("Frame %d : Negentropy ratio = %12.4e\n", frame_no, logLikelihoodRatio);

  return logLikelihoodRatio;
}

double NegentropyVADMetric::next(int frame_no)
{
#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  if (calc_negentropy(frame_no) > threshold_)
    return 1.0;
  return 0.0;
}

void NegentropyVADMetric::reset()
{
  source_->reset();  spectral_estimator_->reset();
}

void NegentropyVADMetric::next_speaker()
{
  cout << "NegentropyVADMetric::next_speaker" << endl;
}

bool NegentropyVADMetric::above_threshold_(int frame_no)
{
  return next(frame_no) > threshold_;
}

unsigned NegentropyVADMetric::set_lowX_(double lowCutoff) const
{
  if (lowCutoff < 0.0) return 0;

  if (lowCutoff >= samplerate_ / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", lowCutoff);

  unsigned binX = (unsigned) ((lowCutoff / samplerate_) * fftLen_);

  printf("Setting lowest bin to %d\n", binX);

  return binX;
}

unsigned NegentropyVADMetric::set_highX_(double highCutoff) const
{
  if (highCutoff < 0.0) return fftLen2_;

  if (highCutoff >= samplerate_ / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", highCutoff);

  unsigned binX = (unsigned) ((highCutoff / samplerate_) * fftLen_ + 0.5);

  printf("Setting highest bin to %d\n", binX);

  return binX;
}

unsigned NegentropyVADMetric::set_binN_() const
{
  if (lowX_ > 0)
    return 2 * (highX_ - lowX_ + 1);

  return 2 * (highX_ - lowX_) + 1;
}


// ----- Methods for class 'MutualInformationVADMetric::JointComplexGeneralizedGaussian_' -----
//
MutualInformationVADMetric::JointComplexGeneralizedGaussian_::
JointComplexGeneralizedGaussian_(const ComplexGeneralizedGaussianPtr_& ggaussian)
  : ComplexGeneralizedGaussian_(match_(ggaussian->shapeFactor())),
    X_(gsl_vector_complex_calloc(2)), scratch_(gsl_vector_complex_calloc(2)),
    SigmaX_inverse_(gsl_matrix_complex_calloc(2, 2))
{
  Bc_ = calc_Bc_();
  normalization_ = calc_normalization_();
}

MutualInformationVADMetric::JointComplexGeneralizedGaussian_::
~JointComplexGeneralizedGaussian_()
{
  gsl_vector_complex_free(X_);
  gsl_vector_complex_free(scratch_);
  gsl_matrix_complex_free(SigmaX_inverse_);
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::calc_Bc_() const
{
  double lg1 = gsl_sf_lngamma(4.0 / shape_factor_);
  double lg2 = gsl_sf_lngamma(6.0 / shape_factor_);

  return exp((lg1 - lg2) / 2.0);
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::calc_normalization_() const
{
  // cout << "Executing JointComplexGeneralizedGaussian_::calc_normalization_()" << endl;

  return log(shape_factor_ / (8.0 * M_PI * M_PI * Bc_ * Bc_ * Bc_ * Bc_ * gsl_sf_gamma(4.0 / shape_factor_)));
}

const double      MutualInformationVADMetric::JointComplexGeneralizedGaussian_::sqrt_two_	= sqrt(2.0);
const gsl_complex MutualInformationVADMetric::JointComplexGeneralizedGaussian_::complex_one_	= gsl_complex_rect(1.0, 0.0);
const gsl_complex MutualInformationVADMetric::JointComplexGeneralizedGaussian_::complex_zero_	= gsl_complex_rect(0.0, 0.0);

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::
logLhood(gsl_complex X1, gsl_complex X2, double scaleFactor1, double scaleFactor2, gsl_complex rho12) const
{
  // create the combined feature vector
  gsl_vector_complex_set(X_, 0, X1);
  gsl_vector_complex_set(X_, 1, X2);

  // calculate inverse of Sigma_X (up to scale factor)
  gsl_complex sigma12 = gsl_complex_mul_real(rho12, scaleFactor1 * scaleFactor2);
  gsl_complex sigma11 = gsl_complex_rect(scaleFactor1 * scaleFactor1, 0.0);
  gsl_complex sigma22 = gsl_complex_rect(scaleFactor2 * scaleFactor2, 0.0);

  gsl_matrix_complex_set(SigmaX_inverse_, 0, 0, sigma22);
  gsl_matrix_complex_set(SigmaX_inverse_, 1, 1, sigma11);
  gsl_matrix_complex_set(SigmaX_inverse_, 0, 1, gsl_complex_rect(-GSL_REAL(sigma12), -GSL_IMAG(sigma12)));
  gsl_matrix_complex_set(SigmaX_inverse_, 1, 0, gsl_complex_rect(-GSL_REAL(sigma12),  GSL_IMAG(sigma12)));

  // calculate determinant
  double determinant    = scaleFactor1 * scaleFactor1 * scaleFactor2 * scaleFactor2 * (1.0 - gsl_complex_abs2(rho12));
  double logDeterminant = log(determinant);

  // scale inverse of Sigma_X
  gsl_complex determinantComplex = gsl_complex_rect(1.0 / determinant, 0.0);
  gsl_matrix_complex_scale(SigmaX_inverse_, determinantComplex);

  // calculate (square-root of) s
  gsl_complex s;
  gsl_blas_zgemv(CblasNoTrans, complex_one_, SigmaX_inverse_, X_, complex_zero_, scratch_);
  gsl_blas_zdotc(X_, scratch_, &s);
  double ssqrt = sqrt(gsl_complex_abs(s));

  return normalization_ - pow(ssqrt / (sqrt_two_ * Bc_), shape_factor_) - logDeterminant;
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::lngamma_ratio_(double f) const
{
  return gsl_sf_lngamma(2.0 / f) + gsl_sf_lngamma(6.0 / f) - 2.0 * gsl_sf_lngamma(4.0 / f);
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::lngamma_ratio_joint_(double f) const
{
  return gsl_sf_lngamma(4.0 / f) + gsl_sf_lngamma(8.0 / f) - 2.0 * gsl_sf_lngamma(6.0 / f);
}

#if 0

// these methods are for kurtosis matching
double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::match_score_marginal_(double f) const
{
  return log(0.5 * (exp(lngamma_ratio_(f)) + 1.0));
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::match_score_joint_(double f) const
{
  return lngamma_ratio_joint_(f);
}

#else

// these methods are for entropy matching
double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::match_score_marginal_(double f) const
{
  double lg1		= gsl_sf_lngamma(2.0 / f);
  double lg2		= gsl_sf_lngamma(4.0 / f);
  double Bc2		= exp(lg1 - lg2);
  double gamma2f	= gsl_sf_gamma(2.0 / f);
  double match		= 2.0 * ((2.0 / f) - log(f / (2.0 * M_PI * Bc2 * gamma2f)));

  return -match;
}

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::match_score_joint_(double fJ) const
{
  double lg1		= gsl_sf_lngamma(4.0 / fJ);
  double lg2		= gsl_sf_lngamma(6.0 / fJ);
  double BJ4 		= exp((lg1 - lg2) * 2.0);
  double gamma4fJ	= gsl_sf_gamma(4.0 / fJ);
  double match		= ((4.0 / fJ) - log(fJ / (8.0 * M_PI * M_PI * BJ4 * gamma4fJ)));

  return -match;
}

#endif

const double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::tolerance_ = 1.0e-06;

double MutualInformationVADMetric::JointComplexGeneralizedGaussian_::match_(double f) const
{
  double a     = f / 3.0;					// lower bound: univariate shape factor
  double c     = 2.0;						// upper bound: Gaussian shape factor
  double match = match_score_marginal_(f);			// must match this value

  // printf("f = %10.6f\n", f);

  // return f;

  // execute binary search
  while (true) {
    double b		= (a + c) / 2.0;
    double ratiob	= match_score_joint_(b);

    if (fabs(match - ratiob) < tolerance_) {
      return b;
    }

    if (ratiob > match)
      a = b;		// ratio too high, take lower interval
    else		
      c = b;		// ratio too low, take upper interval
  }
}


// ----- Methods for class 'MutualInformationVADMetric' -----
//
MutualInformationVADMetric::
MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName, double twiddle, double threshold, double beta,
			   double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : NegentropyVADMetric(source1, spectralEstimator1, shapeFactorFileName, /* threshold= */ 0.0,
			sampleRate, lowCutoff, highCutoff, nm),
    source2_(source2), spectral_estimator2_(spectralEstimator2),
    ccs_(fftLen2_ + 1), fixed_threshold_(calc_fixed_threshold_()),
    twiddle_(twiddle), threshold_(threshold), beta_(beta) { }

MutualInformationVADMetric::~MutualInformationVADMetric() { }

void MutualInformationVADMetric::initialize_pdfs_()
{
  unsigned fbinX = 0;
  for (GaussianListConstIterator_ itr = generalized_gaussians_.begin(); itr != generalized_gaussians_.end(); itr++) {
    // cout << "Initializing Bin " << fbinX << ":" << endl;
    const NegentropyVADMetric::ComplexGeneralizedGaussianPtr_& gg(*itr);
    joint_generalized_gaussians_.push_back(JointComplexGeneralizedGaussianPtr_(new JointComplexGeneralizedGaussian_(gg)));
    fbinX++;
  }
}

// this is the fixed portion of the decision threshold
double MutualInformationVADMetric::calc_fixed_threshold_()
{
  initialize_pdfs_();

  unsigned fbinX 		= 0;
  double   threshold		= 0.0;
  GaussianListConstIterator_ mitr = generalized_gaussians_.begin();
  for (JointGaussianListConstIterator_ itr = joint_generalized_gaussians_.begin(); itr != joint_generalized_gaussians_.end(); itr++) {
    JointComplexGeneralizedGaussianPtr_ joint(*itr);
    ComplexGeneralizedGaussianPtr_ marginal(*mitr);

    // marginal pdf contribution
    double f			= marginal->shapeFactor();
    double Bc2			= marginal->Bc() * marginal->Bc();
    double gamma2f		= gsl_sf_gamma(2.0 / f);
    double thresh		= 2.0 * ((2.0 / f) - log(f / (2.0 * M_PI * Bc2 * gamma2f)));

    // joint pdf contribution
    double fJ			= joint->shapeFactor();
    double BJ4			= pow(joint->Bc(), 4.0);
    double gamma4fJ		= gsl_sf_gamma(4.0 / fJ);
    thresh		       -= ((4.0 / fJ) - log(fJ / (8.0 * M_PI * M_PI * BJ4 * gamma4fJ)));

    if (fbinX >= lowX_ && fbinX <= highX_) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1))
	threshold += thresh;
      else
	threshold += 2.0 * thresh;
    }

    fbinX++;  mitr++;
  }

  // don't normalize yet
  return threshold;
}

// complete threshold, fixed component and that depending on CC coefficients
double MutualInformationVADMetric::calc_total_threshold_() const
{
  double totalThreshold = fixed_threshold_;
  for (unsigned fbinX = lowX_; fbinX <= highX_; fbinX++) {
    gsl_complex rho12	= ccs_[fbinX];
    double thresh	= - log(1.0 - gsl_complex_abs2(rho12));

    if (fbinX == 0 || fbinX == (fftLen2_ + 1))
      totalThreshold += thresh;
    else
      totalThreshold += 2.0 * thresh;
  }

  // now normalize the total threshold
  totalThreshold *= (twiddle_ / binN_);

  return totalThreshold;
}

const double MutualInformationVADMetric::epsilon_ = 0.10;

double MutualInformationVADMetric::calc_mutual_information(int frame_no)
{
  const gsl_vector_complex* sample1   = source_->next(frame_no);
  const gsl_vector_complex* sample2   = source2_->next(frame_no);
  const gsl_vector_float*   envelope1 = spectral_estimator_->next(frame_no);
  const gsl_vector_float*   envelope2 = spectral_estimator2_->next(frame_no);
  unsigned fbinX 		= 0;
  double   mutualInformation	= 0.0;

#ifdef  _LOG_SAD_ 
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  GaussianListConstIterator_ mitr = generalized_gaussians_.begin();
  for (JointGaussianListConstIterator_ itr = joint_generalized_gaussians_.begin(); itr != joint_generalized_gaussians_.end(); itr++) {
    JointComplexGeneralizedGaussianPtr_ joint(*itr);
    ComplexGeneralizedGaussianPtr_ marginal(*mitr);
    gsl_complex X1	= gsl_vector_complex_get(sample1, fbinX);
    gsl_complex X2	= gsl_vector_complex_get(sample2, fbinX);
    double      sigma1	= sqrt(gsl_vector_float_get(envelope1, fbinX));
    double      sigma2	= sqrt(gsl_vector_float_get(envelope2, fbinX));
    gsl_complex rho12	= ccs_[fbinX];

    // calculate empirical mutual information
    double jointLhood	= joint->logLhood(X1, X2, sigma1, sigma2, rho12);
    double marginal1	= marginal->logLhood(X1, sigma1);
    double marginal2	= marginal->logLhood(X2, sigma2);
    double mutual	= jointLhood - marginal1 - marginal2;

    /*
    if (fbinX == 15) {
      printf("(|X_1|, sigma1, |X_1| / sigma1) = (%12.4f, %12.4f, %12.4f)\n", gsl_complex_abs(X1), sigma1, gsl_complex_abs(X1) / sigma1);
      printf("(|X_2|, sigma2, |X_2| / sigma2) = (%12.4f, %12.4f, %12.4f)\n", gsl_complex_abs(X2), sigma2, gsl_complex_abs(X2) / sigma2);
      printf("rho12 = (%12.4f, %12.4f)\n", GSL_REAL(rho12), GSL_IMAG(rho12));
      printf("Mutual %12.4f\n", mutual);
      printf("Check here\n");
    } */

    if (fbinX >= lowX_ && fbinX <= highX_) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1))
	mutualInformation += mutual;
      else
	mutualInformation += 2.0 * mutual;
    }

    // update cross-correlation coefficient for next frame
    gsl_complex cross	= gsl_complex_div_real(gsl_complex_mul(X1, gsl_complex_conjugate(X2)), sigma1 * sigma2);
    rho12		= gsl_complex_add(gsl_complex_mul_real(rho12, beta_), gsl_complex_mul_real(cross, 1.0 - beta_));
    if (gsl_complex_abs(rho12) >= (1.0 - epsilon_)) {
      // printf("Rescaling rho12 = (%12.4f, %12.4f) in bin %d\n", GSL_REAL(rho12), GSL_IMAG(rho12), fbinX);
      rho12 = gsl_complex_mul_real(rho12, ((1.0 - epsilon_) / gsl_complex_abs(rho12)));
    }
    ccs_[fbinX] = rho12;

    fbinX++;  mitr++;
  }

  mutualInformation /= binN_;

  // printf("Frame %d : MI = %12.4e\n", frame_no, mutualInformation);

  return mutualInformation;
}

double MutualInformationVADMetric::next(int frame_no)
{
  double threshold	= (twiddle_ < 0.0) ? threshold_ : calc_total_threshold_();
  double mutual		= calc_mutual_information(frame_no);

  // printf("Frame %d : Mutual Information %12.4f : Threshold %12.4f\n", frame_no, mutual, threshold);
#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  if (mutual > threshold)
    return 1.0;
  return 0.0;
}

void MutualInformationVADMetric::reset()
{
  source_->reset();             source2_->reset();
  spectral_estimator_->reset();  spectral_estimator2_->reset();
}

void MutualInformationVADMetric::next_speaker()
{
  cout << "MutualInformationVADMetric::next_speaker" << endl;

  // reset all cross correlations
  for (unsigned i = 0; i < ccs_.size(); i++)
    ccs_[i] = gsl_complex_rect(0.0, 0.0);
}

bool MutualInformationVADMetric::above_threshold_(int frame_no)
{
  return next(frame_no) > threshold_;
}


// ----- Methods for class 'LikelihoodRatioVADMetric' -----
//
LikelihoodRatioVADMetric::
LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			 const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			 const String& shapeFactorFileName, double threshold,
			 double sampleRate, double lowCutoff, double highCutoff,
			 const String& nm)
  : NegentropyVADMetric(source1, spectralEstimator1, shapeFactorFileName, threshold,
			sampleRate, lowCutoff, highCutoff, nm),
    source2_(source2), spectral_estimator2_(spectralEstimator2) { }

LikelihoodRatioVADMetric::~LikelihoodRatioVADMetric() { }

double LikelihoodRatioVADMetric::calc_likelihood_ratio(int frame_no)
{
  const gsl_vector_complex* sample1   = source_->next(frame_no);
  const gsl_vector_complex* sample2   = source2_->next(frame_no);
  const gsl_vector_float*   envelope1 = spectral_estimator_->next(frame_no);
  const gsl_vector_float*   envelope2 = spectral_estimator2_->next(frame_no);

  unsigned fbinX 		= 0;
  double likelihoodRatio	= 0.0;

#ifdef  _LOG_SAD_
  frame_no_ = frame_no;
#endif /* _LOG_SAD_ */

  for (GaussianListConstIterator_ itr = generalized_gaussians_.begin(); itr != generalized_gaussians_.end(); itr++) {
    ComplexGeneralizedGaussianPtr_ marginal(*itr);
    gsl_complex X1	= gsl_vector_complex_get(sample1, fbinX);
    gsl_complex X2	= gsl_vector_complex_get(sample2, fbinX);
    // double      sigma1	= sqrt(gsl_vector_float_get(envelope1, fbinX));
    // double      sigma2	= sqrt(gsl_vector_float_get(envelope2, fbinX));
    // double	lr	= marginal->logLhood(X1, sigma1) - marginal->logLhood(X2, sigma2);

    double      sigma1	= gsl_vector_float_get(envelope1, fbinX);
    double      sigma2	= gsl_vector_float_get(envelope2, fbinX);
    double	sigma	= sqrt((sigma1 + sigma2) / 2);
    double	marg1	= marginal->logLhood(X1, sigma);
    double	marg2	= marginal->logLhood(X2, sigma);
    double	lr	= marg1 - marg2;

    if (fbinX == 4) {
      printf("(|X_1|, |X_1| / sigma) = (%12.4f, %12.4f)\n", gsl_complex_abs(X1), gsl_complex_abs(X1) / sigma);
      printf("(|X_2|, |X_2| / sigma) = (%12.4f, %12.4f)\n", gsl_complex_abs(X2), gsl_complex_abs(X2) / sigma);
      printf("Likelihood Ratio = %12.4f\n", lr);
      printf("Check here\n");
    }

    if (fbinX >= lowX_ && fbinX <= highX_) {
      if (fbinX == 0 || fbinX == (fftLen2_ + 1))
	likelihoodRatio += lr;
      else
	likelihoodRatio += 2.0 * lr;
    }
    fbinX++;
  }
  likelihoodRatio /= binN_;

  printf("Frame %d : LR = %12.4e\n", frame_no, likelihoodRatio);

  return likelihoodRatio;
}

double LikelihoodRatioVADMetric::next(int frame_no)
{
  if (calc_likelihood_ratio(frame_no) > threshold_)
    return 1.0;
  return 0.0;
}

void LikelihoodRatioVADMetric::reset()
{
  source_->reset();             source2_->reset();
  spectral_estimator_->reset();  spectral_estimator2_->reset();
}

void LikelihoodRatioVADMetric::next_speaker()
{
  cout << "LikelihoodRatioVADMetric::next_speaker" << endl;
}


// ----- Methods for class 'LowFullBandEnergyRatioVADMetric' -----
//
LowFullBandEnergyRatioVADMetric::
LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold, const String& nm)
  : source_(source), _lagsN(lowpass->size), _lowpass(gsl_vector_calloc(_lagsN)), scratch_(gsl_vector_calloc(_lagsN)),
    _autocorrelation(new double[_lagsN]), _covariance(gsl_matrix_calloc(_lagsN, _lagsN))
{
  gsl_vector_memcpy(_lowpass, lowpass);
}

LowFullBandEnergyRatioVADMetric::~LowFullBandEnergyRatioVADMetric()
{
  gsl_vector_free(_lowpass);  gsl_vector_free(scratch_);  delete[] _autocorrelation;  gsl_matrix_free(_covariance);
}

void LowFullBandEnergyRatioVADMetric::calc_auto_correlation_vector_(int frame_no)
{
  const gsl_vector_float* samples = source_->next(frame_no);
  unsigned		  sampleN = samples->size;

  for (unsigned lag = 0; lag < _lagsN; lag++) {
    double r_xx = 0.0;
    for (unsigned i = lag; i < sampleN; i++)
      r_xx += gsl_vector_float_get(samples, i) * gsl_vector_float_get(samples, i - lag);
    _autocorrelation[lag] = r_xx / (sampleN - lag);
  }
}

void LowFullBandEnergyRatioVADMetric::calc_covariance_matrix_()
{
  for (unsigned rowX = 0; rowX < _lagsN; rowX++) {
    for (unsigned colX = rowX; colX < _lagsN; colX++) {
      gsl_matrix_set(_covariance, rowX, colX, _autocorrelation[colX - rowX]);
      gsl_matrix_set(_covariance, colX, rowX, _autocorrelation[colX - rowX]);
    }
  }
}

double LowFullBandEnergyRatioVADMetric::calc_lower_band_energy_()
{
  double innerProduct;
  gsl_blas_dgemv(CblasNoTrans, 1.0, _covariance, _lowpass, 1.0, scratch_);
  gsl_blas_ddot(_lowpass, scratch_, &innerProduct);

  return innerProduct;
}

double LowFullBandEnergyRatioVADMetric::next(int frame_no)
{
  calc_auto_correlation_vector_(frame_no);
  calc_covariance_matrix_();
  double le = calc_lower_band_energy_();

  return le / _autocorrelation[0];
}

void LowFullBandEnergyRatioVADMetric::reset() { }

void LowFullBandEnergyRatioVADMetric::next_speaker() { }

bool LowFullBandEnergyRatioVADMetric::above_threshold_(int frame_no) { return false; }


// ----- Methods for class 'HangoverVADFeature' -----
//
HangoverVADFeature::
HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold,
		   unsigned headN, unsigned tailN, const String& nm)
  : VectorFloatFeatureStream(source->size(), nm),
    source_(source), recognizing_(false),
    buffer_(NULL),
    headN_(headN), tailN_(tailN),
    bufferX_(0), bufferedN_(0),
    abovethresholdN_(0), belowThresholdN_(0), prefixN_(0)
{
  metricList_.push_back(MetricPair_(metric, threshold));

  buffer_ = new gsl_vector_float*[headN_];
  for (unsigned n = 0; n < headN_; n++)
    buffer_[n] = gsl_vector_float_calloc(source_->size());
}

HangoverVADFeature::~HangoverVADFeature()
{
  for (unsigned n = 0; n < headN_; n++)
    gsl_vector_float_free(buffer_[n]);

  delete[] buffer_;
}

void HangoverVADFeature::reset()
{
  source_->reset();  VectorFloatFeatureStream::reset();

  bufferX_ = bufferedN_ = abovethresholdN_ = belowThresholdN_ = prefixN_ = 0;
  recognizing_ = false;

  for (MetricListIterator_ itr = metricList_.begin(); itr != metricList_.end(); itr++) {
    VADMetricPtr& metric((*itr).first);
    metric->reset();
  }
}

void HangoverVADFeature::next_speaker()
{
  source_->reset();  VectorFloatFeatureStream::reset();

  bufferX_ = bufferedN_ = abovethresholdN_ = belowThresholdN_ = prefixN_ = 0;
  recognizing_ = false;

  for (MetricListIterator_ itr = metricList_.begin(); itr != metricList_.end(); itr++) {
    VADMetricPtr& metric((*itr).first);
    metric->next_speaker();
  }
}

bool HangoverVADFeature::above_threshold_(int frame_no)
{
  MetricListIterator_ itr = metricList_.begin();
  VADMetricPtr& metric((*itr).first);
  double threshold((*itr).second);
  double val = metric->next(frame_no);

  return (val > threshold);
}

const gsl_vector_float* HangoverVADFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_){
    fprintf(stderr,"Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);
  }

  // printf("HangoverVADFeature::next: FrameX = %d : Source FrameX = %d\n", frame_no, source_->frame_no()); fflush(stdout);
  if (recognizing_) {

    // use up the buffered blocks
    if (bufferedN_ > 0) {
      const gsl_vector_float* vector = buffer_[bufferX_];
      bufferX_ = (bufferX_ + 1) % headN_;
      bufferedN_--;
      increment_();
      return vector;
    }

    // buffer is empty; take blocks directly from source
    const gsl_vector_float* vector = source_->next(frame_no + prefixN());
    if (above_threshold_(frame_no + prefixN())) {
      belowThresholdN_ = 0;

      // printf("Decoding: FrameX = %d : Source FrameX = %d\n", frame_no, source_->frame_no()); fflush(stdout);

    } else {
      belowThresholdN_++;

      // printf("Tail: FrameX = %d : Source FrameX = %d : Below Threshold = %d\n", frame_no, source_->frame_no(), belowThresholdN_); fflush(stdout);

      if (belowThresholdN_ == tailN_)
	throw jiterator_error("end of samples!");
    }
    increment_();
    return vector;

  } else {

    // buffer sample blocks until sufficient blocks have energy above the threshold
    while (true) {
      const gsl_vector_float* vector = source_->next(prefixN_);

      gsl_vector_float_memcpy(buffer_[bufferX_], vector);
      bufferX_ = (bufferX_ + 1) % headN_;
      bufferedN_ = min(headN_, bufferedN_ + 1);

      if (above_threshold_(prefixN_++)) {
	abovethresholdN_++;

	// printf("FrameX = %d : Source FrameX = %d : Above Threshold = %d : Prefix = %d\n", frame_no, source_->frame_no(), abovethresholdN_, prefixN_ - 1);  fflush(stdout);

	if (abovethresholdN_ == headN_) {
	  recognizing_ = true;
	  vector = buffer_[bufferX_];
	  bufferX_ = (bufferX_ + 1) % headN_;
	  bufferedN_--;
	  increment_();
	  return vector;
	}
      } else {

	// printf("Waiting : FrameX = %d : Source FrameX = %d : Prefix = %d\n", frame_no, source_->frame_no(), prefixN_ - 1);  fflush(stdout);

	abovethresholdN_ = 0;
      }
    }
  }
}

// ----- Methods for class 'HangoverMIVADFeature' -----
//
HangoverMIVADFeature::
HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		     const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		     double energyThreshold, double mutualInformationThreshold, double powerThreshold,
		     unsigned headN, unsigned tailN, const String& nm)
  : HangoverVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm)
{
  metricList_.push_back(MetricPair_(mutualInformationMetric, mutualInformationThreshold));
  metricList_.push_back(MetricPair_(powerMetric, powerThreshold));
}

// ad hoc decision making process (should create a separate class for this)
bool HangoverMIVADFeature::above_threshold_(int frame_no)
{
  VADMetricPtr& energyMetric            = metricList_[EnergyVADMetricX].first;
  VADMetricPtr& mutualInformationMetric = metricList_[MutualInformationVADMetricX].first;
  VADMetricPtr& likelihoodRatioMetric   = metricList_[LikelihoodRatioVADMetricX].first;

  if (energyMetric->next(frame_no) < 0.5) {
    decision_metric_ = -1;
    mutualInformationMetric->next(frame_no);
    likelihoodRatioMetric->next(frame_no);
    return false;
  }

  if (mutualInformationMetric->next(frame_no) < 0.5) {
    decision_metric_ = 2;
    likelihoodRatioMetric->next(frame_no);
    return true;
  }

  if (likelihoodRatioMetric->next(frame_no) > 0.5) {
    decision_metric_ = 3;
    return true;
  }

  decision_metric_ = -3;
  return false;
}

// ----- Methods for class 'HangoverMIVADFeature' -----
//
HangoverMultiStageVADFeature::
HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
			     const VADMetricPtr& energyMetric, double energyThreshold, 
			     unsigned headN, unsigned tailN, const String& nm)
  : HangoverVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm)
#ifdef _LOG_SAD_
  ,_scores(NULL)
#endif /* _LOG_SAD_ */
{
}

HangoverMultiStageVADFeature::~HangoverMultiStageVADFeature()
{
#ifdef _LOG_SAD_
  if( NULL!=_scores ){
    gsl_vector_free( _scores );
  }
#endif /* _LOG_SAD_ */
}

// ad hoc decision making process (should create a separate class for this)
bool HangoverMultiStageVADFeature::above_threshold_(int frame_no)
{
  if( metricList_.size() < 3 ){
    fprintf(stderr,"HangoverMultiStage::setMetric()\n");
    return false;
  }
#ifdef _LOG_SAD_
  if( NULL==_scores )
    initScores();
#endif /* _LOG_SAD_ */

  VADMetricPtr& energyMetric = metricList_[0].first; // the first stage

  if (energyMetric->next(frame_no) < 0.5) {
    decision_metric_ = -1; // determine non-voice activity based on the energy measure.

    for(unsigned metricX=1;metricX<metricList_.size();metricX++){
      VADMetricPtr& vadMetricPtr = metricList_[metricX].first;
      vadMetricPtr->next(frame_no);
    }
    return false;
  }

  for(unsigned stageX=1;stageX<metricList_.size();stageX++){
    VADMetricPtr& currentStageMetric = metricList_[stageX].first;

    if ( currentStageMetric->next(frame_no) > 0.5 ){
      decision_metric_ = stageX + 1;
      for(unsigned metricX=2;metricX<metricList_.size();metricX++){
	VADMetricPtr& vadMetricPtr = metricList_[metricX].first;
	vadMetricPtr->next(frame_no);
      }
      return true; // determine  voice activity.
    }
    else{
      decision_metric_ = - ( stageX + 1 );
    }
  }
  decision_metric_ = - ( metricList_.size() );

  return false;
}


#ifdef _LOG_SAD_

void HangoverMultiStageVADFeature::initScores()
{
  if( NULL==_scores ){
    _scores = gsl_vector_calloc( metricList_.size() );
  }
  for(unsigned metricX=0;metricX<metricList_.size();metricX++){
    VADMetricPtr& vadMetricPtr = metricList_[metricX].first;
    vadMetricPtr->initScore();
  }
}

gsl_vector *HangoverMultiStageVADFeature::getScores()
{
  for(unsigned metricX=0;metricX<metricList_.size();metricX++){
    VADMetricPtr& vadMetricPtr = metricList_[metricX].first;
    gsl_vector_set( _scores, metricX, vadMetricPtr->getAverageScore() );
  }
  
  return _scores;
}

#endif /* _LOG_SAD_ */

