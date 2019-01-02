/**
 * @file sad_feature.cc
 * @brief Feature for voice activity detection.
 * @author John McDonough
 */

#include "sad/sad_feature.h"


// ----- define auxiliary functions -----
//
static float norm(float* vec, int n) {
  double norm = 0.0;
  for (unsigned i = 0; i < n; i++)
    norm += vec[i] * vec[i];

  return sqrt(norm);
}

static void normalize(float* vec, int n) {
  double sigma = norm(vec, n);
  for (unsigned i = 0; i < n; i++)
    vec[i] /= sigma;
}


// ----- methods for class `BrightnessFeature' -----
//
BrightnessFeature::BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm),
    src_(src), samplerate_(sampleRate), max_(samplerate_ / 2.0), df_(max_ / src_->size()), frs_(new float[src_->size()])
{
  for (unsigned i = 0; i < src_->size(); i++)
    frs_[i] = df_ * (float) (i + 1);
}

BrightnessFeature::~BrightnessFeature() { delete[] frs_; }

const gsl_vector_float* BrightnessFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = src_->next(frame_no);

  float n = 0.0f;
  float d = 0.0f;
  for (unsigned j = 0; j < src_->size(); j++) {

    if (weight_)
      n += frs_[j] * gsl_vector_float_get(block, j);
    else
      n += j * gsl_vector_float_get(block, j);

    d += gsl_vector_float_get(block, j);

    float val = n / d;
    if (!weight_)
      val /= src_->size();

    gsl_vector_float_set(vector_, j, val);
  }

  increment_();
  return vector_;
}


// ----- methods for class `EnergyDiffusionFeature' -----
//
EnergyDiffusionFeature::EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src) { }

EnergyDiffusionFeature::~EnergyDiffusionFeature() { }

const gsl_vector_float* EnergyDiffusionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = src_->next(frame_no);

  double norm = 0.0;
  for (unsigned j = 0; j < src_->size(); j++) {
    double val = gsl_vector_float_get(block, j);
    norm += val * val;
  }

  norm = sqrt(norm);
  double diff = 0.0;
  for (unsigned j = 0; j < src_->size(); j++) {
    double nval =  gsl_vector_float_get(block, j) / norm;
    diff -= (nval > 0.0 ? nval * log10(nval) : 0.0);
  }
  gsl_vector_float_set(vector_, 0, diff);

  increment_();
  return vector_;
}


// ----- methods for class `BandEnergyRatioFeature' -----
//
BandEnergyRatioFeature::BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src), samplerate_(sampleRate), max_(samplerate_ / 2.0),
    df_(max_ / src_->size()), threshF_((threshF > 0.0) ? threshF : max_ / 2.0f), threshX_(int(floor(threshF_ / df_))) { }

BandEnergyRatioFeature::~BandEnergyRatioFeature() { }

const gsl_vector_float* BandEnergyRatioFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = src_->next(frame_no);

  float ssLow  = 0.0f;
  float ssHigh = 0.0f;
  for (unsigned j = 0; j < threshX_; j++) {
    float val = gsl_vector_float_get(block, j);
    ssLow += val * val;
  }

  for (unsigned j = threshX_; j < src_->size(); j++) {
    float val = gsl_vector_float_get(block, j);
    ssHigh += val * val;
  }
  gsl_vector_float_set(vector_, 0, sqrt(ssLow / ssHigh));

  increment_();
  return vector_;
}


// ----- methods for class `NormalizedFluxFeature' -----
//
NormalizedFluxFeature::NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src), win0_(new float[src_->size()]), win1_(new float[src_->size()]) { }

NormalizedFluxFeature::~NormalizedFluxFeature() { delete[] win0_; delete[] win1_; }

const gsl_vector_float* NormalizedFluxFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = src_->next(frame_no);

  if (frame_no == 0) {
    memcpy(win0_, block->data, src_->size() * sizeof(float));
    normalize(win0_, src_->size());

    gsl_vector_float_set(vector_, 0, 0.0);
    increment_();
    return vector_;
  }
    
  memcpy(win1_, block->data, src_->size() * sizeof(float));
  normalize(win1_, src_->size());

  double sum = 0.0;
  for (unsigned j = 0; j < src_->size(); j++) {
    float diff = win0_[j] - win1_[j];
    sum += diff * diff;
  }
  gsl_vector_float_set(vector_, 0, sqrt(sum));
    
  memcpy(win1_, win0_, src_->size() * sizeof(float));

  increment_();
  return vector_;
}


// ----- methods for class `NegativeEntropyFeature' -----
//
NegativeEntropyFeature::NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src), win_(new float[src_->size()]) { }

NegativeEntropyFeature::~NegativeEntropyFeature() { delete[] win_; }

const gsl_vector_float* NegativeEntropyFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  const gsl_vector_float* block = src_->next(frame_no);

  // half-wave rectify
  for (unsigned j = 0; j < src_->size(); j++) {
    float val = gsl_vector_float_get(block, j);
    win_[j] = (val < 0 ? -val : val);
  }

  // normalize window to 0 mean, unity variance
  double sum  = 0.0;
  double sumS = 0.0;
  for (unsigned j = 0; j < src_->size(); j++) {
    sum  += win_[j];
    sumS += win_[j] * win_[j];
  }
  double mean = sum / src_->size();
  double dev  = sqrt((sumS / (src_->size() - 1)) - (mean * mean));
  for (unsigned j = 0; j < src_->size(); j++)
    win_[j] = (win_[j] - mean) / dev;

  // calculate E(G(y)), where G(u) = ln cosh(u) */
  sum = 0.0;
  for (unsigned j = 0; j < src_->size(); j++)
    sum += log(cosh(win_[j]));

  double EGy = sum / src_->size();
  static const double EGgy = 0.374576;		// according to mathematica
  gsl_vector_float_set(vector_, 0, 100.0 * (EGy - EGgy) * (EGy - EGgy));

  increment_();
  return vector_;
}


// ----- methods for class `SignificantSubbandsFeature' -----
//
SignificantSubbandsFeature::SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src), thresh_(thresh), win_(new float[src_->size()]) { }

SignificantSubbandsFeature::~SignificantSubbandsFeature() { delete[] win_; }

const gsl_vector_float* SignificantSubbandsFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  unsigned dimN = src_->size();
  const gsl_vector_float* block = src_->next(frame_no);

  memcpy(win_, block->data, dimN * sizeof(float));
  normalize(win_, dimN);
  double sum = 0.0;
  for (unsigned j = 0; j < dimN; j++)
    if (win_[j] > thresh_)
      sum += 1.0f;

  gsl_vector_float_set(vector_, 0, sum);

  increment_();
  return vector_;
}


// ----- methods for class `NormalizedBandwidthFeature' -----
//
NormalizedBandwidthFeature::NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), src_(src), thresh_(thresh), samplerate_(sampleRate), df_((samplerate_ / 2.0f) / src_->size()),
    frs_(new float[src_->size()]), win_(new float[src_->size()])
{
  for (unsigned i = 0; i < src_->size(); i++) {
    frs_[i] = df_ * (i + 1);
  }
}

NormalizedBandwidthFeature::~NormalizedBandwidthFeature() { delete[] frs_; delete[] win_; }

const gsl_vector_float* NormalizedBandwidthFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem: %d != %d\n", frame_no - 1, frame_no_);

  unsigned dimN = src_->size();
  const gsl_vector_float* block = src_->next(frame_no);

  memcpy(win_, block->data, dimN * sizeof(float));
  normalize(win_, dimN);

  int min = dimN;
  for (unsigned j = 0; j < dimN && min == dimN; j++) {
    if (win_[j] > thresh_) min = j;
  }
  int max = 0;
  for (int j = dimN - 1; j >= min && max == 0; j--) {
    if (win_[j] > thresh_) max = j;
  }
  gsl_vector_float_set(vector_, 0, frs_[max] - frs_[min]);

  increment_();
  return vector_;
}
