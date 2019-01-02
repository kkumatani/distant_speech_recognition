/*
 * @file dereverberation.cc
 * @brief Single- and multi-channel dereverberation base on linear prediction in the subband domain.
 * @author John McDonough
 */

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "common/jpython_error.h"
#include "dereverberation/dereverberation.h"

#ifdef HAVE_CONFIG_H
#include <btk.h>
#endif
#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif


// ----- methods for class `SingleChannelWPEDereverberationFeature' -----
//
SingleChannelWPEDereverberationFeature::
SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN, double loadDb, double bandWidth, double sampleRate, const String& nm)
  : VectorComplexFeatureStream(samples->size(), nm), samples_(samples),
    lowerN_(lowerN), upperN_(upperN), predictionN_(upperN_ - lowerN_ + 1), iterationsN_(iterationsN), first_frame_(true), framesN_(0), load_factor_(pow(10.0, loadDb / 10.0)),
    lower_bandWidthN_(set_band_width_(bandWidth, sampleRate)), upper_bandWidthN_(size() - lower_bandWidthN_),
    thetan_(NULL), gn_(new gsl_vector_complex*[size()]), R_(gsl_matrix_complex_alloc(predictionN_, predictionN_)), r_(gsl_vector_complex_alloc(predictionN_)),
    lag_samples_(gsl_vector_complex_alloc(predictionN_))
{
  // allocate prediction vectors
  for (unsigned n = 0; n < size(); n++)
    gn_[n] = gsl_vector_complex_calloc(predictionN_);
}

SingleChannelWPEDereverberationFeature::~SingleChannelWPEDereverberationFeature()
{
  if (thetan_ != NULL) gsl_matrix_free(thetan_);
  
  for (unsigned n = 0; n < size(); n++)
    gsl_vector_complex_free(gn_[n]);
  delete[] gn_;

  gsl_matrix_complex_free(R_);
  gsl_vector_complex_free(r_);

  for (SamplesIterator_ itr = yn_.begin(); itr != yn_.end(); itr++)
    gsl_vector_complex_free(*itr);
  yn_.clear();
}

const gsl_vector_complex* SingleChannelWPEDereverberationFeature::get_lags_(unsigned subbandX, unsigned sampleX)
{
  static const gsl_complex _Zero = gsl_complex_rect(0.0, 0.0);

  for (unsigned lagX = 0; lagX < predictionN_; lagX++) {
    int index = sampleX;  index -= lagX;
    gsl_complex val = (index < 0) ? _Zero : gsl_vector_complex_get(yn_[index], subbandX);
    gsl_vector_complex_set(lag_samples_, lagX, val);
  }

  return lag_samples_;
}

void SingleChannelWPEDereverberationFeature::fill_buffer_()
{
  framesN_ = 0;
  while (true) {
    const gsl_vector_complex* block;
    try {
      block = samples_->next();
    } catch (jiterator_error& e) {
      break;
    }
    gsl_vector_complex* sample = gsl_vector_complex_alloc(size());
    gsl_vector_complex_memcpy(sample, block);
    yn_.push_back(sample);
    framesN_++;
  }
  thetan_  = gsl_matrix_alloc(framesN_, size());
  gsl_matrix_set_zero(thetan_);
}

void SingleChannelWPEDereverberationFeature::calc_Rr_(unsigned subbandX)
{
  gsl_matrix_complex_set_zero(R_);
  gsl_vector_complex_set_zero(r_);

  // calculate R_
  for (unsigned sampleX = lowerN_; sampleX < framesN_; sampleX++) {
    double thetan = gsl_matrix_get(thetan_, sampleX, subbandX);
    const gsl_vector_complex* lag = get_lags_(subbandX, sampleX - lowerN_);
    for (unsigned rowX = 0; rowX < predictionN_; rowX++) {
      gsl_complex rowS = gsl_vector_complex_get(lag, rowX);
      for (unsigned colX = 0; colX <= rowX; colX++) {
	gsl_complex colS = gsl_vector_complex_get(lag, colX);
	gsl_complex val = gsl_matrix_complex_get(R_, rowX, colX);
	val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(rowS, gsl_complex_conjugate(colS)), thetan));
	gsl_matrix_complex_set(R_, rowX, colX, val);
      }
    }
  }

  // calculate r_
  unsigned sampleX = 0;
  double optimization = 0.0;
  for (SamplesIterator_ itr = yn_.begin(); itr != yn_.end(); itr++) {
    if (sampleX < lowerN_) { sampleX++; continue; }
    double thetan = gsl_matrix_get(thetan_, sampleX, subbandX);
    gsl_complex current = gsl_vector_complex_get(*itr, subbandX);
    const gsl_vector_complex* lags = get_lags_(subbandX, sampleX - lowerN_);

    gsl_complex dereverb;
    gsl_blas_zdotc(gn_[subbandX], lags, &dereverb);
    gsl_complex diff = gsl_complex_sub(current, dereverb);
    double dist = gsl_complex_abs(diff);
    optimization += dist * dist / thetan + log(thetan);

    for (unsigned lagX = 0; lagX < predictionN_; lagX++) {
      gsl_complex val = gsl_vector_complex_get(r_, lagX);
      val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(gsl_complex_conjugate(current), gsl_vector_complex_get(lags, lagX)), thetan));
      gsl_vector_complex_set(r_, lagX, val);
    }
    sampleX++;
  }

  if (subbandX == 100) {
    printf("Subband %4d : Criterion Value %10.4e\n", subbandX, optimization);
  }
}

const double SingleChannelWPEDereverberationFeature::subband_floor_ = 1.0E-03;

void SingleChannelWPEDereverberationFeature::calc_Thetan_()
{
  unsigned sampleX = 0;
  for (SamplesIterator_ itr = yn_.begin(); itr != yn_.end(); itr++) {
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
      gsl_complex current = gsl_vector_complex_get(*itr, subbandX);

      if (sampleX >= lowerN_) {
	gsl_complex dereverb;
	const gsl_vector_complex* lags = get_lags_(subbandX, sampleX - lowerN_);
	gsl_blas_zdotc(gn_[subbandX], lags, &dereverb);
	current = gsl_complex_sub(current, dereverb);
      }

      double thetan = gsl_complex_abs(current);
      if (thetan < subband_floor_) {
	// printf("Sample %d Subband %d theta_n = %0.2f\n", sampleX, subbandX, thetan);
	thetan = subband_floor_;
      }

      gsl_matrix_set(thetan_, sampleX, subbandX, thetan * thetan);
    }
    sampleX++;
  }
}

void SingleChannelWPEDereverberationFeature::load_R_()
{
  double maximumDiagonal = 0.0;
  for (unsigned componentX = 0; componentX < predictionN_; componentX++) {
    double diag = gsl_complex_abs(gsl_matrix_complex_get(R_, componentX, componentX));
    if (diag > maximumDiagonal) maximumDiagonal = diag;
  }

  for (unsigned componentX = 0; componentX < predictionN_; componentX++) {
    double diag = gsl_complex_abs(gsl_matrix_complex_get(R_, componentX, componentX)) + maximumDiagonal * load_factor_;
    gsl_matrix_complex_set(R_, componentX, componentX, gsl_complex_rect(diag, 0.0));
  }
}

void SingleChannelWPEDereverberationFeature::estimate_Gn_()
{
  for (unsigned iterationX = 0; iterationX < iterationsN_; iterationX++) {
    calc_Thetan_();
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {

      if ((subbandX > lower_bandWidthN_) && (subbandX < upper_bandWidthN_)) continue;

      calc_Rr_(subbandX);
      load_R_();
      gsl_linalg_complex_cholesky_decomp(R_);
      gsl_linalg_complex_cholesky_solve(R_, r_, gn_[subbandX]);

      if (subbandX == 100) {
	double sum = 0.0;
	for (unsigned componentX = 0; componentX < predictionN_; componentX++) {
	  double gn = gsl_complex_abs(gsl_vector_complex_get(gn_[subbandX], componentX));
	  sum += gn * gn;
	}

	double wng = 10.0 * log10(sum);
	printf("Iteration %d: Subband %4d WNG %6.2f\n", iterationX, subbandX, wng);
      }
    }
  }
}

const gsl_vector_complex* SingleChannelWPEDereverberationFeature::next(int frame_no) {
  if (first_frame_) { fill_buffer_();  estimate_Gn_();  first_frame_ = false; }

  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();
  if (frame_no_ == yn_.size())
    throw jiterator_error("end of samples!");

  const gsl_vector_complex* current = yn_[frame_no_];

  for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
    gsl_complex cur = gsl_vector_complex_get(current, subbandX);
    if ((frame_no_ >= lowerN_) && ((subbandX <= lower_bandWidthN_) || (subbandX >= upper_bandWidthN_))) {
      gsl_complex dereverb;
      const gsl_vector_complex* lags = get_lags_(subbandX, frame_no_ - lowerN_);
      gsl_blas_zdotc(gn_[subbandX], lags, &dereverb);

      cur = gsl_complex_sub(cur, dereverb);
    }
    gsl_vector_complex_set(vector_, subbandX, cur);
  }

  return vector_;
}

unsigned SingleChannelWPEDereverberationFeature::set_band_width_(double bandWidth, double sampleRate)
{
  if (bandWidth == 0.0) return (size() / 2);

  if (bandWidth > (sampleRate / 2.0))
    throw jdimension_error("Bandwidth is greater than the Nyquist rate.\n", bandWidth, (sampleRate / 2.0));

  return unsigned((bandWidth / (sampleRate / 2.0)) * (size() / 2));
}

void SingleChannelWPEDereverberationFeature::reset()
{
  samples_->reset();  VectorComplexFeatureStream::reset();  first_frame_ = true;  framesN_ = 0;

  if (thetan_ != NULL) { gsl_matrix_free(thetan_);  thetan_ = NULL; }

  for (SamplesIterator_ itr = yn_.begin(); itr != yn_.end(); itr++)
    gsl_vector_complex_free(*itr);
  yn_.clear();
}

void SingleChannelWPEDereverberationFeature::next_speaker()
{
  reset();
  for (unsigned n = 0; n < size(); n++)
    gsl_vector_complex_set_zero(gn_[n]);
}


// ----- methods for class `MultiChannelWPEDereverberation' -----
//
MultiChannelWPEDereverberation::MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN, double loadDb, double bandWidth,  double sampleRate)
  : sources_(0), subbandsN_(subbandsN), channelsN_(channelsN),
    lowerN_(lowerN), upperN_(upperN), predictionN_(upperN_ - lowerN_ + 1), iterationsN_(iterationsN), totalPredictionN_(predictionN_ * channelsN_),
    first_frame_(true), framesN_(0), load_factor_(pow(10.0, loadDb / 10.0)),
    lower_bandWidthN_(set_band_width_(bandWidth, sampleRate)), upper_bandWidthN_(size() - lower_bandWidthN_),
    thetan_(new gsl_matrix*[channelsN_]), Gn_(new gsl_vector_complex**[channelsN]),
    R_(new gsl_matrix_complex*[channelsN_]), r_(new gsl_vector_complex*[channelsN_]), lag_samples_(gsl_vector_complex_alloc(totalPredictionN_)),
    output_(new gsl_vector_complex*[channelsN]), initial_frame_no_(-1), frame_no_(initial_frame_no_)
{
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    thetan_[channelX] = NULL;
    R_[channelX]  = gsl_matrix_complex_alloc(totalPredictionN_, totalPredictionN_);
    r_[channelX]  = gsl_vector_complex_alloc(totalPredictionN_);
    Gn_[channelX] = new gsl_vector_complex*[subbandsN_];

    for (unsigned subbandX = 0; subbandX < subbandsN_; subbandX++)
      Gn_[channelX][subbandX] = gsl_vector_complex_alloc(totalPredictionN_);

    output_[channelX] = gsl_vector_complex_alloc(subbandsN_);
  }
}

MultiChannelWPEDereverberation::~MultiChannelWPEDereverberation()
{
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    if (thetan_[channelX] != NULL)
      gsl_matrix_free(thetan_[channelX]);

    for (unsigned subbandX = 0; subbandX < subbandsN_; subbandX++) {
      gsl_vector_complex_free(Gn_[channelX][subbandX]);
    }

    gsl_matrix_complex_free(R_[channelX]);
    gsl_vector_complex_free(r_[channelX]);
    delete[] Gn_[channelX];
    gsl_vector_complex_free(output_[channelX]);
  }

  delete[] thetan_;
  delete[] Gn_;
  delete[] R_;
  delete[] r_;
  delete[] output_;

  gsl_vector_complex_free(lag_samples_);
}

unsigned MultiChannelWPEDereverberation::set_band_width_(double bandWidth, double sampleRate)
{
  if (bandWidth == 0.0) return (size() / 2);

  if (bandWidth > (sampleRate / 2.0))
    throw jdimension_error("Bandwidth is greater than the Nyquist rate.\n", bandWidth, (sampleRate / 2.0));

  return unsigned((bandWidth / (sampleRate / 2.0)) * (size() / 2));
}

void MultiChannelWPEDereverberation::reset()
{
  first_frame_ = true;  framesN_ = 0;  frame_no_ = initial_frame_no_;

  for (SourceListIterator_ itr = sources_.begin(); itr != sources_.end(); itr++)
    (*itr)->reset();

  for (FrameBraceListIterator_ itr = frames_.begin(); itr != frames_.end(); itr++) {
    FrameBrace_& fbrace(*itr);
    for (FrameBraceIterator_ fitr = fbrace.begin(); fitr != fbrace.end(); fitr++) {
      gsl_vector_complex_free(*fitr);
    }
  }
  frames_.clear();
}

void MultiChannelWPEDereverberation::set_input(VectorComplexFeatureStreamPtr& samples)
{
  if (sources_.size() == channelsN_)
    throw jallocation_error("Channel capacity exceeded.");

  sources_.push_back(samples);
}

const gsl_vector_complex* MultiChannelWPEDereverberation::get_output(unsigned channelX, int frame_no)
{
  if (first_frame_) { fill_buffer_();  estimate_Gn_();  first_frame_ = false; }

  if (frame_no == frame_no_) return output_[channelX];

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in 'MultiChannelWPEDereverberation'\n");

  increment_();
  if (frame_no_ == framesN_)
    throw jiterator_error("end of samples!");

  // generate dereverberated output for *all* channels
  for (unsigned chanX = 0; chanX < channelsN_; chanX++ ) {
    const gsl_vector_complex* current = frames_[frame_no_][chanX];

    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
      gsl_complex cur = gsl_vector_complex_get(current, subbandX);
      if ((frame_no_ >= lowerN_) && ((subbandX <= lower_bandWidthN_) || (subbandX >= upper_bandWidthN_))) {
	gsl_complex dereverb;
	const gsl_vector_complex* lags = get_lags_(subbandX, frame_no_ - lowerN_);
	gsl_blas_zdotc(Gn_[channelX][subbandX], lags, &dereverb);

	cur = gsl_complex_sub(cur, dereverb);
      }
      gsl_vector_complex_set(output_[chanX], subbandX, cur);
    }
  }
  return output_[channelX];
}

void MultiChannelWPEDereverberation::fill_buffer_()
{
  while (true) {
    FrameBrace_ fbrace(channelsN_);
    try {
      for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
	VectorComplexFeatureStreamPtr src(sources_[channelX]);
	const gsl_vector_complex* block = src->next();
	gsl_vector_complex* sample = gsl_vector_complex_alloc(size());
	gsl_vector_complex_memcpy(sample, block);
	fbrace[channelX] = sample;
      }
    } catch (jiterator_error& e) {
      break;
    }
    frames_.push_back(fbrace);
  }
  framesN_ = frames_.size();
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    if (thetan_[channelX] != NULL) gsl_matrix_free(thetan_[channelX]);
    thetan_[channelX] = gsl_matrix_alloc(framesN_, size());
    gsl_matrix_set_zero(thetan_[channelX]);
  }
}

const gsl_vector_complex* MultiChannelWPEDereverberation::get_lags_(unsigned subbandX, unsigned sampleX)
{
  static const gsl_complex _Zero = gsl_complex_rect(0.0, 0.0);

  unsigned totalX = 0;
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    for (unsigned lagX = 0; lagX < predictionN_; lagX++) {
      int index = sampleX;  index -= lagX;
      gsl_complex val = (index < 0) ? _Zero : gsl_vector_complex_get(frames_[index][channelX], subbandX);
      gsl_vector_complex_set(lag_samples_, totalX, val);
      totalX++;
    }
  }

  return lag_samples_;
}

void MultiChannelWPEDereverberation::calc_Rr_(unsigned subbandX)
{
  // calculate (lower triangle of) R_ for all channels
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    gsl_matrix_complex* R = R_[channelX];
    gsl_matrix_complex_set_zero(R);
    for (unsigned sampleX = lowerN_; sampleX < framesN_; sampleX++) {
      double thetan = gsl_matrix_get(thetan_[channelX], sampleX, subbandX);
      const gsl_vector_complex* lags = get_lags_(subbandX, sampleX - lowerN_);
      for (unsigned rowX = 0; rowX < totalPredictionN_; rowX++) {
	gsl_complex rowS = gsl_vector_complex_get(lags, rowX);
	for (unsigned colX = 0; colX <= rowX; colX++) {
	  gsl_complex colS = gsl_vector_complex_get(lags, colX);
	  gsl_complex val = gsl_matrix_complex_get(R, rowX, colX);
	  val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(rowS, gsl_complex_conjugate(colS)), thetan));
	  gsl_matrix_complex_set(R, rowX, colX, val);
	}
      }
    }
  }

  // calculate r_ for all channels
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    gsl_vector_complex* r = r_[channelX];
    gsl_vector_complex_set_zero(r);

    unsigned sampleX = 0;
    double optimization = 0.0;
    for (FrameBraceListIterator_ itr = frames_.begin(); itr != frames_.end(); itr++) {
      if (sampleX < lowerN_) { sampleX++; continue; }

      FrameBrace_& frame(*itr);
      double thetan = gsl_matrix_get(thetan_[channelX], sampleX, subbandX);
      gsl_complex current = gsl_vector_complex_get(frame[channelX], subbandX);
      const gsl_vector_complex* lags = get_lags_(subbandX, sampleX - lowerN_);

      gsl_complex dereverb;
      gsl_blas_zdotc(Gn_[channelX][subbandX], lags, &dereverb);
      gsl_complex diff = gsl_complex_sub(current, dereverb);
      double dist = gsl_complex_abs(diff);
      optimization += dist * dist / thetan + log(thetan);

      for (unsigned lagX = 0; lagX < totalPredictionN_; lagX++) {
	gsl_complex val = gsl_vector_complex_get(r, lagX);
	val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(gsl_complex_conjugate(current), gsl_vector_complex_get(lags, lagX)), thetan));
	gsl_vector_complex_set(r, lagX, val);
      }
      sampleX++;
    }

    if (subbandX == 100) {
      // gsl_vector_complex_fprintf(stdout, Gn_[channelX][subbandX], "%0.2f");
      printf("Channel %d : Subband %4d : Criterion Value %10.4e\n", channelX, subbandX, optimization);
    }
  }
}

const double MultiChannelWPEDereverberation::subband_floor_ = 1.0E-03;

void MultiChannelWPEDereverberation::calc_Thetan_()
{
  unsigned sampleX = 0;
  for (FrameBraceListIterator_ itr = frames_.begin(); itr != frames_.end(); itr++) {
    const FrameBrace_& brace(*itr);
    for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
      const gsl_vector_complex* observation = brace[channelX];
      for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
	gsl_complex current = gsl_vector_complex_get(observation, subbandX);

	if (sampleX >= lowerN_) {
	  gsl_complex dereverb;
	  const gsl_vector_complex* lags = get_lags_(subbandX, sampleX - lowerN_);
	  gsl_blas_zdotc(Gn_[channelX][subbandX], lags, &dereverb);
	  current = gsl_complex_sub(current, dereverb);
	}

	double thetan = gsl_complex_abs(current);
	if (thetan < subband_floor_) {
	  thetan = subband_floor_;
	}

	gsl_matrix_set(thetan_[channelX], sampleX, subbandX, thetan * thetan);
      }
    }
    sampleX++;
  }
}

void MultiChannelWPEDereverberation::load_R_()
{
  for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
    gsl_matrix_complex* R = R_[channelX];
    double maximumDiagonal = 0.0;
    for (unsigned componentX = 0; componentX < totalPredictionN_; componentX++) {
      double diag = gsl_complex_abs(gsl_matrix_complex_get(R, componentX, componentX));
      if (diag > maximumDiagonal) maximumDiagonal = diag;
    }

    for (unsigned componentX = 0; componentX < totalPredictionN_; componentX++) {
      double diag = gsl_complex_abs(gsl_matrix_complex_get(R, componentX, componentX)) + maximumDiagonal * load_factor_;
      gsl_matrix_complex_set(R, componentX, componentX, gsl_complex_rect(diag, 0.0));
    }
  }
}

void MultiChannelWPEDereverberation::estimate_Gn_()
{
  for (unsigned iterationX = 0; iterationX < iterationsN_; iterationX++) {
    calc_Thetan_();
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {

      if ((subbandX > lower_bandWidthN_) && (subbandX < upper_bandWidthN_)) continue;

      calc_Rr_(subbandX);
      load_R_();
      for (unsigned channelX = 0; channelX < channelsN_; channelX++) {
	gsl_linalg_complex_cholesky_decomp(R_[channelX]);
	gsl_linalg_complex_cholesky_solve(R_[channelX], r_[channelX], Gn_[channelX][subbandX]);

	if (subbandX == 100) {
	  double sum = 0.0;
	  for (unsigned componentX = 0; componentX < predictionN_; componentX++) {
	    double gn = gsl_complex_abs(gsl_vector_complex_get(Gn_[channelX][subbandX], componentX));
	    sum += gn * gn;
	  }

	  double wng = 10.0 * log10(sum);
	  printf("Channel %d: Iteration %d Subband %4d WNG %6.2f\n", channelX, iterationX, subbandX, wng);
	}
      }
    }
  }
}

void MultiChannelWPEDereverberation::next_speaker()
{
  reset();
  for (unsigned channelX = 0; channelX < channelsN_; channelX++)
    for (unsigned n = 0; n < size(); n++)
      gsl_vector_complex_set_zero(Gn_[channelX][n]);
}


// ----- methods for class `MultiChannelWPEDereverberationFeature' -----
//
MultiChannelWPEDereverberationFeature::
MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm)
  : VectorComplexFeatureStream(source->size(), nm), source_(source), channelX_(channelX)
{
}

MultiChannelWPEDereverberationFeature::~MultiChannelWPEDereverberationFeature() { }

const gsl_vector_complex* MultiChannelWPEDereverberationFeature::next(int frame_no) {
  return source_->get_output(channelX_, frame_no);
}

void MultiChannelWPEDereverberationFeature::reset() { source_->reset(); }
