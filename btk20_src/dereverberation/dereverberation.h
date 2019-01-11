/**
 * @file dereverberation.h
 * @brief Single- and multi-channel dereverberation base on linear prediction in the subband domain.
 * @author John McDonough
 */

#ifndef DEREVERBERATION_H
#define DEREVERBERATION_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"

/*
#include <Eigen/Cholesky>
#include <iostream>
#include <Eigen/Dense>
*/


// ----- definition for class `SingleChannelWPEDereverberationFeature' -----
//
class SingleChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
  typedef vector<gsl_vector_complex*>			Samples_;
  typedef Samples_::iterator				SamplesIterator_;

 public:
  SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0, const String& nm = "SingleChannelWPEDereverberationFeature");

  ~SingleChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  unsigned estimate_filter(int start_frame_no = 0, int frame_num = -1);
  void reset_filter();
  void next_speaker();
  void print_objective_func(int subbandX){ printing_subbandX_ = subbandX;}

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker(){ next_speaker(); }
#endif

private:
  static const double					subband_floor_;

  void fill_buffer_(int start_frame_no, int frame_num);
  void estimate_Gn_();
  void calc_Rr_(unsigned subbandX);
  void calc_Thetan_();
  void load_R_();
  unsigned set_band_width_(double bandWidth, double sampleRate);

  const gsl_vector_complex* get_lags_(unsigned subbandX, unsigned sampleX);
  VectorComplexFeatureStreamPtr				samples_;

  const unsigned					lowerN_;
  const unsigned					upperN_;
  const unsigned					predictionN_;
  const unsigned					iterationsN_;
  bool							estimated_;
  unsigned						framesN_; // no frames used for filter estimation
  const double						load_factor_;
  const unsigned					lower_bandWidthN_;
  const unsigned					upper_bandWidthN_;

  Samples_						yn_; // buffer to keep observations
  gsl_matrix*						thetan_;
  gsl_vector_complex**					gn_;
  gsl_matrix_complex*					R_;
  gsl_vector_complex*					r_;
  gsl_vector_complex*					lag_samples_;
  int                                                   printing_subbandX_;
};

typedef Inherit<SingleChannelWPEDereverberationFeature, VectorComplexFeatureStreamPtr> SingleChannelWPEDereverberationFeaturePtr;


// ----- definition for class `MultiChannelWPEDereverberation' -----
//
class MultiChannelWPEDereverberation : public Countable {
  typedef vector<VectorComplexFeatureStreamPtr>		SourceList_;
  typedef SourceList_::iterator				SourceListIterator_;

  typedef vector<gsl_vector_complex*>			FrameBrace_;
  typedef FrameBrace_::iterator				FrameBraceIterator_;

  typedef vector<FrameBrace_>				FrameBraceList_;
  typedef FrameBraceList_::iterator			FrameBraceListIterator_;

 public:
  MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0, double diagonal_bias = 0.0, double sampleRate = 16000.0);

  ~MultiChannelWPEDereverberation();

  void reset();
  unsigned size() const { return subbandsN_; }
  void set_input(VectorComplexFeatureStreamPtr& samples);
  const gsl_vector_complex* get_output(unsigned channelX);
  gsl_vector_complex** calc_every_channel_output(int frame_no = -5);
  unsigned estimate_filter(int start_frame_no = 0, int frame_num = -1);
  void reset_filter();
  void next_speaker();
  int  frame_no() const { return frame_no_; }
  void print_objective_func(int subbandX){ printing_subbandX_ = subbandX;}

#ifdef ENABLE_LEGACY_BTK_API
  void setInput(VectorComplexFeatureStreamPtr& samples){ set_input(samples); }
  const gsl_vector_complex* getOutput(unsigned channelX, int frame_no = -5){ return get_output(channelX); }
  void nextSpeaker(){ next_speaker(); }
#endif

private:
  static const double					subband_floor_;

  void fill_buffer_(int start_frame_no, int frame_num);
  void estimate_Gn_();
  void calc_Rr_(unsigned subbandX);
  void calc_Thetan_();
  void load_R_();
  unsigned set_band_width_(double bandWidth, double sampleRate);

  void increment_() { frame_no_++; }
  const gsl_vector_complex* get_lags_(unsigned subbandX, unsigned sampleX);

  SourceList_						sources_;
  const unsigned					subbandsN_;
  const unsigned					channelsN_;
  const unsigned					lowerN_;
  const unsigned					upperN_;
  const unsigned					predictionN_;
  const unsigned					iterationsN_;
  const unsigned					totalPredictionN_;

  bool							estimated_;
  unsigned						framesN_;
  const double						load_factor_;
  const unsigned					lower_bandWidthN_;
  const unsigned					upper_bandWidthN_;

  FrameBraceList_					frames_;
  gsl_matrix**						thetan_;
  gsl_vector_complex***					Gn_;
  gsl_matrix_complex**					R_;
  gsl_vector_complex**					r_;
  gsl_vector_complex*					lag_samples_;
  gsl_vector_complex**					output_;

  const int						initial_frame_no_;
  int							frame_no_;

  const double                                          diagonal_bias_;
  int                                                   printing_subbandX_;
};

typedef refcountable_ptr<MultiChannelWPEDereverberation> MultiChannelWPEDereverberationPtr;


// ----- definition for class `MultiChannelWPEDereverberationFeature' -----
//
class MultiChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
public:
  MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, unsigned primaryChannelX = 0, const String& nm = "MultiChannelWPEDereverberationFeature");

  ~MultiChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

private:
  MultiChannelWPEDereverberationPtr			source_;
  const unsigned					channelX_;
  const unsigned					primaryChannelX_; // compute the dereverbed output only when channelX_ == primaryChannelX_. Otherwise copy the precomputed output
};

typedef Inherit<MultiChannelWPEDereverberationFeature, VectorComplexFeatureStreamPtr> MultiChannelWPEDereverberationFeaturePtr;

#endif // DEREVERBERATION_H
