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
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker(){ next_speaker(); }
#endif

private:
  static const double					subband_floor_;

  void fill_buffer_();
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
  bool							first_frame_;
  unsigned						framesN_;
  const double						load_factor_;
  const unsigned					lower_bandWidthN_;
  const unsigned					upper_bandWidthN_;

  Samples_						yn_;
  gsl_matrix*						thetan_;
  gsl_vector_complex**					gn_;
  gsl_matrix_complex*					R_;
  gsl_vector_complex*					r_;
  gsl_vector_complex*					lag_samples_;
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
  MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0);

  ~MultiChannelWPEDereverberation();

  void reset();
  unsigned size() const { return subbandsN_; }
  void set_input(VectorComplexFeatureStreamPtr& samples);
  const gsl_vector_complex* get_output(unsigned channelX, int frame_no = -5);
  void next_speaker();

#ifdef ENABLE_LEGACY_BTK_API
  void setInput(VectorComplexFeatureStreamPtr& samples){ set_input(samples); }
  const gsl_vector_complex* getOutput(unsigned channelX, int frame_no = -5){ return get_output(channelX, frame_no); }
  void nextSpeaker(){ next_speaker(); }
#endif

private:
  static const double					subband_floor_;

  void fill_buffer_();
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

  bool							first_frame_;
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
};

typedef refcountable_ptr<MultiChannelWPEDereverberation> MultiChannelWPEDereverberationPtr;


// ----- definition for class `MultiChannelWPEDereverberationFeature' -----
//
class MultiChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
public:
  MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm = "MultiChannelWPEDereverberationFeature");

  ~MultiChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();

private:
  MultiChannelWPEDereverberationPtr			source_;
  const unsigned					channelX_;
};

typedef Inherit<MultiChannelWPEDereverberationFeature, VectorComplexFeatureStreamPtr> MultiChannelWPEDereverberationFeaturePtr;

#endif // DEREVERBERATION_H
