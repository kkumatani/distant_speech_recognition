/**
 * @file dereverberation.i
 * @brief Single- and multi-channel dereverberation base on linear prediction in the subband domain.
 * @author John McDonough
 */

%module(package="btk20") dereverberation

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "dereverberation/dereverberation.h"
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
from btk20 import feature
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}


// ----- definition for class `SingleChannelWPEDereverberationFeature' -----
//
%ignore SingleChannelWPEDereverberationFeature;
class SingleChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") next_speaker;
  %feature("kwargs") estimate_filter;
  %feature("kwargs") print_objective_func;
 public:
  SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0, double samplerate = 16000.0, const String& nm = "SingleChannelWPEDereverberationFeature");
  ~SingleChannelWPEDereverberationFeature();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
  void next_speaker();
  unsigned estimate_filter(int start_frame_no = 0, int frame_num = -1);
  void print_objective_func(int subband_no);

#ifdef ENABLE_LEGACY_BTK_API
  void nextSpeaker();
#endif
};

class SingleChannelWPEDereverberationFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") SingleChannelWPEDereverberationFeaturePtr;
public:
  %extend {
    SingleChannelWPEDereverberationFeaturePtr(VectorComplexFeatureStreamPtr& samples,
                                              unsigned lower_num,
                                              unsigned upper_num,
                                              unsigned iterations_num = 2,
                                              double load_db = -20.0,
                                              double band_width = 0.0,
                                              double samplerate = 16000.0,
                                              const String& nm = "SingleChannelWPEDereverberationFeature") {
      return new SingleChannelWPEDereverberationFeaturePtr(new SingleChannelWPEDereverberationFeature(samples, lower_num, upper_num, iterations_num, load_db, band_width, samplerate, nm));
    }

    SingleChannelWPEDereverberationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SingleChannelWPEDereverberationFeature* operator->();
};


// ----- definition for class `MultiChannelWPEDereverberation' -----
// 
%ignore MultiChannelWPEDereverberation;
class MultiChannelWPEDereverberation {
  %feature("kwargs") next;
  %feature("kwargs") reset;
  %feature("kwargs") size;
  %feature("kwargs") set_input;
  %feature("kwargs") get_output;
  %feature("kwargs") calc_every_channel_output;
  %feature("kwargs") estimate_filter;
  %feature("kwargs") reset_filter;
  %feature("kwargs") next_speaker;
  %feature("kwargs") print_objective_fun;
  %feature("kwargs") frame_no;
#ifdef ENABLE_LEGACY_BTK_API
  %feature("kwargs") setInput;
  %feature("kwargs") getOutput;
  %feature("kwargs") nextSpeaker();
#endif
public:
  MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0, double diagonal_bias = 0.0, double samplerate = 16000.0);
  ~MultiChannelWPEDereverberation();
  void reset();
  unsigned size() const;
  void set_input(VectorComplexFeatureStreamPtr& samples);
  const gsl_vector_complex* get_output(unsigned channelX);
  gsl_vector_complex** calc_every_channel_output(int frame_no = -5);
  unsigned estimate_filter(int start_frame_no = 0, int frame_num = -1);
  void reset_filter();
  void next_speaker();
  void print_objective_func(int subband_no);
  int frame_no() const;

#ifdef ENABLE_LEGACY_BTK_API
  void setInput(VectorComplexFeatureStreamPtr& samples);
  const gsl_vector_complex* getOutput(unsigned channelX, int frame_no = -1);
  void nextSpeaker();
#endif
};

class MultiChannelWPEDereverberationPtr {
  %feature("kwargs") MultiChannelWPEDereverberationPtr;
public:
  %extend {
    MultiChannelWPEDereverberationPtr(unsigned subbands_num,
                                      unsigned channels_num,
                                      unsigned lower_num,
                                      unsigned upper_num,
                                      unsigned iterations_num = 2,
                                      double load_db = -20.0,
                                      double band_width = 0.0,
                                      double diagonal_bias = 0.0,
                                      double samplerate = 16000.0) {
      return new MultiChannelWPEDereverberationPtr(new MultiChannelWPEDereverberation(subbands_num, channels_num, lower_num, upper_num, iterations_num, load_db, band_width, diagonal_bias, samplerate));
    }

    MultiChannelWPEDereverberationPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiChannelWPEDereverberation* operator->();
};


// ----- definition for class `MultiChannelWPEDereverberationFeature' -----
//
%ignore MultiChannelWPEDereverberationFeature;
class MultiChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channel_no, unsigned primaryChannelX = 0, const String& nm = "MultiChannelWPEDereverberationFeature");
  ~MultiChannelWPEDereverberationFeature();
  virtual const gsl_vector_complex* next(int frame_no = -5);
  virtual void reset();
};

class MultiChannelWPEDereverberationFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") MultiChannelWPEDereverberationFeaturePtr;
public:
  %extend {
    MultiChannelWPEDereverberationFeaturePtr(MultiChannelWPEDereverberationPtr& source,
                                             unsigned channel_no,
                                             unsigned primary_channel_no = 0,
                                             const String& nm = "MultiChannelWPEDereverberationFeature") {
      return new MultiChannelWPEDereverberationFeaturePtr(new MultiChannelWPEDereverberationFeature(source, channel_no, primary_channel_no, nm));
    }

    MultiChannelWPEDereverberationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiChannelWPEDereverberationFeature* operator->();
};

%rename(__str__) print;
%ignore *::print();
