/*
 * @file stream.cc
 * @brief Representation of feature streams.
 * @author John McDonough
 */

#include "stream/stream.h"


// ----- partial specializations for class template 'FeatureStream' -----
//
template<> FeatureStream<gsl_vector_char, char>::
FeatureStream(unsigned sz, const String& nm) :
  frame_reset_no_(-1), size_(sz), frame_no_(-1), vector_(gsl_vector_char_calloc(size_)), is_end_(false),
  name_(nm)
{
  gsl_vector_char_set_zero(vector_);
}

template<> FeatureStream<gsl_vector_short, short>::
FeatureStream(unsigned sz, const String& nm) :
  frame_reset_no_(-1), size_(sz), frame_no_(-1), vector_(gsl_vector_short_calloc(size_)), is_end_(false),
  name_(nm)
{
  gsl_vector_short_set_zero(vector_);
}

template<> FeatureStream<gsl_vector_float, float>::
FeatureStream(unsigned sz, const String& nm) :
  frame_reset_no_(-1), size_(sz), frame_no_(-1), vector_(gsl_vector_float_calloc(size_)), is_end_(false),
  name_(nm)
{
  gsl_vector_float_set_zero(vector_);
}

template<> FeatureStream<gsl_vector, double>::
FeatureStream(unsigned sz, const String& nm) :
  frame_reset_no_(-1), size_(sz), frame_no_(-1), vector_(gsl_vector_calloc(size_)), is_end_(false),
  name_(nm)
{
  gsl_vector_set_zero(vector_);
}

template<> FeatureStream<gsl_vector_complex, gsl_complex>::
FeatureStream(unsigned sz, const String& nm) :
  frame_reset_no_(-1), size_(sz), frame_no_(-1), vector_(gsl_vector_complex_calloc(size_)),is_end_(false),
  name_(nm)
{
  gsl_vector_complex_set_zero(vector_);
}

template<> FeatureStream<gsl_vector_char, char>::~FeatureStream()    { gsl_vector_char_free(vector_); }
template<> FeatureStream<gsl_vector_short, short>::~FeatureStream()   { gsl_vector_short_free(vector_); }
template<> FeatureStream<gsl_vector_float, float>::~FeatureStream()   { gsl_vector_float_free(vector_); }
template<> FeatureStream<gsl_vector, double>::~FeatureStream()         { gsl_vector_free(vector_); }
template<> FeatureStream<gsl_vector_complex, gsl_complex>::~FeatureStream() { gsl_vector_complex_free(vector_); }

template<>
void FeatureStream<gsl_vector_char, char>::gsl_vector_set_(gsl_vector_char *vector, int index, char value) {
  gsl_vector_char_set(vector, index, value);
};

template<>
void FeatureStream<gsl_vector_short, short>::gsl_vector_set_(gsl_vector_short *vector, int index, short value) {
  gsl_vector_short_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector_float, float>::gsl_vector_set_(gsl_vector_float *vector, int index, float value) {
  gsl_vector_float_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector, double>::gsl_vector_set_(gsl_vector *vector, int index, double value) {
  gsl_vector_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector_complex, gsl_complex>::gsl_vector_set_(gsl_vector_complex *vector, int index, gsl_complex value) {
  gsl_vector_complex_set(vector, index, value);
};
