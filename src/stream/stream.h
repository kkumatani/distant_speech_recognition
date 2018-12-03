/**
 * @file stream.h
 * @brief Representation of feature streams.
 * @author John McDonough.
 */

#ifndef STREAM_H
#define STREAM_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include "common/refcount.h"

// ----- interface class for 'FeatureStream' -----
//
template <typename Type, typename item_type>
class FeatureStream : public Countable {
 public:
  virtual ~FeatureStream();

  const String& name() const { return name_; }
  unsigned      size() const { return size_; }

  virtual const Type* next(int frame_no = -5) = 0;
  const Type* current() {
    if (frame_no_ < 0)
      throw jconsistency_error("Frame index (%d) < 0.", frame_no_);
    return next(frame_no_);
  }

  bool is_end(){ return is_end_; };

  virtual void reset() {
    frame_no_ = frame_reset_no_;
    is_end_ = false;
  }

  virtual int frame_no() const { return frame_no_; }
  size_t itemsize() { return sizeof(item_type); };

 protected:
  FeatureStream(unsigned sz, const String& nm);
  void gsl_vector_set_(Type *vector, int index, item_type value);
  void increment_() { frame_no_++; }

  const int					frame_reset_no_;
  const unsigned				size_;
  int						frame_no_; /*!< lapse time after reset() */
  Type*						vector_;
  bool                                          is_end_;

 private:
  const String					name_;
};


// ----- declare partial specializations of 'FeatureStream' -----
//
template<> FeatureStream<gsl_vector_char, char>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_short, short>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_float, float>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector, double>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_complex, gsl_complex>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_char, char>::~FeatureStream();
template<> FeatureStream<gsl_vector_short, short>::~FeatureStream();
template<> FeatureStream<gsl_vector_float, float>::~FeatureStream();
template<> FeatureStream<gsl_vector, double>::~FeatureStream();
template<> FeatureStream<gsl_vector_complex, gsl_complex>::~FeatureStream();
template<> void FeatureStream<gsl_vector_char, char>::gsl_vector_set_(gsl_vector_char *vector, int index, char value);
template<> void FeatureStream<gsl_vector_short, short>::gsl_vector_set_(gsl_vector_short *vector, int index, short value);
template<> void FeatureStream<gsl_vector_float, float>::gsl_vector_set_(gsl_vector_float *vector, int index, float value);
template<> void FeatureStream<gsl_vector, double>::gsl_vector_set_(gsl_vector *vector, int index, double value);
template<> void FeatureStream<gsl_vector_complex, gsl_complex>::gsl_vector_set_(gsl_vector_complex *vector, int index, gsl_complex value);


typedef FeatureStream<gsl_vector_char, char>		VectorCharFeatureStream;
typedef FeatureStream<gsl_vector_short, short>		VectorShortFeatureStream;
typedef FeatureStream<gsl_vector_float, float>		VectorFloatFeatureStream;
typedef FeatureStream<gsl_vector, double>		VectorFeatureStream;
typedef FeatureStream<gsl_vector_complex, gsl_complex>	VectorComplexFeatureStream;

typedef refcountable_ptr<VectorCharFeatureStream>	VectorCharFeatureStreamPtr;
typedef refcountable_ptr<VectorShortFeatureStream>	VectorShortFeatureStreamPtr;
typedef refcountable_ptr<VectorFloatFeatureStream>	VectorFloatFeatureStreamPtr;
typedef refcountable_ptr<VectorFeatureStream>		VectorFeatureStreamPtr;
typedef refcountable_ptr<VectorComplexFeatureStream>	VectorComplexFeatureStreamPtr;

#endif
