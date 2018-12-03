/**
 * @file lms.i
 * @brief Implementation of LMS algorithms.
 * @author John McDonough
 */

%module(package="btk20") lms

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "lms/lms.h"
%}

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

// ----- definition for class `FastBlockLMSFeature' -----
//
%ignore FastBlockLMSFeature;
class FastBlockLMSFeature : public VectorFloatFeatureStream {
public:
  FastBlockLMSFeature(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha = 0.0001, float gamma = 0.98, const String& nm = "Fast Block LMS Feature");

  const gsl_vector_float* next() const;
};

class FastBlockLMSFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FastBlockLMSFeaturePtr(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha = 0.0001, float gamma = 0.98, const String& nm = "Fast Block LMS Feature") {
      return new FastBlockLMSFeaturePtr(new FastBlockLMSFeature(desired, samp, alpha, gamma, nm));
    }

    FastBlockLMSFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FastBlockLMSFeature* operator->();
};


%rename(__str__) print;
%ignore *::print();
