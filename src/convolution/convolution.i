/**
 * @file convolution.i
 * @brief Block convolution realization of an LTI system with the FFT.Time delay estimation
 * @author John McDonough
 */

%module(package="btk20") convolution

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "feature/feature.h"
#include "convolution/convolution.h"
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
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

// ----- definition for class `OverlapAdd' -----
//
%ignore OverlapAdd;
class OverlapAdd : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  OverlapAdd(VectorFloatFeatureStreamPtr& samp, const gsl_vector* impulseResponse, unsigned fftLen = 0,
             const String& nm = "Overlap Add");
  ~OverlapAdd();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();
};

class OverlapAddPtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") OverlapAddPtr;
public:
  %extend {
    OverlapAddPtr(VectorFloatFeatureStreamPtr& samp, const gsl_vector* impulseResponse, unsigned fftLen = 0,
                  const String& nm = "Overlap Add") {
      return new OverlapAddPtr(new OverlapAdd(samp, impulseResponse, fftLen, nm));
    }

    OverlapAddPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverlapAdd* operator->();
};


// ----- definition for class `OverlapSave' -----
//
%ignore OverlapSave;
class OverlapSave : public VectorFloatFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
 public:
  OverlapSave(VectorFloatFeatureStreamPtr& samp,
              const gsl_vector* impulseResponse, const String& nm = "Overlap Save");

  ~OverlapSave();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();
};

class OverlapSavePtr : public VectorFloatFeatureStreamPtr {
  %feature("kwargs") OverlapSavePtr;
public:
  %extend {
    OverlapSavePtr(VectorFloatFeatureStreamPtr& samp,
                   const gsl_vector* impulseResponse, const String& nm = "Overlap Save") {
      return new OverlapSavePtr(new OverlapSave(samp, impulseResponse, nm));
    }

    OverlapSavePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverlapSave* operator->();
};


%rename(__str__) print;
%ignore *::print();
