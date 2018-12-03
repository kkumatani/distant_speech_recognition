/**
 * @file aec.i
 * @brief Acoustic echo cancelation based on either NLMS or Kalman filter algorithms
 * @author John McDonough, Wei Chu and Kenichi Kumatani
 */

%module(package="btk20") aec

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "square_root/square_root.h"
#include "aec/aec.h"
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


// ----- definition for class `NLMSAcousticEchoCancellationFeature' -----
// 
%ignore NLMSAcousticEchoCancellationFeature;
class NLMSAcousticEchoCancellationFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& original, const VectorComplexFeatureStreamPtr& distorted,
				  double delta = 100.0, double epsilon = 1.0E-04, double threshold = 100.0, const String& nm = "AEC");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class NLMSAcousticEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") NLMSAcousticEchoCancellationFeaturePtr;
 public:
  %extend {
    NLMSAcousticEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& original,
                                           const VectorComplexFeatureStreamPtr& distorted,
                                           double delta = 100.0,
                                           double epsilon = 1.0E-04,
                                           double threshold = 100.0,
                                           const String& nm = "AEC") {
      return new NLMSAcousticEchoCancellationFeaturePtr(new NLMSAcousticEchoCancellationFeature(original, distorted, delta, epsilon, threshold, nm));
    }

    NLMSAcousticEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NLMSAcousticEchoCancellationFeature* operator->();
};


// ----- definition for class `KalmanFilterEchoCancellationFeature' -----
//
%ignore KalmanFilterEchoCancellationFeature;
class KalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, double beta = 0.95, double sigmau2 = 10e-4, double sigma2 = 5.0, double threshold = 100.0, const String& nm = "KFEchoCanceller");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class KalmanFilterEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") KalmanFilterEchoCancellationFeaturePtr;
 public:
  %extend {
    KalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played,
                                           const VectorComplexFeatureStreamPtr& recorded,
                                           double beta = 0.95,
                                           double sigmau2 = 10e-4,
                                           double sigma2 = 5.0,
                                           double threshold = 100.0,
                                           double crossCorrTh = 0.5,
                                           const String& nm = "KFEchoCanceller")
    {
      return new KalmanFilterEchoCancellationFeaturePtr(new KalmanFilterEchoCancellationFeature(played, recorded, beta, sigma2, threshold, nm));
    }

    KalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  KalmanFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `BlockKalmanFilterEchoCancellationFeature' -----
//
%ignore BlockKalmanFilterEchoCancellationFeature;
class BlockKalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double threshold = 100.0,
					   double amp4play = 1.0,
					   const String& nm = "BlockKFEchoCanceller");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class BlockKalmanFilterEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
  %feature("kwargs") BlockKalmanFilterEchoCancellationFeaturePtr;
 public:
  %extend {
    BlockKalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played,
                                                const VectorComplexFeatureStreamPtr& recorded,
                                                unsigned sampleN = 1,
                                                double beta = 0.95,
                                                double sigmau2 = 10e-4,
                                                double sigmak2 = 5.0,
                                                double threshold = 100.0,
                                                double amp4play = 1.0,
                                                const String& nm = "BlockKFEchoCanceller") {
      return new BlockKalmanFilterEchoCancellationFeaturePtr(new BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, threshold, amp4play, nm));
    }

    BlockKalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockKalmanFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `InformationFilterEchoCancellationFeature' -----
//
%ignore InformationFilterEchoCancellationFeature;
class InformationFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
					   double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
					   double amp4play = 1.0,
					   const String& nm = "DTDBlockKFEchoCanceller");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class InformationFilterEchoCancellationFeaturePtr : public BlockKalmanFilterEchoCancellationFeaturePtr {
  %feature("kwargs") InformationFilterEchoCancellationFeaturePtr;
 public:
  %extend {
    InformationFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played,
                                                const VectorComplexFeatureStreamPtr& recorded,
                                                unsigned sampleN = 1,
                                                double beta = 0.95,
                                                double sigmau2 = 10e-4,
                                                double sigmak2 = 5.0,
                                                double snrTh = 2.0,
                                                double engTh = 100.0,
                                                double smooth = 0.9,
                                                double loading = 1.0e-02,
                                                double amp4play = 1.0,
                                                const String& nm = "DTDBlockKFEchoCanceller") {
      return new InformationFilterEchoCancellationFeaturePtr(new InformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm));
    }

    InformationFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  InformationFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
%ignore SquareRootInformationFilterEchoCancellationFeature;
class SquareRootInformationFilterEchoCancellationFeature : public InformationFilterEchoCancellationFeature {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						     unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
						     double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
                                                     double amp4play = 1.0,
						     const String& nm = "Square Root Information Filter Echo Cancellation Feature");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class SquareRootInformationFilterEchoCancellationFeaturePtr : public InformationFilterEchoCancellationFeaturePtr {
  %feature("kwargs") SquareRootInformationFilterEchoCancellationFeaturePtr;
 public:
  %extend {
    SquareRootInformationFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played,
                                                          const VectorComplexFeatureStreamPtr& recorded,
                                                          unsigned sampleN = 1,
                                                          double beta = 0.95,
                                                          double sigmau2 = 10e-4,
                                                          double sigmak2 = 5.0,
                                                          double snrTh = 2.0,
                                                          double engTh = 100.0,
                                                          double smooth = 0.9,
                                                          double loading = 1.0e-02,
                                                          double amp4play = 1.0,
                                                          const String& nm = "Square Root Information Filter Echo Cancellation Feature") {
      return new SquareRootInformationFilterEchoCancellationFeaturePtr(new SquareRootInformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm));
    }

    SquareRootInformationFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SquareRootInformationFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
%ignore DTDBlockKalmanFilterEchoCancellationFeature;
class DTDBlockKalmanFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
  %feature("kwargs") next;
  %feature("kwargs") reset;
public:
  DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					      unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
					      double engTh = 100.0, double smooth = 0.9,
					      double amp4play = 1.0,
					      const String& nm = "DTDBlockKFEchoCanceller");

  const gsl_vector_complex* next(int frame_no = -5) const;
  void reset();
};

class DTDBlockKalmanFilterEchoCancellationFeaturePtr : public BlockKalmanFilterEchoCancellationFeaturePtr {
  %feature("kwargs") DTDBlockKalmanFilterEchoCancellationFeaturePtr;
 public:
  %extend {
    DTDBlockKalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played,
                                                   const VectorComplexFeatureStreamPtr& recorded,
                                                   unsigned sampleN = 1,
                                                   double beta = 0.95,
                                                   double sigmau2 = 10e-4,
                                                   double sigmak2 = 5.0,
                                                   double snrTh = 2.0,
                                                   double engTh = 100.0,
                                                   double smooth = 0.9,
                                                   double amp4play = 1.0,
                                                   const String& nm = "DTDBlockKFEchoCanceller") {
      return new DTDBlockKalmanFilterEchoCancellationFeaturePtr(new DTDBlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, amp4play, nm));
    }

    DTDBlockKalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DTDBlockKalmanFilterEchoCancellationFeature* operator->();
};
