/**
 * @file pyStream.h
 * @brief Representation of Python feature streams.
 * @author John McDonough
 */

#ifndef PYSTREAM_H
#define PYSTREAM_H

#include <Python.h>
#include "stream/stream.h"
#include <numpy/arrayobject.h>
#include "common/jexception.h"
#include "common/jpython_error.h"

#ifdef THREADING
#define PYTHON_START PyGILState_STATE _state = PyGILState_Ensure()
#define PYTHON_END PyGILState_Release(_state)
#else
#define PYTHON_START
#define PYTHON_END
#endif

// ----- interface class for 'PyFeatureStream' -----
//
template <typename gsl_type, typename c_type, int numpy_type>
class PyFeatureStream : public FeatureStream<gsl_type, c_type> {
  typedef FeatureStream<gsl_type, c_type> _FeatureStream;
 public:
  PyFeatureStream(PyObject* c, const String& nm);
  ~PyFeatureStream();

  virtual const gsl_type* next(int frame_no = -5);
  virtual void reset();

 private:
  unsigned get_size_(PyObject* c) const;

  PyObject*		cont_;
  PyObject*		iter_;
};


template <typename gsl_type, typename c_type, int numpy_type>
PyFeatureStream<gsl_type, c_type, numpy_type>::PyFeatureStream(PyObject* c, const String& nm) :
  FeatureStream<gsl_type, c_type>(get_size_(c), nm), cont_(c)
{
  char iter[] = "__iter__";

  PYTHON_START;
  iter_ = PyObject_CallMethod(cont_, iter, NULL);
  if (iter_ == NULL) {
    PYTHON_END;
    throw jpython_error();
  }
  Py_INCREF(cont_); Py_INCREF(iter_);
  PYTHON_END;
}


template <typename gsl_type, typename c_type, int numpy_type>
PyFeatureStream<gsl_type, c_type, numpy_type>::~PyFeatureStream() {
  PYTHON_START;
  Py_DECREF(cont_);
  Py_DECREF(iter_);
  PYTHON_END;
}


template <typename gsl_type, typename c_type, int numpy_type>
const gsl_type* PyFeatureStream<gsl_type, c_type, numpy_type>::next(int frame_no)
{
  if (frame_no == _FeatureStream::frame_no_) return _FeatureStream::vector_;

  PYTHON_START;
  // import_array();

  bool err = false;
  c_type *data;
  PyObject* pyObj = NULL;
  PyArrayObject* pyVec = NULL;

  char next[] = "next";
  pyObj = PyObject_CallMethod(iter_, next, NULL);
  if (pyObj == NULL) {
    if (!PyErr_ExceptionMatches(PyExc_StopIteration)) {
      err = true;
      goto error;
    }
    PyErr_Clear();
    throw jiterator_error("No more samples!");
  }

  pyVec = (PyArrayObject*) PyArray_ContiguousFromObject(pyObj, numpy_type, 1, 1);
  if (pyVec == NULL) {
    err = true;
    goto error;
  }

  data = (c_type*) pyVec->data;
  for (unsigned i = 0; i < _FeatureStream::size_; i++)
    this->gsl_vector_set_(_FeatureStream::vector_, i, data[i]);

  // cleanup code
 error:
  Py_XDECREF(pyObj);
  Py_XDECREF(pyVec);
  PYTHON_END;
  if (err) throw jpython_error();
  _FeatureStream::increment_();
  return _FeatureStream::vector_;
}


template <typename gsl_type, typename c_type, int numpy_type>
void PyFeatureStream<gsl_type, c_type, numpy_type>::reset() {
  char reset[] = "reset";
  char iter[]  = "__iter__";

  PYTHON_START;
  if (PyObject_CallMethod(cont_, reset, NULL) == NULL) {
    PYTHON_END;
    throw jpython_error();
  }
  Py_DECREF(iter_);
  iter_ = PyObject_CallMethod(cont_, iter, NULL);
  if (iter_ == NULL) {
    PYTHON_END;
    throw jpython_error();
  }
  Py_INCREF(iter_);
  PYTHON_END;
  FeatureStream<gsl_type, c_type>::reset();
}


template <typename gsl_type, typename c_type, int numpy_type>
unsigned PyFeatureStream<gsl_type, c_type, numpy_type>::get_size_(PyObject* c) const {
  PYTHON_START;
  bool err = false;
  long sz;
  Py_INCREF(c);
  char size[] = "size";

  PyObject* pyObj = NULL;
  pyObj = PyObject_CallMethod(c, size, NULL);
  if (pyObj == NULL) {
    err = true;
    goto error;
  }

  sz = PyInt_AsLong(pyObj);

  // clean up
 error:
  Py_XDECREF(c);  Py_XDECREF(pyObj);
  PYTHON_END;
  if (err) throw jpython_error();

  return unsigned(sz);
}

// typedef PyFeatureStream<gsl_vector_char, char, PyArray_SBYTE>		  PyVectorCharFeatureStream;
typedef PyFeatureStream<gsl_vector_short, short, PyArray_SHORT>		  PyVectorShortFeatureStream;
typedef PyFeatureStream<gsl_vector_float, float, PyArray_FLOAT>		  PyVectorFloatFeatureStream;
typedef PyFeatureStream<gsl_vector, double, PyArray_DOUBLE>		  PyVectorFeatureStream;
typedef PyFeatureStream<gsl_vector_complex, gsl_complex, PyArray_CDOUBLE> PyVectorComplexFeatureStream;


// ----- smart pointer declarations 'PyFeatureStream' -----
//
// typedef Inherit<PyVectorCharFeatureStream,    VectorCharFeatureStreamPtr>    PyVectorCharFeatureStreamPtr;
typedef Inherit<PyVectorShortFeatureStream,   VectorShortFeatureStreamPtr>   PyVectorShortFeatureStreamPtr;
typedef Inherit<PyVectorFloatFeatureStream,   VectorFloatFeatureStreamPtr>   PyVectorFloatFeatureStreamPtr;
typedef Inherit<PyVectorFeatureStream,        VectorFeatureStreamPtr>        PyVectorFeatureStreamPtr;
typedef Inherit<PyVectorComplexFeatureStream, VectorComplexFeatureStreamPtr> PyVectorComplexFeatureStreamPtr;

#endif
