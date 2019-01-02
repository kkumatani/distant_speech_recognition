//  Module:  btk20.vector
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs.

%{
#include <gsl/gsl_vector.h>
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

// typemaps for 'gsl_vector_char'
//
%typemap(in) gsl_vector_char* %{
  PyArrayObject *_PyVector$argnum;
  gsl_vector_char vector$argnum;
  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_CHAR, 1, 1);
      if (_PyVector$argnum == NULL)
        return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (char*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
  }
%}

%typemap(freearg) gsl_vector_char* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector_char* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_CHAR);
      for (int i=0; i < $1->size; i++) {
        ((PyArrayObject*)$result)->data[i] = gsl_vector_char_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_CHAR, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// typemaps for 'gsl_vector_uchar'
//
%typemap(in) gsl_vector_uchar* %{
  PyArrayObject *_PyVector$argnum;
  gsl_vector_uchar vector$argnum;
  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
	PyArray_ContiguousFromObject($input, PyArray_CHAR, 1, 1);
      if (_PyVector$argnum == NULL)
	return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (unsigned char*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
  }
%}

%typemap(freearg) gsl_vector_uchar* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector_uchar* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_CHAR);
      for (int i=0; i < $1->size; i++) {
	((PyArrayObject*)$result)->data[i] = gsl_vector_uchar_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_CHAR, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// typemaps for 'gsl_vector_short'
//
%typemap(in) gsl_vector_short* %{
  PyArrayObject *_PyVector$argnum ;
  gsl_vector_short vector$argnum;
  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_SHORT, 1, 1);
      if (_PyVector$argnum == NULL)
        return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (short*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
  }
%}

%typemap(freearg) gsl_vector_short* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector_short* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_SHORT);
      for (int i=0; i < $1->size; i++) {
        ((short*)((PyArrayObject*)$result)->data)[i] = gsl_vector_short_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_SHORT, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}


// typemaps for 'gsl_vector_float'
//
%typemap(in) gsl_vector_float* %{
  PyArrayObject *_PyVector$argnum;
  gsl_vector_float vector$argnum;

  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_FLOAT, 1, 1);
      if (_PyVector$argnum == NULL)
        return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (float*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
  }
%}

%typemap(freearg) gsl_vector_float* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector_float* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_FLOAT);
      for (int i=0; i < $1->size; i++) {
        ((float*)((PyArrayObject*)$result)->data)[i] = gsl_vector_float_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_FLOAT, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// typemaps for 'gsl_vector'
//
%typemap(in) gsl_vector* %{
  PyArrayObject *_PyVector$argnum;
  gsl_vector vector$argnum;

  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_DOUBLE, 1, 1);
      if (_PyVector$argnum == NULL)
        return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (double*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
 }
%}

%typemap(freearg) gsl_vector* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_DOUBLE);
      for (int i=0; i < $1->size; i++) {
        ((double*)((PyArrayObject*)$result)->data)[i] = gsl_vector_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_DOUBLE, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}


// typemaps for 'gsl_vector_complex'
//
%typemap(in) gsl_vector_complex* %{
  PyArrayObject *_PyVector$argnum;
  gsl_vector_complex vector$argnum;

  {
    if ($input == Py_None) {
      _PyVector$argnum = NULL;
      $1 = NULL;
    } else {
      int len;
      _PyVector$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_CDOUBLE, 1, 1);
      if (_PyVector$argnum == NULL)
        return NULL;
      len = _PyVector$argnum->dimensions[0];
      vector$argnum.size = len;
      vector$argnum.stride = 1;
      vector$argnum.data = (double*)_PyVector$argnum->data;
      vector$argnum.block = NULL;
      vector$argnum.owner = 1;
      $1 = &vector$argnum;
    }
  }
%}

%typemap(freearg) gsl_vector_complex* {
  if (_PyVector$argnum != NULL)
    Py_DECREF(_PyVector$argnum);
}

%typemap(out) gsl_vector_complex* {
  int dims[1];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size;
    if ($1->stride != 1) {
      $result = PyArray_FromDims(1, dims, PyArray_CDOUBLE);
      for (int i=0; i < $1->size; i++) {
        ((gsl_complex*)((PyArrayObject*)$result)->data)[i] = gsl_vector_complex_get($1, i);
      }
    } else
      $result = PyArray_FromDimsAndData(1, dims, PyArray_CDOUBLE, (char*)$1->data);
    //  ((PyArrayObject*)$result)->flags |= 8;
  }
}
