//  Module:  btk20.matrix
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs

%{
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_matrix_complex_double.h>
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

%include typemaps.i

// gsl_matrix_uchar typemaps
%typemap(in) gsl_matrix_uchar* %{
  PyArrayObject *_PyMatrix$argnum;
  gsl_matrix_uchar_view matrix$argnum;
  {
    _PyMatrix$argnum = NULL;
    if ($input == Py_None) {
      $1 = NULL;
    } else {
      _PyMatrix$argnum = (PyArrayObject*)
        PyArray_ContiguousFromObject($input, PyArray_CHAR, 2, 2);
      if (_PyMatrix$argnum == NULL)
        return NULL;
      matrix$argnum
        = gsl_matrix_uchar_view_array((unsigned char*)_PyMatrix$argnum->data,
				      _PyMatrix$argnum->dimensions[0],
				      _PyMatrix$argnum->dimensions[1]);
      $1 = &matrix$argnum.matrix;
    }
  }
%}

%typemap(freearg) gsl_matrix_uchar* {
  if (_PyMatrix$argnum != NULL)
    Py_DECREF(_PyMatrix$argnum);
}

%typemap(out) gsl_matrix_uchar* {
  int dims[2];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size1;
    dims[1] = $1->size2;
    $result = PyArray_FromDimsAndData(2, dims, PyArray_CHAR, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// gsl_matrix_uint typemaps
%typemap(in) gsl_matrix_uint* %{
  PyArrayObject *_PyMatrix$argnum;
  gsl_matrix_uint_view matrix$argnum;
  {
    if ($input == Py_None) {
      _PyMatrix$argnum = NULL;
      $1 = NULL;
    } else {
      _PyMatrix$argnum = (PyArrayObject*)
	PyArray_ContiguousFromObject($input, PyArray_INT, 2, 2);
      if (_PyMatrix$argnum == NULL)
	return NULL;
      matrix$argnum
	= gsl_matrix_uint_view_array((unsigned int*)_PyMatrix$argnum->data,
				     _PyMatrix$argnum->dimensions[0],
				     _PyMatrix$argnum->dimensions[1]);    
      $1 = &matrix$argnum.matrix;
    }
  }
%}

%typemap(freearg) gsl_matrix_uint* {
  if (_PyMatrix$argnum != NULL)
    Py_DECREF(_PyMatrix$argnum);
}

%typemap(out) gsl_matrix_uint* {
  int dims[2];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size1;
    dims[1] = $1->size2;
    $result = PyArray_FromDimsAndData(2, dims, PyArray_INT, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// gsl_matrix typemaps
%typemap(in) gsl_matrix* %{
  PyArrayObject *_PyMatrix$argnum;
  gsl_matrix_view matrix$argnum;
  {
    if ($input == Py_None) {
      _PyMatrix$argnum = NULL;
      $1 = NULL;
    } else {
      _PyMatrix$argnum = (PyArrayObject*)
	PyArray_ContiguousFromObject($input, PyArray_DOUBLE, 2, 2);
      if (_PyMatrix$argnum == NULL)
	return NULL;
      matrix$argnum
	= gsl_matrix_view_array((double*)_PyMatrix$argnum->data,
				_PyMatrix$argnum->dimensions[0],
				_PyMatrix$argnum->dimensions[1]);    
      $1 = &matrix$argnum.matrix;
    }
  }
%}

%typemap(freearg) gsl_matrix* {
  if (_PyMatrix$argnum != NULL)
    Py_DECREF(_PyMatrix$argnum);
}

%typemap(out) gsl_matrix* {
  int dims[2];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size1;
    dims[1] = $1->size2;
    $result = PyArray_FromDimsAndData(2, dims, PyArray_DOUBLE, (char*)$1->data);
    // ((PyArrayObject*)$result)->flags |= 8;
  }
}

// gsl_matrix_complex typemaps
%typemap(in) gsl_matrix_complex* %{
  PyArrayObject *_PyMatrix$argnum;
  gsl_matrix_complex_view matrix$argnum;
  {
    if ($input == Py_None) {
      _PyMatrix$argnum = NULL;
      $1 = NULL;
    } else {
      _PyMatrix$argnum = (PyArrayObject*)
	PyArray_ContiguousFromObject($input, PyArray_CDOUBLE, 2, 2);
      if (_PyMatrix$argnum == NULL)
	return NULL;
      matrix$argnum
	= gsl_matrix_complex_view_array((double*)_PyMatrix$argnum->data,
					_PyMatrix$argnum->dimensions[0],
					_PyMatrix$argnum->dimensions[1]);
      $1 = &matrix$argnum.matrix;
    }
  }
%}

%typemap(freearg) gsl_matrix_complex* {
  if (_PyMatrix$argnum != NULL)
    Py_DECREF(_PyMatrix$argnum);
}

%typemap(out) gsl_matrix_complex* {
  int dims[2];
  if ($1 == NULL) {
    $result = Py_None;
  } else {
    dims[0] = $1->size1;
    dims[1] = $1->size2;
    $result = PyArray_FromDimsAndData(2, dims, PyArray_CDOUBLE, (char*)$1->data);
    //  ((PyArrayObject*)$result)->flags |= 8;
  }
}
