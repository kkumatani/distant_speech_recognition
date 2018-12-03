//  Module:  btk20.complex
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs

%{
#include <gsl/gsl_complex_math.h>
%}

%typemap(in) gsl_complex {
  Py_complex temp;
  if (!PyComplex_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a complex");
    return NULL;
  }
  temp = PyComplex_AsCComplex($input);
  $1.dat[0] = temp.real;
  $1.dat[1] = temp.imag;
};

%typemap(out) gsl_complex {
  Py_complex temp;
  temp.real = $1.dat[0];
  temp.imag = $1.dat[1];
  $result = PyComplex_FromCComplex(temp);
}


%typemap(in) gsl_complex * (gsl_complex *tempComplex){
  /* Complex Pointer (in) */
  Py_complex temp;
  tempComplex = new(gsl_complex);
  tempComplex->dat[0] = temp.real;
  tempComplex->dat[1] = temp.imag;
  $1 = tempComplex;
};

%typemap(out) gsl_complex * {
  /* Complex Pointer (out) */
  Py_complex temp;
  temp.real = $1->dat[0];
  temp.imag = $1->dat[1];
  $result = PyComplex_FromCComplex(temp);
}

%typemap(argout) gsl_complex * {
  /* Complex Pointer (argout) */
  Py_complex temp;
  temp.real = $1->dat[0];
  temp.imag = $1->dat[1];
  $result = t_output_helper($result, PyComplex_FromCComplex(temp));
}

//%include gsl/gsl_complex_math.h
