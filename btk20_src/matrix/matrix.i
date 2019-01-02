/**
 * @file matrix.i
 * @brief Wrapper for GSL matrix objects.
 * @author John McDonough
 */

%module(package="btk20") matrix

%{
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_matrix_float.h>
#include "matrix/gslmatrix.h"
%}

#ifdef AUTODOC
%section "Matrix", before
#endif

%include typedefs.i
%include jexception.i

#ifndef INLINE_DECL
#define INLINE_DECL extern inline
#endif

%include <gsl/gsl_matrix_double.h>
%include <gsl/gsl_matrix_float.h>

%extend gsl_matrix {
  gsl_matrix(unsigned m, unsigned n) {
    return gsl_matrix_alloc(m, n);
  }

  ~gsl_matrix() {
    gsl_matrix_free(self);
  }

  unsigned nrows() const {
    return self->size1;
  }

  unsigned ncols() const {
    return self->size2;
  }

  float __getitem__(int m, int n) {
    return gsl_matrix_get(self, m, n);
  }

  void __setitem__(float item, int n, int m) {
    gsl_matrix_set(self, m, n, item);
  }
}

%extend gsl_matrix_float {
  gsl_matrix_float(unsigned m, unsigned n) {
    return gsl_matrix_float_alloc(m, n);
  }

  ~gsl_matrix_float() {
      // gsl_matrix_float_free(self);
  }

  unsigned nrows() const {
    return self->size1;
  }

  unsigned ncols() const {
    return self->size2;
  }

  float __getitem__(int m, int n) {
    return gsl_matrix_float_get(self, m, n);
  }

  void __setitem__(float item, int n, int m) {
    gsl_matrix_float_set(self, m, n, item);
  }
}

gsl_matrix_float* gsl_matrix_float_load(gsl_matrix_float* m, const char* filename, bool old = false);

gsl_matrix_float* gsl_matrix_float_resize(gsl_matrix_float* m, size_t size1, size_t size2);
