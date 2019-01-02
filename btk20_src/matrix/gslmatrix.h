/**
 * @file matrix.i
 * @brief Wrapper for GSL matrix objects.
 * @author John McDonough, Kenichi Kumatani
 */

#ifndef GSLMATRIX_H
#define GSLMATRIX_H

#include <gsl/gsl_matrix.h>

#define GSL_MATRIX_NROWS(x) ((x)->size2)
#define GSL_MATRIX_NCOLS(x) ((x)->size2)

gsl_matrix_float* gsl_matrix_float_resize(gsl_matrix_float* m, size_t size1, size_t size2);

void gsl_matrix_float_set_cosine(gsl_matrix_float* m, size_t i, size_t j, int type);

gsl_matrix_float* gsl_matrix_float_load(gsl_matrix_float* m, const char* filename, bool old = false);

gsl_vector_float* gsl_vector_float_load(gsl_vector_float* m, const char* filename, bool old = false);

#endif
