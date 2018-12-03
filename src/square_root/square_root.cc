/**
 * @file square_root.cc
 * @brief Updating complex Cholesky factorizations.
 * @author John McDonough and Kenichi Kumatani
 */

#include <math.h>
#include "square_root.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

void vector_matrix_product(const gsl_vector_complex* vec,
			 const gsl_matrix_complex* mat, gsl_matrix* D)
{
  int cepLen = mat->size1;
  int fftLen = mat->size2;

  if (int(vec->size) != fftLen)
    throw jdimension_error("Size (%d) of 'vec' does not match 'fftLen' (%d).",
			   vec->size, fftLen);
  if ((int(D->size1) != cepLen) || (int(D->size2) != cepLen))
    throw jdimension_error("Dimensions (%d, %d) of 'D' do not match 'cepLen' (%d).",
			   D->size1, D->size2, cepLen);

  for (int n = 0; n < cepLen; n++) {
    for (int m = 0; m < cepLen; m++) {

      // add contribution of 0-th (real) element
      gsl_complex first = gsl_complex_mul(gsl_matrix_complex_get(mat, n, 0),
					  gsl_vector_complex_get(vec, 0));
      first =
	gsl_complex_mul(first,
			gsl_complex_conjugate(gsl_matrix_complex_get(mat, m, 0)));

      // add contribution of last (real) element
      gsl_complex last =
	gsl_complex_mul(gsl_matrix_complex_get(mat, n, fftLen-1),
			gsl_vector_complex_get(vec, fftLen-1));
      last =
	gsl_complex_mul(last,
			gsl_complex_conjugate(gsl_matrix_complex_get(mat, m, fftLen-1)));

      double sum = GSL_REAL(first) + GSL_REAL(last);

      // add contributions of all other (complex) elements
      for (int k = 1; k < fftLen-1; k++) {
	gsl_complex other =
	  gsl_complex_mul(gsl_matrix_complex_get(mat, n, k),
			  gsl_vector_complex_get(vec, k));
	other =
	  gsl_complex_mul(other,
			  gsl_complex_conjugate(gsl_matrix_complex_get(mat, m, k)));

	sum += 2.0 * GSL_REAL(other);
      }

      // scale final answer and insert in output matrix
      gsl_matrix_set(D, n, m, sum);
    }
  }

  /*
  for (int n = 0; n < cepLen; n++) {
    for (int m = 0; m < cepLen; m++)
      printf("%8.4e  ", gsl_matrix_get(D, n, m));
    printf("\n");
  }
  */
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
static double
calcGivensRotation(const double v1, const double v2,
		   double& c, double& s)
{
  double norm = sqrt(v1*v1 + v2*v2);

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = v1 / norm;
  s = -v2 / norm;
  
  return norm;
}

// apply a previously calculated Givens rotation
static void applyGivensRotation(const double& v1, const double& v2,
				double& c, double& s,
				double& v1p, double& v2p)
{
  v1p = c * v1 - s * v2;
  v2p = c * v2 + s * v1;
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
static gsl_complex calcGivensRotation(const gsl_complex& v1, const gsl_complex& v2,
			       gsl_complex& c, gsl_complex& s)
{
  double norm = sqrt(gsl_complex_abs2(v1) + gsl_complex_abs2(v2));

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = gsl_complex_div_real(v1, norm);
  s = gsl_complex_div_real(gsl_complex_conjugate(v2), norm);

  return gsl_complex_rect(norm, 0.0);
}

// apply a previously calculated Givens rotation
static void applyGivensRotation(const gsl_complex& v1, const gsl_complex& v2,
			 const gsl_complex& c, const gsl_complex& s,
			 gsl_complex& v1p, gsl_complex& v2p)
{
  v1p =
    gsl_complex_add(gsl_complex_mul(gsl_complex_conjugate(c), v1),
		    gsl_complex_mul(s, v2));
  v2p =
    gsl_complex_sub(gsl_complex_mul(c, v2),
		    gsl_complex_mul(gsl_complex_conjugate(s), v1));
}

// perform backward substitution on an upper triangular matrix 'lt'
void cholesky_backsub_complex(const gsl_matrix_complex* lt,
                              const gsl_vector_complex* rhs,
                              gsl_vector_complex* lhs,
                              bool conjugate)
{
#if 0
  int size = lt->size1;
  for (int i = size-1; i >= 0; i--) {
    gsl_complex result = gsl_vector_complex_get(rhs, i);
    for (int j = size-1; j > i; j--)
      result =
	gsl_complex_sub(result,
			gsl_complex_mul(gsl_vector_complex_get(lhs, j),
					gsl_matrix_complex_get(lt, j, i)));

    result = gsl_complex_div(result, gsl_matrix_complex_get(lt, i, i));
    if (conjugate)
      result = gsl_complex_conjugate(result);

    gsl_vector_complex_set(lhs, i, result);
  }
#else
  gsl_vector_complex_memcpy(lhs, rhs);
  gsl_blas_ztrsv(CblasLower, CblasTrans, CblasNonUnit, lt, lhs);
#endif
}

// perform forward substitution for a lower triangular matrix 'lt'
void cholesky_forwardsub_complex(const gsl_matrix_complex* lt,
			       const gsl_vector_complex* rhs,
			       gsl_vector_complex* lhs,
			       bool conjugate)
{
  int size = lt->size1;
  for (int i = 0; i < size; i++) {
    gsl_complex result = gsl_vector_complex_get(rhs, i);
    for (int j = 0; j < i; j++)
      result =
	gsl_complex_sub(result,
			gsl_complex_mul(gsl_vector_complex_get(lhs, j),
					gsl_matrix_complex_get(lt, i, j)));

    result = gsl_complex_div(result, gsl_matrix_complex_get(lt, i, i));
    if (conjugate)
      result = gsl_complex_conjugate(result);

    gsl_vector_complex_set(lhs, i, result);
  }
}

// perform back substitution in place on an upper triangular matrix 'A'
void cholesky_backsub(const gsl_matrix* A, gsl_vector* x)
{
  gsl_blas_dtrsv (CblasUpper, CblasNoTrans, CblasNonUnit, A, x);
}

// perform forward substitution in place on a lower triangular matrix 'A'
void cholesky_forwardsub(const gsl_matrix* A, gsl_vector* x)
{
  gsl_blas_dtrsv (CblasLower, CblasNoTrans, CblasNonUnit, A, x);
}

static gsl_complex ComplexZero = gsl_complex_rect(0.0, 0.0);

// given 'sqrt_D', 'alpha_m' and 'c_m', calculate 'A11'
// through a series of Givens rotations
void rank_one_update_cholesky_factor(gsl_matrix_complex* A11,
				 const double alpha_m,
				 const gsl_vector_complex* c_m)
{
  int sizeDm = A11->size1;

  static gsl_vector_complex* gH_m    = NULL;
  static gsl_vector_complex* scratch = NULL;
  if (gH_m == NULL) {
    gH_m    = gsl_vector_complex_calloc(sizeDm);
    scratch = gsl_vector_complex_calloc(sizeDm);
  }
  cholesky_backsub_complex(A11, c_m, gH_m, /* conjugate = */ true);
  gsl_vector_complex_set_zero(scratch);

  double norm2 = 0.0;
  for (int i = 0; i < sizeDm; i++)
    norm2 += gsl_complex_abs2(gsl_vector_complex_get(gH_m, i));

  double delta2_m = (1.0 - alpha_m * norm2) / alpha_m;

  if (delta2_m < 0.0)
    throw jarithmetic_error("rank_one_update_Cholesky_factor: delta2_m (%g) < 0.0",
			    delta2_m);
  
  gsl_complex delta_m =
    gsl_complex_rect(sqrt(delta2_m), 0.0);

  // rotate top row into last column
  for (int i = sizeDm-1; i >= 0; i--) {

    // calculate Givens rotation from first row of 'gH_m' and apply
    gsl_complex c, s;
    try {
      delta_m =
	calcGivensRotation(delta_m, gsl_vector_complex_get(gH_m, i), c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_vector_complex_set(gH_m, i, ComplexZero);

    // apply rotation to affected rows of 'A11'
    for (int j = i; j < sizeDm; j++) {
      gsl_complex v1 = gsl_vector_complex_get(scratch, j);
      gsl_complex v2 = gsl_matrix_complex_get(A11, j, i);

      gsl_complex v1p, v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_vector_complex_set(scratch, j, v1p);
      gsl_matrix_complex_set(A11, j, i, v2p);
    }
  }
}

// --------------------------------------------------------------------------

// Step 1: rotate rows of 'A12' into last column
void propagate_covar_square_root_step1(gsl_matrix* A12, gsl_matrix* A22)
{
  int sizeDm = A12->size1;
  int sizePm = A22->size1;

  // iterate over columns of 'A12'
  for (int i = sizePm-2; i >= 0; i--) {

    // calculate Givens rotation from first row of 'A12' and apply
    double v1 = gsl_matrix_get(A12, 0, sizePm-1);
    double v2 = gsl_matrix_get(A12, 0, i);

    double c, s;
    double v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_set(A12, 0, sizePm-1, v1p);
    gsl_matrix_set(A12, 0, i, 0);

    // apply rotation to remaining rows of 'A12'
    for (int j = 1; j < sizeDm; j++) {
      v1 = gsl_matrix_get(A12, j, sizePm-1);
      v2 = gsl_matrix_get(A12, j, i);

      double v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      // 'v2p' should be zero
      gsl_matrix_set(A12, j, sizePm-1, v1p);
      gsl_matrix_set(A12, j, i, 0);
    }

    // apply rotation to affected rows of 'A22'
    for (int j = i; j < sizePm; j++) {
      v1 = gsl_matrix_get(A22, j, sizePm-1);
      v2 = gsl_matrix_get(A22, j, i);

      double v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_set(A22, j, sizePm-1, v1p);
      gsl_matrix_set(A22, j, i, v2p);
    }
  }
}

// Step 2: rotate last column into leading diagonal
//     2a: iterate over rows of 'A12'
void propagate_covar_square_root_step2a(gsl_matrix* A11,
				    gsl_matrix* A12,
				    gsl_matrix* A21,
				    gsl_matrix* A22)
{
  int sizeDm = A11->size1;
  int sizePm = A22->size1;
  
  for (int i = 0; i < sizeDm; i++) {
    double v1 = gsl_matrix_get(A11, i, i);
    double v2 = gsl_matrix_get(A12, i, sizePm-1);

    double c, s;
    double v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_set(A11, i, i, v1p);
    gsl_matrix_set(A12, i, sizePm-1, 0);

    // apply rotation to remaining rows of 'A11'
    for (int j = i+1; j < sizeDm; j++) {
      v1 = gsl_matrix_get(A11, j, i);
      v2 = gsl_matrix_get(A12, j, sizePm-1);

      double v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_set(A11, j, i, v1p);
      gsl_matrix_set(A12, j, sizePm-1, v2p);
    }

    // apply rotation to all rows of 'A21'
    for (int j = 0; j < sizePm; j++) {
      v1 = 0;  // = gsl_matrix_get(A21, j, i);
      v2 = gsl_matrix_get(A22, j, sizePm-1);

      double v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_set(A21, j, i, v1p);
      gsl_matrix_set(A22, j, sizePm-1, v2p);
    }
  }
}

// Step 2: rotate last column into leading diagonal
//     2b: iterate over rows of 'A22'
void propagate_covar_square_root_step2b(gsl_matrix* A22)
{
  int sizePm = A22->size1;

  /*
  printf("Before:\n");
  for (int n = 0; n < sizePm; n++) {
    for (int m = 0; m < sizePm; m++)
      printf("%8.4e  ",fabs(gsl_matrix_get(A22, n, m)));
    printf("\n");
  }
  */

  for (int i = 0; i < sizePm-1; i++) {
    double v1 = gsl_matrix_get(A22, i, i);
    double v2 = gsl_matrix_get(A22, i, sizePm-1);

    double c, s;
    double v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_set(A22, i, i, v1p);
    gsl_matrix_set(A22, i, sizePm-1, 0);

    // apply rotation to remaining rows of 'A22'
    for (int j = i+1; j < sizePm; j++) {
      v1 = gsl_matrix_get(A22, j, i);
      v2 = gsl_matrix_get(A22, j, sizePm-1);

      double v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_set(A22, j, i, v1p);
      gsl_matrix_set(A22, j, sizePm-1, v2p);
    }
  }

  /*
  printf("After:\n");
  for (int n = 0; n < sizePm; n++) {
    for (int m = 0; m < sizePm; m++) {
      if (isnan(fabs(gsl_matrix_get(A22, n, m))))
	  printf("Problem here.");
      printf("%8.4e  ",fabs(gsl_matrix_get(A22, n, m)));
    }
    printf("\n");
  }
  printf("\n");
  */
}

static void propagate_covar_square_rootAll(gsl_matrix* A11, gsl_matrix* A12,
					gsl_matrix* A21, gsl_matrix* A22)
{
  int size1 = A12->size1;
  int size2 = A12->size2;

  for (int colX = size2 - 1; colX >= 0; colX--) {
    for (int rowX = 0; rowX < size1; rowX++) {
      double v1 = gsl_matrix_get(A11, rowX, rowX);
      double v2 = gsl_matrix_get(A12, rowX, colX);

      // calculate Givens rotation from relevant components
      // of 'A11' and 'A12'
      double c, s;
      double v1p;
      try {
	v1p = calcGivensRotation(v1, v2, c, s);
      } catch (jarithmetic_error) {
	continue;
      }

      gsl_matrix_set(A11, rowX, rowX, v1p);
      gsl_matrix_set(A12, rowX, colX, 0.0);

      // apply Givens rotation to remaining rows of 'A11' and 'A12'
      for (int j = rowX + 1; j < size1; j++) {
	v1 = gsl_matrix_get(A11, j, rowX);
	v2 = gsl_matrix_get(A12, j, colX);

	double v2p;
	applyGivensRotation(v1, v2, c, s, v1p, v2p);

	gsl_matrix_set(A11, j, rowX, v1p);
	gsl_matrix_set(A12, j, colX, v2p);
      }

      // apply Givens rotation to components of 'A21' and 'A22'
      for (int j = 0; j < size2; j++) {
	v1 = gsl_matrix_get(A21, j, rowX);
	v2 = gsl_matrix_get(A22, j, colX);

	double v2p;
	applyGivensRotation(v1, v2, c, s, v1p, v2p);

	gsl_matrix_set(A21, j, rowX, v1p);
	gsl_matrix_set(A22, j, colX, v2p);
      }
    }
  }
}

void restoreLowerTriangular(gsl_matrix* A)
{
  if (A->size1 != A->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'A' do not match.",
			   A->size1, A->size2);
  int size = A->size1;
  for (int colX = size - 1; colX > 0; colX--) {
    for (int rowX = 0; rowX < colX; rowX++) {
      double v1 = gsl_matrix_get(A, rowX, rowX);
      double v2 = gsl_matrix_get(A, rowX, colX);

      // calculate Givens rotation
      double c, s;
      double v1p;
      try {
	v1p = calcGivensRotation(v1, v2, c, s);
      } catch (jarithmetic_error) {
	continue;
      }

      gsl_matrix_set(A, rowX, rowX, v1p);
      gsl_matrix_set(A, rowX, colX, 0.0);

      // apply Givens rotation to remaining rows of 'A'
      for (int j = rowX + 1; j < size; j++) {
	v1 = gsl_matrix_get(A, j, rowX);
	v2 = gsl_matrix_get(A, j, colX);

	double v2p;
	applyGivensRotation(v1, v2, c, s, v1p, v2p);

	gsl_matrix_set(A, j, rowX, v1p);
	gsl_matrix_set(A, j, colX, v2p);
      }
    }
  }
}

void propagate_covar_square_root(gsl_matrix* A11,
			      gsl_matrix* A12,
			      gsl_matrix* A21,
			      gsl_matrix* A22, bool flag)
{
  // dimensionality check
  if (A11->size1 != A11->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Dm' do not match.",
			   A11->size1, A11->size2);
  if (A22->size1 != A22->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Pm' do not match.",
			   A22->size1, A22->size2);
  if (A11->size1 != A12->size1)
    throw jdimension_error("Rows of 'Dm' (%d) and 'A12' (%d) do not match.",
			   A11->size1, A12->size1);
  if (A22->size2 != A12->size2)
    throw jdimension_error("Columns of 'Pm' (%d) and 'A12' (%d) do not match.",
			   A22->size2, A12->size2);

  // update 'A22' and calculate 'A21'
  if (flag) {
    propagate_covar_square_root_step1(A12, A22);
    propagate_covar_square_root_step2a(A11, A12, A21, A22);
    propagate_covar_square_root_step2b(A22);
  } else {
    propagate_covar_square_rootAll(A11, A12, A21, A22);
    restoreLowerTriangular(A22);
  }
  
}

void propagate_covar_square_root_real(gsl_matrix* A11,
                                      gsl_matrix* A12,
                                      gsl_matrix* A21,
                                      gsl_matrix* A22, bool flag)
{
  propagate_covar_square_root(A11, A12, A21, A22, flag);
}

// sweep lower triangular B into lower triangular A
void sweep_lower_triangular(gsl_matrix* A, gsl_matrix* B)
{
  int size = A->size1;

  if (size != A->size2)
    throw jdimension_error("Matrices must be square (%d vs. %d).", size, A->size2);
  if (size != B->size1)
    throw jdimension_error("Number of rows (%d vs. %d) does not match.", size, B->size1);
  if (size != B->size2)
    throw jdimension_error("Number of columns (%d vs. %d) does not match.", size, B->size2);

  for (int colX = 0; colX < size; colX++) {
    for (int rowX = colX; rowX < size; rowX++) {
      //double v1 = gsl_matrix_get(A, rowX, colX);
      double v1 = gsl_matrix_get(A, rowX, rowX);
      double v2 = gsl_matrix_get(B, rowX, colX);

      // calculate Givens rotation from relevant components of 'A' and 'B'
      double c, s;
      double v1p;
      try {
	v1p = calcGivensRotation(v1, v2, c, s);
      } catch (jarithmetic_error) {
	continue;
      }

      //gsl_matrix_set(A, rowX, colX, v1p);
      gsl_matrix_set(A, rowX, rowX, v1p);
      gsl_matrix_set(B, rowX, colX, 0.0);

      // apply Givens rotation to remaining rows of 'A' and 'B'
      for (int j = rowX + 1; j < size; j++) {
	//v1 = gsl_matrix_get(A, j, colX);
	v1 = gsl_matrix_get(A, j, rowX);
	v2 = gsl_matrix_get(B, j, colX);

	double v2p;
	applyGivensRotation(v1, v2, c, s, v1p, v2p);

	//gsl_matrix_set(A, j, colX, v1p);
	gsl_matrix_set(A, j, rowX, v1p);
	gsl_matrix_set(B, j, colX, v2p);
      }
    }
  }
}

// --------------------------------------------------------------------------

// Step 1: rotate rows of 'A12' into last column
void propagate_covar_square_root_step1(gsl_matrix_complex* A12,
				   gsl_matrix_complex* A22)
{
  int sizeDm = A12->size1;
  int sizePm = A22->size1;

  // iterate over columns of 'A12'
  for (int i = sizePm-2; i >= 0; i--) {

    // calculate Givens rotation from first row of 'A12' and apply
    gsl_complex v1 = gsl_matrix_complex_get(A12, 0, sizePm-1);
    gsl_complex v2 = gsl_matrix_complex_get(A12, 0, i);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(A12, 0, sizePm-1, v1p);
    gsl_matrix_complex_set(A12, 0, i, ComplexZero);

    // apply rotation to remaining rows of 'A12'
    for (int j = 1; j < sizeDm; j++) {
      v1 = gsl_matrix_complex_get(A12, j, sizePm-1);
      v2 = gsl_matrix_complex_get(A12, j, i);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      // 'v2p' should be zero
      gsl_matrix_complex_set(A12, j, sizePm-1, v1p);
      gsl_matrix_complex_set(A12, j, i, ComplexZero);
    }

    // apply rotation to affected rows of 'A22'
    for (int j = i; j < sizePm; j++) {
      v1 = gsl_matrix_complex_get(A22, j, sizePm-1);
      v2 = gsl_matrix_complex_get(A22, j, i);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(A22, j, sizePm-1, v1p);
      gsl_matrix_complex_set(A22, j, i, v2p);
    }
  }
}

// Step 2: rotate last column into leading diagonal
//     2a: iterate over rows of 'A12'
void propagate_covar_square_root_step2a(gsl_matrix_complex* A11,
				    gsl_matrix_complex* A12,
				    gsl_matrix_complex* A21,
				    gsl_matrix_complex* A22)
{
  int sizeDm = A11->size1;
  int sizePm = A22->size1;
  
  for (int i = 0; i < sizeDm; i++) {
    gsl_complex v1 = gsl_matrix_complex_get(A11, i, i);
    gsl_complex v2 = gsl_matrix_complex_get(A12, i, sizePm-1);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(A11, i, i, v1p);
    gsl_matrix_complex_set(A12, i, sizePm-1, ComplexZero);

    // apply rotation to remaining rows of 'A11'
    for (int j = i+1; j < sizeDm; j++) {
      v1 = gsl_matrix_complex_get(A11, j, i);
      v2 = gsl_matrix_complex_get(A12, j, sizePm-1);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(A11, j, i, v1p);
      gsl_matrix_complex_set(A12, j, sizePm-1, v2p);
    }

    // apply rotation to all rows of 'A21'
    for (int j = 0; j < sizePm; j++) {
      v1 = ComplexZero;
      v2 = gsl_matrix_complex_get(A22, j, sizePm-1);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(A21, j, i, v1p);
      gsl_matrix_complex_set(A22, j, sizePm-1, v2p);
    }
  }
}

// Step 2: rotate last column into leading diagonal
//     2b: iterate over rows of 'A22'
void propagate_covar_square_root_step2b(gsl_matrix_complex* A22)
{
  int sizePm = A22->size1;

  /*
  printf("Before:\n");
  for (int n = 0; n < sizePm; n++) {
    for (int m = 0; m < sizePm; m++)
      printf("%8.4e  ",gsl_complex_abs(gsl_matrix_complex_get(A22, n, m)));
    printf("\n");
  }
  */

  for (int i = 0; i < sizePm-1; i++) {
    gsl_complex v1 = gsl_matrix_complex_get(A22, i, i);
    gsl_complex v2 = gsl_matrix_complex_get(A22, i, sizePm-1);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(A22, i, i, v1p);
    gsl_matrix_complex_set(A22, i, sizePm-1, ComplexZero);

    // apply rotation to remaining rows of 'A22'
    for (int j = i+1; j < sizePm; j++) {
      v1 = gsl_matrix_complex_get(A22, j, i);
      v2 = gsl_matrix_complex_get(A22, j, sizePm-1);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(A22, j, i, v1p);
      gsl_matrix_complex_set(A22, j, sizePm-1, v2p);
    }
  }

  /*
  printf("After:\n");
  for (int n = 0; n < sizePm; n++) {
    for (int m = 0; m < sizePm; m++) {
      if (isnan(gsl_complex_abs(gsl_matrix_complex_get(A22, n, m))))
	  printf("Problem here.");
      printf("%8.4e  ",gsl_complex_abs(gsl_matrix_complex_get(A22, n, m)));
    }
    printf("\n");
  }
  printf("\n");
  */
}

void propagate_covar_square_root(gsl_matrix_complex* A11,
                                 gsl_matrix_complex* A12,
                                 gsl_matrix_complex* A21,
                                 gsl_matrix_complex* A22)
{
  // dimensionality check
  if (A11->size1 != A11->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Dm' do not match.",
			   A11->size1, A11->size2);
  if (A22->size1 != A22->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Pm' do not match.",
			   A22->size1, A22->size2);
  if (A11->size1 != A12->size1)
    throw jdimension_error("Rows of 'Dm' (%d) and 'A12' (%d) do not match.",
			   A11->size1, A12->size1);
  if (A22->size2 != A12->size2)
    throw
      jdimension_error("Columns of 'Pm' (%d) and 'A12' (%d) do not match.",
		       A22->size2, A12->size2);

  // update 'A22' and calculate 'A21'
  propagate_covar_square_root_step1(A12, A22);
  propagate_covar_square_root_step2a(A11, A12, A21, A22);
  propagate_covar_square_root_step2b(A22);
}

// --------------------------------------------------------------------------

// Step 1: rotate rows of 'A12' into last column
static void propagate_info_square_root_step1_(gsl_matrix_complex* A12,
                                              gsl_vector_complex* a_22)
{
  int nrows = A12->size1;
  int ncols = A12->size2;

  // iterate over columns of 'A12'
  for (int i = ncols-2; i >= 0; i--) {

    // calculate Givens rotation from first row of 'A12' and apply
    gsl_complex v1 = gsl_matrix_complex_get(A12, 0, ncols-1);
    gsl_complex v2 = gsl_matrix_complex_get(A12, 0, i);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(A12, 0, ncols-1, v1p);
    gsl_matrix_complex_set(A12, 0, i, ComplexZero);

    // apply rotation to remaining rows of 'A12'
    for (int j = 1; j < nrows; j++) {
      v1 = gsl_matrix_complex_get(A12, j, ncols-1);
      v2 = gsl_matrix_complex_get(A12, j, i);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      // 'v2p' should be zero
      gsl_matrix_complex_set(A12, j, ncols-1, v1p);
      gsl_matrix_complex_set(A12, j, i, ComplexZero);
    }

    // apply rotation to 'a_22'
    v1 = gsl_vector_complex_get(a_22, ncols-1);
    v2 = gsl_vector_complex_get(a_22, i);

    gsl_complex v2p;
    applyGivensRotation(v1, v2, c, s, v1p, v2p);

    gsl_vector_complex_set(a_22, ncols-1, v1p);
    gsl_vector_complex_set(a_22, i, v2p);
  }
}

// Step 2: rotate last column into leading diagonal
static void propagate_info_square_root_step2_(gsl_matrix_complex* sqrt_Pm_inv,
                                              gsl_matrix_complex* A12,
                                              gsl_vector_complex* a_21,
                                              gsl_vector_complex* a_22)
{
  int nrows = A12->size1;
  int ncols = A12->size2;

  for (int i = 0; i < nrows; i++) {
    // calculate rotation between 'sqrt_Pm_inv' and 'A12'
    gsl_complex v1 = gsl_matrix_complex_get(sqrt_Pm_inv, i, i);
    gsl_complex v2 = gsl_matrix_complex_get(A12, i, ncols-1);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(sqrt_Pm_inv, i, i, v1p);
    gsl_matrix_complex_set(A12, i, ncols-1, ComplexZero);

    // apply rotation to remaining rows of 'sqrt_Pm_inv'
    for (int j = i+1; j < nrows; j++) {
      v1 = gsl_matrix_complex_get(sqrt_Pm_inv, j, i);
      v2 = gsl_matrix_complex_get(A12, j, ncols-1);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(sqrt_Pm_inv, j, i, v1p);
      gsl_matrix_complex_set(A12, j, ncols-1, v2p);
    }

    // apply rotation to 'a_21' and 'a_22'
    v1 = gsl_vector_complex_get(a_21, i);
    v2 = gsl_vector_complex_get(a_22, ncols-1);

    gsl_complex v2p;
    applyGivensRotation(v1, v2, c, s, v1p, v2p);

    gsl_vector_complex_set(a_21, i, v1p);
    gsl_vector_complex_set(a_22, ncols-1, v2p);
  }
}

static void propagate_info_square_root_in_one_step_(gsl_matrix_complex* sqrt_Pm_inv,
                                                    gsl_matrix_complex* A12,
                                                    gsl_vector_complex* a_21,
                                                    gsl_vector_complex* a_22)
{
  int nrows = A12->size1;
  int ncols = A12->size2;

  for (int i = 0; i < nrows; i++) {
    for (int k = 0; k < ncols; k++) {

      // calculate rotation between 'sqrt_Pm_inv' and 'A12'
      gsl_complex v1 = gsl_matrix_complex_get(sqrt_Pm_inv, i, i);
      gsl_complex v2 = gsl_matrix_complex_get(A12, i, k);

      gsl_complex c, s;
      gsl_complex v1p;
      try {
	v1p = calcGivensRotation(v1, v2, c, s);
      } catch (jarithmetic_error) {
	continue;
      }

      gsl_matrix_complex_set(sqrt_Pm_inv, i, i, v1p);
      gsl_matrix_complex_set(A12, i, k, ComplexZero);

      // apply rotation to remaining rows of 'sqrt_Pm_inv'
      for (int j = i+1; j < nrows; j++) {
	v1 = gsl_matrix_complex_get(sqrt_Pm_inv, j, i);
	v2 = gsl_matrix_complex_get(A12, j, k);

	gsl_complex v2p;
	applyGivensRotation(v1, v2, c, s, v1p, v2p);

	gsl_matrix_complex_set(sqrt_Pm_inv, j, i, v1p);
	gsl_matrix_complex_set(A12, j, k, v2p);
      }

      // apply rotation to 'a_21' and 'a_22'
      v1 = gsl_vector_complex_get(a_21, i);
      v2 = gsl_vector_complex_get(a_22, k);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_vector_complex_set(a_21, i, v1p);
      gsl_vector_complex_set(a_22, k, v2p);
    }
  }
}

void propagate_info_square_root(gsl_matrix_complex* sqrt_Pm_inv,
                                gsl_matrix_complex* A12,
                                gsl_vector_complex* a_21,
                                gsl_vector_complex* a_22, bool rankOneA12)
{
  // dimensionality check
  if (sqrt_Pm_inv->size1 != sqrt_Pm_inv->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Pm' do not match.",
			   sqrt_Pm_inv->size1, sqrt_Pm_inv->size2);
  if (sqrt_Pm_inv->size1 != A12->size1)
    throw jdimension_error("Rows of 'Pm' (%d) and 'A12' (%d) do not match.",
			   sqrt_Pm_inv->size1, A12->size1);
  if (sqrt_Pm_inv->size2 != a_21->size)
    throw jdimension_error("Cols of 'Pm' (%d) and 'a_21' (%d) do not match.",
			   sqrt_Pm_inv->size2, a_21->size);
  if (A12->size2 != a_22->size)
    throw jdimension_error("Cols of 'A12' (%d) and 'a_22' (%d) do not match.",
			   A12->size2, a_22->size);

  // perform update
  if (rankOneA12) {
    propagate_info_square_root_step1_(A12, a_22);
    propagate_info_square_root_step2_(sqrt_Pm_inv, A12, a_21, a_22);
  } else {
    propagate_info_square_root_in_one_step_(sqrt_Pm_inv, A12, a_21, a_22);
  }
}

// --------------------------------------------------------------------------

// Step 2: rotate last column into leading diagonal
void propagate_info_square_root_step2_rls(gsl_matrix_complex* sqrt_Pm_inv,
				      gsl_vector_complex* a_12,
				      gsl_vector_complex* a_21,
				      gsl_complex a_22)
{
  int nrows = a_12->size;

  for (int i = 0; i < nrows; i++) {
    // calculate rotation between 'sqrt_Pm_inv' and 'a_12'
    gsl_complex v1 = gsl_matrix_complex_get(sqrt_Pm_inv, i, i);
    gsl_complex v2 = gsl_vector_complex_get(a_12, i);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(sqrt_Pm_inv, i, i, v1p);
    gsl_vector_complex_set(a_12, i, ComplexZero);

    // apply rotation to remaining rows of 'sqrt_Pm_inv'
    for (int j = i+1; j < nrows; j++) {
      v1 = gsl_matrix_complex_get(sqrt_Pm_inv, j, i);
      v2 = gsl_vector_complex_get(a_12, j);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(sqrt_Pm_inv, j, i, v1p);
      gsl_vector_complex_set(a_12, j, v2p);
    }

    // apply rotation to 'a_21' and 'a_22'
    v1 = gsl_vector_complex_get(a_21, i);
    v2 = a_22;

    gsl_complex v2p;
    applyGivensRotation(v1, v2, c, s, v1p, v2p);

    gsl_vector_complex_set(a_21, i, v1p);
    a_22 = v2p;
  }
}

void propagate_info_square_root_rls(gsl_matrix_complex* sqrt_Pm_inv,
                                    gsl_vector_complex* a_12,
                                    gsl_vector_complex* a_21,
                                    gsl_complex a_22)
{
  // dimensionality check
  if (sqrt_Pm_inv->size1 != sqrt_Pm_inv->size2)
    throw jdimension_error("Dimensions (%d, %d) of 'Pm' do not match.",
			   sqrt_Pm_inv->size1, sqrt_Pm_inv->size2);
  if (sqrt_Pm_inv->size1 != a_12->size)
    throw jdimension_error("Rows of 'Pm' (%d) and 'a_12' (%d) do not match.",
			   sqrt_Pm_inv->size1, a_12->size);
  if (sqrt_Pm_inv->size2 != a_21->size)
    throw jdimension_error("Cols of 'Pm' (%d) and 'a_21' (%d) do not match.",
			   sqrt_Pm_inv->size2, a_21->size);

  // perform update
  propagate_info_square_root_step2_rls(sqrt_Pm_inv, a_12, a_21, a_22);
}

// --------------------------------------------------------------------------

// add loading 'wght^2' to diagonal component 'sqrt_Pm[dim][dim]'
void add_diagonal_loading(gsl_matrix_complex* sqrt_Pm_inv, int dim, double wght)
{
  int size = sqrt_Pm_inv->size1;

  static gsl_vector_complex* scratch = NULL;
  if (scratch == NULL)
    scratch = gsl_vector_complex_calloc(size);
  gsl_vector_complex_set_zero(scratch);

  if (dim < 0 || dim >= size)
    throw jdimension_error("Dimensions (%d) to be loaded is out of range.",
			   dim);

  gsl_vector_complex_set(scratch, dim, gsl_complex_rect(wght, 0.0));

  // annihilate the nonzero components of 'scratch'
  for (int i = dim; i < size; i++) {
    gsl_complex v1 = gsl_matrix_complex_get(sqrt_Pm_inv, i, i);
    gsl_complex v2 = gsl_vector_complex_get(scratch, i);

    gsl_complex c, s;
    gsl_complex v1p;
    try {
      v1p = calcGivensRotation(v1, v2, c, s);
    } catch (jarithmetic_error) {
      continue;
    }

    gsl_matrix_complex_set(sqrt_Pm_inv, i, i, v1p);
    gsl_vector_complex_set(scratch, i, ComplexZero);

    // apply rotation to remaining rows of 'sqrt_Pm_inv'
    for (int j = i+1; j < size; j++) {
      v1 = gsl_matrix_complex_get(sqrt_Pm_inv, j, i);
      v2 = gsl_vector_complex_get(scratch, j);

      gsl_complex v2p;
      applyGivensRotation(v1, v2, c, s, v1p, v2p);

      gsl_matrix_complex_set(sqrt_Pm_inv, j, i, v1p);
      gsl_vector_complex_set(scratch, j, v2p);
    }
  }
}

void make_conjugate_symmetric(gsl_matrix_complex* mat)
{
  int nrows = mat->size1;

  for (int n = 1; n < nrows; n++)
    for (int m = 0; m < n; m++)
      gsl_matrix_complex_set(mat, m, n, gsl_complex_conjugate(gsl_matrix_complex_get(mat, n, m)));
}

/*
 * Calculate the diagonal of a cholesky decomposition.
 * If the output vector given is NULL then a vector of the needed size is allocated, that has to be freed than in main program
 */

gsl_vector* cholesky_diagonal(gsl_vector* v, const gsl_matrix* m)
{
  double d;

  if (m->size1 != m->size2)
    throw jdimension_error("Dimensions (%d, %d) of matrix do not match.", m->size1, m->size2);
  if (v == NULL)
    v = gsl_vector_calloc(m->size1);
  if (m->size1 != v->size)
    throw jdimension_error("Dimension of vector (%d) does not match with dimension of matrix (%d).", v->size, m->size1);

  for (unsigned i=0; i<v->size; i++) {
    d = 0;
    for (unsigned j=0; j<=i; j++) {
      d += gsl_matrix_get(m, i, j)*gsl_matrix_get(m, i, j);
    }
    gsl_vector_set(v, i, d);
  }
  return v;
}

/*
 * Calculate the diagonal of m*m^T.
 * If the output vector given is NULL then a vector of the needed size is allocated, that has to be freed than in main program
 */

gsl_vector* square_diagonal(gsl_vector* v, const gsl_matrix* m)
{
  double d;

  if (v == NULL)
    v = gsl_vector_calloc(m->size1);
  if (m->size1 != v->size)
    throw jdimension_error("Dimension of vector (%d) does not match with dimension of matrix (%d).", v->size, m->size1);

  for (unsigned i=0; i<m->size1; i++) {
    d = 0;
    for (unsigned j=0; j<m->size2; j++) {
      d += gsl_matrix_get(m, i, j)*gsl_matrix_get(m, i, j);
    }
    gsl_vector_set(v, i, d);
  }
  return v;
}

// -----------------------------------------------------------------
//
//  Implementation Notes:
//
//  1. These routines are useful for implementing a Kalman filter
//     which propagates either the square-root of the predicted
//     state error correlation matrix P_m(t), or the square-root
//     of the information matrix P^-1_m(t). Details can be found
//     in McDonough et al [1].
//
//  2. The propagation of P_m^1/2(t) is more numerically robust
//     than the naive propagation of P_m(t) based on the Riccati
//     equation, as explained in Haykin [2], and Sayed and
//     Kailath [3].
//
//  3. The routine 'rank_one_update_Cholesky_factor' is based on a
//     simple algorithm taken from [4]. It calculates the Cholesky
//     decomposition of
//
//             D_m(t) = D(t) - alpha_m c_m(t) c_m^H(t)
//
//     given D^1/2(t), the vector c_m, and the scalar alpha_m.
//     An explanation of this algorithm is also provided in [1].
//
// References:
//
// [1] McDonough, J., Raub, D., Woelfel, M., and Waibel, A.,
//     "Towards adaptive hidden Markov model beamforming," Tech.
//     Report 101, Interactive Systems Labs, University of
//     Karlsruhe, 2003.

// [2] Haykin, S., Adaptive Filter Theory, fourth edition, Prentice
//     Hall, New York, 2002.
//
// [3] Sayed, A.-H. and Kailath, T., "A State-Space Approach to
//     Adaptive RLS Filtering," IEEE Signal Processing Magazine,
//     pp. 18-60, July, 1994.
// 
// [4] Gill, P. E., Golub, G. H., Murray W., and Saunders, M. A.,
//     "Methods for Modifying Matrix Factorizations," Mathematics
//     of Computation, vol. 28, no. 126, pp. 505-535, April. 1974.
//
// -----------------------------------------------------------------
