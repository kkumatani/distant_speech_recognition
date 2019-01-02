/**
 * @file ica.cc
 * @brief PCA and ICA
 * @author Kenichi Kumatani and John McDonough
 */

#include "sad/ica.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"


// ----- methods for class `PCA' -----
//
PCA::PCA(unsigned dimN)
  : _dimN(dimN), _work(gsl_vector_alloc(_dimN))
{
}

PCA::~PCA()
{
  gsl_vector_free(_work);
}

void PCA::pca_svd(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten)
{
  if (basis->size1 != _dimN || basis->size2 != _dimN)
    throw jdimension_error("Matrix 'basis' (%d x %d) should be (%d x %d)\n", basis->size1, basis->size2, _dimN, _dimN);
  if (whiten->size != _dimN)
    throw jdimension_error("Vector 'whiten' (length %d) should be length %d)\n", whiten->size, _dimN);
  if (eigenVal->size != _dimN)
    throw jdimension_error("Vector 'eigenVal' (length %d) should be length %d\n", eigenVal->size, _dimN);

  gsl_linalg_SV_decomp(input, basis, eigenVal, _work);
  for (unsigned dimX = 0; dimX < _dimN; dimX++)
    gsl_vector_set(whiten, dimX, sqrt(1.0 / gsl_vector_get(eigenVal, dimX)));
}

void PCA::pca_eigen(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten)
{
  if (basis->size1 != _dimN || basis->size2 != _dimN)
    throw jdimension_error("Matrix 'basis' (%d x %d) should be (%d x %d)\n", basis->size1, basis->size2, _dimN, _dimN);
  if (whiten->size != _dimN)
    throw jdimension_error("Vector 'whiten' (length %d) should be length %d)\n", whiten->size, _dimN);
  if (eigenVal->size != _dimN)
    throw jdimension_error("Vector 'eigenVal' (length %d) should be length %d\n", eigenVal->size, _dimN);

  throw jconsistency_error("Not finished yet.");
}


// ----- methods for class `FastICA' -----
//
FastICA::FastICA(unsigned dimN, unsigned maxIterN)
  : _dimN(dimN), _maxIterN(maxIterN),
    _w(gsl_matrix_alloc(1, dimN)),
    _a(gsl_matrix_alloc(1, dimN)),
    _wr(gsl_matrix_alloc(1, dimN)),
    _BTw(gsl_matrix_alloc(1, dimN)),
    _BBTw(gsl_matrix_alloc(1, dimN)),
    _wOld(gsl_matrix_alloc(1, dimN)),
    _wOld2(gsl_matrix_alloc(1, dimN)),
    _wSum(gsl_matrix_alloc(1, dimN)),
    _wDiff(gsl_matrix_alloc(1, dimN)),
    _hypTan(gsl_matrix_alloc(1, dimN)),
    _dHypTan(gsl_matrix_alloc(1, dimN)),
    _dHypTanT(gsl_matrix_alloc(1, dimN)),
    _wDHypTanT(gsl_matrix_alloc(1, dimN)),
    _XHypTan(gsl_matrix_alloc(1, dimN))
{
}

FastICA::~FastICA()
{
  gsl_matrix_free(_w);
  gsl_matrix_free(_a);
  gsl_matrix_free(_wr);
  gsl_matrix_free(_BTw);
  gsl_matrix_free(_BBTw);
  gsl_matrix_free(_wOld);
  gsl_matrix_free(_wOld2);
  gsl_matrix_free(_wSum);
  gsl_matrix_free(_wDiff);
  gsl_matrix_free(_hypTan);
  gsl_matrix_free(_dHypTan);
  gsl_matrix_free(_dHypTanT);
  gsl_matrix_free(_wDHypTanT);
  gsl_matrix_free(_XHypTan);
}

void FastICA::deflation(gsl_matrix* data, gsl_matrix* B, gsl_matrix* A, gsl_matrix* W, gsl_matrix* M,
			gsl_matrix* neg, double eps, int maxIterN)
{
  if (B->size1 != _dimN || B->size2 != _dimN)
    throw jdimension_error("Matrix 'B' (%d x %d) should be (%d x %d)\n", B->size1, B->size2, _dimN, _dimN);
  if (A->size1 != _dimN || B->size2 != _dimN)
    throw jdimension_error("Matrix 'A' (%d x %d) should be (%d x %d)\n", A->size1, A->size2, _dimN, _dimN);
  if (W->size1 != _dimN || B->size2 != _dimN)
    throw jdimension_error("Matrix 'W' (%d x %d) should be (%d x %d)\n", W->size1, W->size2, _dimN, _dimN);
  if (neg->size1 != 1 || B->size2 != _dimN)
    throw jdimension_error("Matrix 'neg' (%d x %d) should be (%d x %d)\n", neg->size1, neg->size2, 1, _dimN);

  gsl_matrix_set_zero(B);
  for (unsigned dimX = 0; dimX < _dimN; dimX++) {
    gsl_matrix_set_zero(_wOld);
    gsl_matrix_set_zero(_wOld2);
    // dmatrixInitRandom(w, 1.0);

    for (unsigned i = 0; i < _dimN; i++)
      gsl_matrix_set(_w, 0, i, fabs(gsl_matrix_get(_w, 0, i) - 0.5));

    static const double alpha = 1.0;
    static const double beta  = 0.0;

    // project w onto space orthogonal to other basis vectors
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,   alpha, _w,   B, beta, _BTw);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, alpha, _BTw, B, beta, _BBTw);
    gsl_matrix_sub(_w, _BBTw);

    // normalize
    double norm = 0.0;
    for (unsigned i = 0; i < _dimN; i++) {
      double val = gsl_matrix_get(_w, 0, i);
      norm += val * val;
    }
    norm = sqrt(norm);
    gsl_matrix_scale(_w, 1.0 / norm);

    for (unsigned iterX = 0; iterX <= _maxIterN; iterX++) {

      // project w onto space orthogonal to other basis vectors
      gsl_blas_dgemm(CblasNoTrans, CblasTrans,   alpha, _w,   B, beta, _BTw);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, alpha, _BTw, B, beta, _BBTw);
      gsl_matrix_sub(_w, _BBTw);

      // normalize
      double norm = 0.0;
      for (unsigned i = 0; i < _dimN; i++) {
	double val = gsl_matrix_get(_w, 0, i);
	norm += val * val;
      }
      norm = sqrt(norm);
      gsl_matrix_scale(_w, 1.0 / norm);

      // test for termination condition
      double norm_s = 0.0;
      double norm_d = 0.0;
      for (unsigned i = 0; i < _dimN; i++) {
	gsl_matrix_set(_wSum, 0, i, gsl_matrix_get(_w, 0, i) + gsl_matrix_get(_wOld, 0, i));
	gsl_matrix_set(_wDiff, 0, i, gsl_matrix_get(_w, 0, i) - gsl_matrix_get(_wOld, 0, i));

	double sum  = gsl_matrix_get(_wSum, 0, i);
	norm_s += sum * sum;

	double diff = gsl_matrix_get(_wDiff, 0, i);
	norm_d += diff * diff;
      }
      norm_s = sqrt(norm_s);
      norm_d = sqrt(norm_d);

      if (norm_s < eps || norm_d < eps) {

	fprintf(stderr, "Converged [%d steps]\n", iterX);

	// copy the vector into the basis matrix
	for (unsigned i = 0; i < _dimN; i++)
	  gsl_matrix_set(B, dimX, i, gsl_matrix_get(_w, 0, i));

	iterX = maxIterN;

      } else {

	gsl_matrix_memcpy(_wOld2, _wOld);
	gsl_matrix_memcpy(_wOld, _w);

	// apply learning rule
	gsl_blas_dgemm(CblasNoTrans, CblasTrans,   alpha, _w,   data, beta, _hypTan);
	unsigned frameN = data->size2;
	for (unsigned i = 0; i < frameN; i++) {
	  gsl_matrix_set(_hypTan,  0, i, tanh(gsl_matrix_get(_hypTan, 0, i)));
	  double val = gsl_matrix_get(_hypTan, 0, i);
	  gsl_matrix_set(_dHypTan, 0, i, (1.0 - val * val));
	}

	gsl_matrix_transpose_memcpy(_dHypTanT, _dHypTan);
	double sum = 0.0;
	for (unsigned i = 0; i < frameN; i++)
	  sum += gsl_matrix_get(_dHypTanT, i, 0);

	gsl_matrix_set_zero(_wDHypTanT);
	for (unsigned i = 0; i < _dimN; i++)
	  gsl_matrix_set(_wDHypTanT, 0, i, sum * gsl_matrix_get(_w, 0, i));

	gsl_blas_dgemm(CblasTrans, CblasTrans,   alpha, _hypTan, data, beta, _XHypTan);

	norm = 0.0;
	for (unsigned i = 0; i < _dimN; i++) {
	  double val = (gsl_matrix_get(_XHypTan, 0, i) - gsl_matrix_get(_wDHypTanT, 0, i)) / frameN;
	  gsl_matrix_set(_w, 0, i, val);
	  norm += val * val;
	}
	norm = sqrt(norm);
	gsl_matrix_scale(_w, 1.0 / norm);
      }
    }
  }
}
