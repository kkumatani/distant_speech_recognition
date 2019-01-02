/**
 * @file ica.h
 * @brief PCA and ICA
 * @author Kenichi Kumatani and John McDonough
 */

#ifndef ICA_H
#define ICA_H

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "common/refcount.h"
#include "common/jexception.h"


// ----- definition for class `PCA' -----
//
class PCA {
 public:
  PCA(unsigned dimN);
  ~PCA();

  void pca_svd(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);
  void pca_eigen(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);

 private:
  const unsigned	_dimN;
  gsl_vector*		_work;
};

typedef refcount_ptr<PCA> PCAPtr;


// ----- definition for class `FastICA' -----
//
class FastICA {
public:
  FastICA(unsigned dimN, unsigned maxIterN);
  ~FastICA();

  void deflation(gsl_matrix* data, gsl_matrix* B, gsl_matrix* A, gsl_matrix* W, gsl_matrix* M,
		 gsl_matrix* neg, double eps, int maxIterN);

 private:
  const unsigned	_dimN;
  const unsigned	_maxIterN;

  gsl_matrix*		_w;
  gsl_matrix*		_a;
  gsl_matrix*		_wr;
  gsl_matrix*		_BTw;
  gsl_matrix*		_BBTw;
  gsl_matrix*		_wOld;
  gsl_matrix*		_wOld2;
  gsl_matrix*		_wSum;
  gsl_matrix*		_wDiff;
  gsl_matrix*		_hypTan;
  gsl_matrix*		_dHypTan;
  gsl_matrix*		_dHypTanT;
  gsl_matrix*		_wDHypTanT;
  gsl_matrix*		_XHypTan;
};

typedef refcount_ptr<FastICA> FastICAPtr;

#endif
