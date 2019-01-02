/*
  File Name: grad.c
  Last Modification Date:	3/9/94	10:03:45
  Current Version: grad.c	1.7
  %File Creation Date: 26/10/91
  Author: Ramesh Gopinath  <ramesh@dsp.rice.edu>

  Copyright: All software, documentation, and related files in this distribution
  are Copyright (c) 1993  Rice University

  Permission is granted for use and non-profit distribution providing that this
  notice be clearly maintained. The right to distribute any portion for profit
  or as part of any commercial product is specifically reserved for the author.
*/

/*
#################################################################################
grad.mex4: 
function [g,f,h] = grad(k,M,N,fs); Computes the gradient of stopband energy of the
                   linear-phase prototype filter in a cosine-modulated filter bank
		   with respect to lattice parameters of the J lattices.

      Input:
            k : Vector of denormalized lattice parameters for the J lattices of
	        CMFB
            M : M-channel, M-band etc
            N : 2Mm, length of the prototype filter
           fs : Stopband edge (as a fraction of pi)
      Output: 
            f : Stopband Energy
            h : Prototype filter (second half)
#################################################################################
*/

#include "pc_lattice.h"
#include "modulated/prototype_design.h"
#include <math.h>

#define TRUE 1
#define PI 3.14159265358979
#define max(A,B) (A > B ? A : B)
#define SQRT_HALF 0.70710678118655

// ----- methods for class 'CosineModulatedPrototypeDesign' -----
//
CosineModulatedPrototypeDesign::
CosineModulatedPrototypeDesign(int M, int N, double fs)
  : _M(M), _N(N), _M2(2*_M), _m(_N/_M2), _Mm(_m*_M), _J(int(_M/2)), _Jm(_m*_J),
    _oddM(int(fmod(double(_M),2.0))), _oddm(int(fmod(double(_m),2.0)))
{
  // memory allocation
  _sinews = new double[_N];
  _h      = new double[_Mm];
  _hh     = new double[_Mm];
  _Ph     = new double[_Mm];
  _jac    = new double[2*_Jm*_m];
  _index  = new int[_Mm];

  _proto  = gsl_vector_calloc(_N / 2);

  // Sinews Computation
  double ws = PI*fs; 
  _sinews[0] = PI-ws;
  for (int i=1;i<_N;i++) {
    double iws = i*ws;
    _sinews[i] = -sin(iws)/i;
  }

  // Index Computation
  int m_div_2 = (int)(_m/2);
  for (int i=0,l=0;i<_J;i++) {
    if (_oddm != TRUE) {
      for (int j=0,M2j=0; j<m_div_2;j++,l++,M2j+=_M2) {
	_index[l] = _Mm-1-M2j-i;
	_index[m_div_2+l] = M2j+i;
	_index[_m+l] = _Mm-1-_M-M2j-i;
	_index[_m+m_div_2+l] = _M+M2j+i;
      }
    } else {
      for (int j=0,M2j=0; j<m_div_2;j++,l++,M2j+=_M2) {
	_index[l] = _Mm-1-M2j-i;
	_index[_m+m_div_2+1+l] = _M2+M2j+i;
	_index[_m+l] = _Mm-1-_M-M2j-i;
	_index[m_div_2+1+l] = _M+M2j+i;
      }
      _index[_m+l] = i;
      _index[l] = _M-1-i;
      l++;
    }
    l += _m+m_div_2;
  }
}

CosineModulatedPrototypeDesign::~CosineModulatedPrototypeDesign()
{
  delete[] _sinews;
  delete[] _h;
  delete[] _hh;
  delete[] _Ph;
  delete[] _jac;
  delete[] _index;
}

const gsl_vector* CosineModulatedPrototypeDesign::proto()
{
  for (int i = 0; i < _Mm; i++)
    gsl_vector_set(_proto, i, _h[i]);

  return _proto;
}

void CosineModulatedPrototypeDesign::grad(const double* x, double* g)
{
  register int i,j, k;
  const double *xk;
  double *jac0i, *jac1i, tmp;
  int *l0i, *l1i, k_ord, m2, im2;

  k_ord = _m-1;
  m2 = 2*_m;
  for (i=0,xk=x,jac0i=_jac,jac1i=_jac+_m,im2=0;i<_J;i++,xk+=_m,im2+=m2) {
    for (j=0,l0i=_index+im2,l1i=l0i+_m;
	 j<= k_ord; j++,g++,l0i=_index+im2,l1i=l0i+_m,jac0i+=_m,jac1i+=_m) {
      Dpclat(jac0i,jac1i,xk,k_ord,j);
      tmp = 0;
       for (k=0; k<_m; k++) {
	tmp += *(_Ph + (*l0i++)) * *(jac0i++);
	tmp += *(_Ph + (*l1i++)) * *(jac1i++);
      }
      *g = tmp;
    }
  }
}

void CosineModulatedPrototypeDesign::fcn(const double* x, double* f)
{
  register int i,j;
  int *l0i, *l1i, k_ord;
  const double *xk;
  double *h0, *h1, *ph, tmp;

  k_ord = _m-1;

  for (i=0,l0i=_index,l1i=l0i+_m,xk=x,h0=_hh,h1=_hh+_m; i<_J ;
       i++,l0i+=_m,l1i+=_m,xk+=_m,h0+=_m,h1+=_m) {
    Pclat(h0,h1,xk,k_ord);
    for (j=0; j<_m; j++) {
      *(_h+(*l0i++)) = *(h0++);
      *(_h+(*l1i++)) = *(h1++);
    }
  }
  if (_oddM == TRUE)
    _h[(_M-1)/2] = SQRT_HALF;

  for (i=0, ph=_Ph,tmp=0; i<_Mm;i++,tmp=0,ph++) {
    for (j=0; j<=i; j++)  tmp += _h[j]*(_sinews[i-j]+_sinews[i+j+1]);
    for (j=i+1;j<_Mm;j++) tmp += _h[j]*(_sinews[j-i]+_sinews[i+j+1]);
    *ph = tmp;
  }
  *f = 0;
  for (i=0;i <_Mm; i++) *f += _h[i]*_Ph[i];
  *f = *f/2;
}

// helper functions for 'CosineModulatedPrototypeDesign'
double design_f(const gsl_vector* v, void* params)
{
  CosineModulatedPrototypeDesign* cos = (CosineModulatedPrototypeDesign*) params;

  double f;
  cos->fcn(v->data, &f);

  return f;
}

void design_df(const gsl_vector* v, void* params, gsl_vector* df)
{
  CosineModulatedPrototypeDesign* cos = (CosineModulatedPrototypeDesign*) params;

  cos->grad(v->data, df->data);
}

void design_fdf(const gsl_vector* v, void* params, double* f, gsl_vector* df)
{
  CosineModulatedPrototypeDesign* cos = (CosineModulatedPrototypeDesign*) params;

  cos->fcn(v->data, f);
  cos->grad(v->data, df->data);
}

// ----- methods for class 'PrototypeDesignBase' -----
//
PrototypeDesignBase::PrototypeDesignBase(int M, int m, unsigned r, double wp, int tau)
  : _M(M), _m(m), _wp(M_PI / (wp * _M)), _R(1 << r), _D(_M / _R), _L(_M*_m), _tau(tau),
    _A(gsl_matrix_calloc(_L, _L)),  _b(gsl_vector_calloc(_L)),
    _cpA(gsl_matrix_calloc(_L, _L)),  _cpb(gsl_vector_calloc(_L)),
    _C(gsl_matrix_calloc(_L, _L)), _cpC(gsl_matrix_calloc(_L, _L)),
    _singularVals(gsl_vector_calloc(_L)), _scratch(gsl_vector_calloc(_L)), _workSpace(gsl_vector_calloc(_L)),
    _prototype(gsl_vector_calloc(_L))
{
  _tau = ( tau < 0 ) ? _L / 2 : tau;
}

PrototypeDesignBase::~PrototypeDesignBase()
{
  gsl_matrix_free(_A);    gsl_vector_free(_b);
  gsl_matrix_free(_cpA);  gsl_vector_free(_cpb);

  gsl_vector_free(_singularVals);  gsl_vector_free(_scratch);  gsl_vector_free(_workSpace);
  gsl_vector_free(_prototype);
}

void PrototypeDesignBase::save(const String& fileName)
{
  FILE* fp = btk_fopen(fileName, "w");
  for (unsigned n = 0; n < _L; n++)
    fprintf(fp, "%24.21e\n", gsl_vector_get(_prototype, n));
  btk_fclose(fileName, fp);
}

void PrototypeDesignBase::_calculateAb()
{
  printf("Calculating 'A' and 'b' ... ");  fflush(stdout);

  for (int m = 0; m < _L; m++) {
    for (int n = 0; n < _L; n++) {

      // A(m,n) = sinc(w_p*(n-m));
      if ( ( n - m ) == 0 )
	gsl_matrix_set(_A, m, n, 1.0);
      else
	gsl_matrix_set(_A, m, n, sin(_wp * (n-m)) / (_wp * (n - m)));
    }

    // b(m) = sinc(w_p*(tau_h-m))
    if ( _tau - m == 0 )
      gsl_vector_set(_b, m, 1.0);
    else
      gsl_vector_set(_b, m, sin(_wp * (_tau - m)) / (_wp * (_tau - m)));
  }

  gsl_matrix_memcpy(_cpA, _A);
  gsl_vector_memcpy(_cpb, _b);

  printf("Done\n");  fflush(stdout);
}

void PrototypeDesignBase::_calculateC()
{
  printf("Calculating 'C' ... ");  fflush(stdout);

  for (int m = 0; m < _L; m++) {
    for (int n = 0; n < _L; n++) {
      double Cmn;
      double factor = -1.0;

      if ( ( n - m ) % _D == 0 )
	factor = _D - 1;
      if ( ( n - m ) == 0 )
	Cmn = factor / _D;
      else
	Cmn = factor * sin(M_PI * ( n - m ) / _D) / (M_PI * ( n - m ));

      gsl_matrix_set(_C, m, n, Cmn);
    }
  }

  gsl_matrix_memcpy(_cpC, _C);

  printf("Done\n");  fflush(stdout);
}

void PrototypeDesignBase::_svd(gsl_matrix* U, gsl_matrix* V, gsl_vector* S, gsl_vector* workSpace)
{
  int M = U->size1;
  int N = U->size2;

  // perform singular value decomposition
  if (M < N) {
    gsl_matrix* UT = gsl_matrix_calloc(N, N);
    gsl_matrix_view ut(gsl_matrix_submatrix(UT, /* k1= */ 0, /* k2= */ 0, /* rowN= */ N, /* colN= */ M));
    gsl_matrix_transpose_memcpy(&ut.matrix, U);
    gsl_linalg_SV_decomp(UT, V, S, workSpace);
    gsl_matrix_view v(gsl_matrix_submatrix(V, /* k1= */ 0, /* k2= */ 0, /* rowN= */ M, /* colN= */ N));
    gsl_matrix_set_zero(U);  gsl_matrix_memcpy(U, &v.matrix);
    gsl_matrix_memcpy(V, UT);
    gsl_matrix_free(UT);
  } else {
    gsl_linalg_SV_decomp(U, V, S, workSpace);
    //gsl_linalg_SV_decomp_jacobi(U, V, S);
  }
}

// calculate a basis for the null space of A^T
gsl_matrix* PrototypeDesignBase::_nullSpace(const gsl_matrix* A, double tolerance)
{
  int         M         = A->size2;
  int         N         = A->size1;
  gsl_matrix* U         = gsl_matrix_calloc(M, N);
  gsl_matrix* V         = gsl_matrix_calloc(N, N);
  gsl_vector* S         = gsl_vector_calloc(N);
  gsl_vector* workSpace = gsl_vector_calloc(N);
  gsl_matrix_transpose_memcpy(U, A);

  _svd(U, V, S, workSpace);

  double   maxSingularValue = gsl_vector_get(S, 0);
  unsigned rankNullSpace    = 0;
  for (int dimX = 0; dimX < N; dimX++) {
    if ((gsl_vector_get(S, dimX) / maxSingularValue) < tolerance) {
      /*
      printf("dimX = %d : singV = %g\n", dimX, gsl_vector_get(S, dimX) / maxSingularValue);
      */
      rankNullSpace++;
    }
  }

  printf("||null(A)|| = %u : ||range(A)|| = %lu\n", rankNullSpace, S->size - rankNullSpace);

  gsl_matrix* nullSpace = gsl_matrix_calloc(N, rankNullSpace);
  gsl_matrix_view v(gsl_matrix_submatrix(V, /* k1= */ 0, /* k2= */ N - rankNullSpace, /* rowN= */ N, /* colN= */ rankNullSpace));
  gsl_matrix_memcpy(nullSpace, &v.matrix);

  gsl_matrix_free(U);  gsl_matrix_free(V);  gsl_vector_free(S);  gsl_vector_free(workSpace);

  return nullSpace;
}

// calculate the pseudoinverse solution of A^T x = b
gsl_vector* PrototypeDesignBase::_pseudoInverse(const gsl_matrix* A, const gsl_vector* b, double tolerance)
{
  int         M          = A->size2;
  int         N          = A->size1;
  gsl_matrix* U          = gsl_matrix_calloc(M, N);
  gsl_matrix* V          = gsl_matrix_calloc(N, N);
  gsl_vector* S          = gsl_vector_calloc(N);
  gsl_vector* workSpace  = gsl_vector_calloc(N);
  gsl_matrix_transpose_memcpy(U, A);

  gsl_vector* x = gsl_vector_calloc(N);

  _svd(U, V, S, workSpace);

  // solve for 'x'
  gsl_blas_dgemv(CblasTrans, 1.0, U, b, 0.0, workSpace);
  double maxSingularValue = gsl_vector_get(S, 0);
  for (int n = 0; n < N; n++) {
    if ((gsl_vector_get(S, n) / maxSingularValue) < tolerance)
      gsl_vector_set(workSpace, n, 0.0);
    else
      gsl_vector_set(workSpace, n, gsl_vector_get(workSpace, n) / gsl_vector_get(S, n));
  }
  gsl_blas_dgemv(CblasNoTrans, 1.0, V, workSpace, 0.0, x);

  gsl_matrix_free(U);  gsl_matrix_free(V);  gsl_vector_free(S);  gsl_vector_free(workSpace);

  return x;
}

void PrototypeDesignBase::_solveNonSingular(const gsl_matrix* H, const gsl_vector* c0, const gsl_matrix* P, double tolerance)
{
  // solve for the particular solution
  cout << "Computing pseudoinverse ..." << endl;
  gsl_vector* g_pt = _pseudoInverse(H, c0, tolerance);

  /*
  cout << endl << endl << "g_pt = " << endl;
  _printVector(g_pt);

  cout << endl << endl << "H = " << endl;
  _printMatrix(H);
  */

  // solve for the null-space solution
  cout << "Computing null space ..." << endl;
  gsl_matrix* H_perp = _nullSpace(H, tolerance);

  // test that K_perp is really the null space of K^T
  /*
  gsl_matrix* test = gsl_matrix_calloc(H->size2, H_perp->size2);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, H, H_perp, 0.0, test);
  printf("MaximumElement(H)          = %g\n", gsl_matrix_max(H));
  printf("MaximumElement(H_perp)     = %g\n", gsl_matrix_max(H_perp));
  printf("MaximumElement(H^T H_perp) = %g\n", gsl_matrix_max(test));

  printf("size(H) = (%d x %d)\n", H->size1, H->size2);
  printf("size(H_perp) = (%d x %d)\n", H_perp->size1, H_perp->size1);  fflush(stdout);
  */

  int         M            = H_perp->size1;
  int         N            = H_perp->size2;
  gsl_matrix* P_tilde      = gsl_matrix_calloc(N, N);
  gsl_matrix* scratchSpace = gsl_matrix_calloc(N, M);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, H_perp, P, 0.0, scratchSpace);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, scratchSpace, H_perp, 0.0, P_tilde);
  
  /*
  cout << endl << endl << "P_tilde = " << endl;
  _printMatrix(P_tilde);
  */

  gsl_matrix* U            = gsl_matrix_calloc(N, N);
  gsl_matrix* V            = gsl_matrix_calloc(N, N);
  gsl_vector* S            = gsl_vector_calloc(N);
  gsl_vector* workSpace    = gsl_vector_calloc(N);
  gsl_matrix_memcpy(U, P_tilde);

  // perform singular value decomposition
  cout << "Computing SVD ..." << endl;
  _svd(U, V, S, workSpace);

  // Calculate g_perp = - H_perp * (V * S * U') * H_perp' * P * g_pt;
  gsl_vector* g_perp         = gsl_vector_calloc(M);
  gsl_vector* scratchVector  = gsl_vector_calloc(M);
  gsl_vector* scratchVector2 = gsl_vector_calloc(N);
  gsl_vector* scratchVector3 = gsl_vector_calloc(N);
  gsl_blas_dgemv(CblasNoTrans, 1.0, P, g_pt, 0.0, scratchVector);
  gsl_blas_dgemv(CblasTrans, 1.0, H_perp, scratchVector, 0.0, scratchVector2);
  gsl_blas_dgemv(CblasTrans, 1.0, U, scratchVector2, 0.0, scratchVector3);

  double maxSingularValue = gsl_vector_get(S, 0);
  for (int n = 0; n < N; n++) {
    if ((gsl_vector_get(S, n) / maxSingularValue) > tolerance)
      gsl_vector_set(scratchVector3, n, gsl_vector_get(scratchVector3, n) / gsl_vector_get(S, n));
    else
      gsl_vector_set(scratchVector3, n, 0.0);
  }
  gsl_blas_dgemv(CblasNoTrans, 1.0, V, scratchVector3, 0.0, scratchVector2);
  gsl_blas_dgemv(CblasNoTrans, 1.0, H_perp, scratchVector2, 0.0, g_perp);
  gsl_vector_scale(g_perp, -1.0);

  /*
  cout << endl << endl << "g_perp = " << endl;
  _printVector(g_perp);
  */

  // g = g_pt + g_perp
  gsl_vector_memcpy(_prototype, g_pt); gsl_vector_add(_prototype, g_perp);

  /*
  cout << endl << endl << "g = " << endl;
  _printVector(_prototype);
  */

  gsl_vector_free(g_pt);  gsl_vector_free(g_perp);
  gsl_matrix_free(H_perp); gsl_matrix_free(P_tilde);  gsl_matrix_free(scratchSpace);
  gsl_matrix_free(U);  gsl_matrix_free(V);  gsl_vector_free(S);  gsl_vector_free(workSpace);
  gsl_vector_free(scratchVector);  gsl_vector_free(scratchVector2);  gsl_vector_free(scratchVector3);
}

void PrototypeDesignBase::_printMatrix(const gsl_matrix* A, FILE* fp) const
{
  int colW  = 7;
  int rowN  = A->size1;
  // int rowN  = 5;
  int colN  = A->size2;
  int begin = 0;
  int end   = begin + colW;
  do {
    int endN = end;
    if (endN > colN) endN = colN;
    cout << endl << "Columns " << begin << " through " << (endN - 1) << ":" << endl << endl;
    for (int m = 0; m < rowN; m++) {
      for (int n = begin; n < endN; n++)
	fprintf(fp, "  %12.4e", gsl_matrix_get(A, m, n));
      fprintf(fp, "\n");
    }
    begin += colW; end += colW;
  } while (begin < colN - 1);
}

void PrototypeDesignBase::_printVector(const gsl_vector* b, FILE* fp) const
{
  int len = b->size;
  for (int n = 0; n < len; n++)
    fprintf(fp, "  %12.4e\n", gsl_vector_get(b, n));
  fprintf(fp, "\n");
}

void PrototypeDesignBase::_solveSingular(gsl_matrix* Ac, gsl_vector* bc, gsl_matrix* K, gsl_vector* dp, double tolerance)
{
  // solve for the particular solution 
  gsl_vector* h_pt = _pseudoInverse(K, dp, tolerance);

  /*
  cout << endl << endl << "h_pt = " << endl;
  _printVector(h_pt);

  cout << endl << endl << "K = " << endl;
  _printMatrix(K);
  */

  // determine the null(K)
  gsl_matrix* K_perp = _nullSpace(K, tolerance);

  // test that K_perp is really the null space of K^T
  /*
  gsl_matrix* test = gsl_matrix_calloc(K->size2, K_perp->size2);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, K, K_perp, 0.0, test);
  printf("MaximumElement(K)          = %g\n", gsl_matrix_max(K));
  printf("MaximumElement(K_perp)     = %g\n", gsl_matrix_max(K_perp));
  printf("MaximumElement(K^T K_perp) = %g\n", gsl_matrix_max(test));
  printf("\n K^T K_perp = \n");
  _printMatrix(test);
  */

  printf("size(K) = (%lu x %lu)\n", K->size1, K->size2);
  printf("size(K_perp) = (%lu x %lu)\n", K_perp->size1, K_perp->size2);
  printf("size(Ac) = (%lu x %lu)\n", Ac->size1, Ac->size2);

  int M                      = K_perp->size1;
  int N                      = K_perp->size2;
  gsl_matrix* A_tilde        = gsl_matrix_calloc(N, N);
  gsl_matrix* scratchSpace   = gsl_matrix_calloc(N, M);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, K_perp, Ac, 0.0, scratchSpace);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, scratchSpace, K_perp, 0.0, A_tilde);

  /*
  cout << endl << endl << "A_tilde = " << endl;
  _printMatrix(A_tilde);
  */

  gsl_matrix* U              = gsl_matrix_calloc(N, N);
  gsl_matrix* V              = gsl_matrix_calloc(N, N);
  gsl_vector* S              = gsl_vector_calloc(N);
  gsl_vector* workSpace      = gsl_vector_calloc(N);
  gsl_matrix_memcpy(U, A_tilde);
  // perform singular value decomposition
  _svd(U, V, S, workSpace);
  double maxSingularValue = gsl_vector_get(S, 0);

  // Calculate h_perp = K_perp * (V * S * U') * K_perp' * (b - A * h_pt)
  gsl_vector* h_perp         = gsl_vector_calloc(M);
  gsl_vector* scratchVector  = gsl_vector_calloc(M);
  gsl_vector* scritchVector  = gsl_vector_calloc(M);
  gsl_vector* scratchVector2 = gsl_vector_calloc(N);
  gsl_vector* scratchVector3 = gsl_vector_calloc(N);
  gsl_vector_memcpy(scratchVector, bc);
  gsl_blas_dgemv(CblasNoTrans, 1.0, Ac, h_pt, 0.0, scritchVector);
  gsl_vector_sub(scratchVector, scritchVector);
  gsl_blas_dgemv(CblasTrans, 1.0, K_perp, scratchVector, 0.0, scratchVector2);
  gsl_blas_dgemv(CblasTrans, 1.0, U, scratchVector2, 0.0, scratchVector3);
  for (int n = 0; n < N; n++)
    if ((gsl_vector_get(S, n) / maxSingularValue) > tolerance)
      gsl_vector_set(scratchVector3, n, gsl_vector_get(scratchVector3, n) / gsl_vector_get(S, n));
  gsl_blas_dgemv(CblasNoTrans, 1.0, V, scratchVector3, 0.0, scratchVector2);
  gsl_blas_dgemv(CblasNoTrans, 1.0, K_perp, scratchVector2, 0.0, h_perp);

  /*
  cout << endl << endl << "h_perp = " << endl;
  _printVector(h_perp);
  */

  // h = h_pt + h_perp
  gsl_vector_memcpy(_prototype, h_pt); gsl_vector_add(_prototype, h_perp);

  /*
  cout << endl << endl << "h = " << endl;
  _printVector(_prototype);
  */

  // free working memory
  gsl_vector_free(h_pt);  gsl_vector_free(h_perp);
  gsl_matrix_free(K_perp); gsl_matrix_free(A_tilde);  gsl_matrix_free(scratchSpace);
  gsl_matrix_free(U);  gsl_matrix_free(V);  gsl_vector_free(S);  gsl_vector_free(workSpace);
  gsl_vector_free(scratchVector);  gsl_vector_free(scritchVector);
  gsl_vector_free(scratchVector2);  gsl_vector_free(scratchVector3);
}

double PrototypeDesignBase::_condition(gsl_matrix* H, gsl_matrix* P)
{
  unsigned rowH = H->size1;
  unsigned colH = H->size2;
  unsigned rowP = P->size1;
  unsigned colP = P->size2;

  if (rowH != rowP)
    throw j_error("Number of columns (%d vs. %d) does not match.", rowH, rowP);

  unsigned ttlCols = colH + colP;
  if (rowH > ttlCols)
    throw j_error("Row and total columns (%d vs. %d) do not match.", rowH, ttlCols);
    
  printf("Constraint Matrix Size = (%d x %d)\n", ttlCols, rowH);
  gsl_matrix* constraintMatrix = gsl_matrix_calloc(ttlCols, rowH);
  gsl_matrix_view h1(gsl_matrix_submatrix(constraintMatrix, /* k1= */ 0, /* k2= */ 0, /* rowN= */ colH, /* colN= */ rowH));
  gsl_matrix_transpose_memcpy(&h1.matrix, H);
  gsl_matrix_view h2(gsl_matrix_submatrix(constraintMatrix, /* k1= */ colH, /* k2= */ 0, /* rowN= */ rowP, /* colN= */ rowP));
  gsl_matrix_memcpy(&h2.matrix, P);

  gsl_matrix* V            = gsl_matrix_calloc(rowH, rowH);
  gsl_vector* singularVals = gsl_vector_calloc(rowH);
  gsl_vector* workSpace    = gsl_vector_calloc(rowH);
  _svd(constraintMatrix, V, singularVals, workSpace);

  double cond = gsl_vector_get(singularVals, 0) / gsl_vector_get(singularVals, rowH - 1);

  gsl_matrix_free(constraintMatrix);  gsl_matrix_free(V);
  gsl_vector_free(singularVals);      gsl_vector_free(workSpace);

  return cond;
}


// ----- methods for class 'AnalysisOversampledDFTDesign' -----
//
// @brief make an analysis filter bank 
// @param int M [in] the number of subbands
// @param int m [in] 
// @param int r [in] the decimation factor
// @param unsigned wpFactor [in] cut-off frequency factor
// @note This is not a QMF bank. 'M' means the FFT length. In functions where QMF filter is assumed, 'Mx2' represents the number of subbands.
//
AnalysisOversampledDFTDesign::
AnalysisOversampledDFTDesign(int M, int m, int r, double wp, int tau)
  : PrototypeDesignBase(M, m, r, wp, tau),
    _error(gsl_vector_calloc(3))
{
}

AnalysisOversampledDFTDesign::~AnalysisOversampledDFTDesign()
{
  gsl_matrix_free(_C);
  gsl_matrix_free(_cpC);
  gsl_vector_free(_error);
}

const gsl_vector* AnalysisOversampledDFTDesign::design(double tolerance)
{
  _calculateAb();

  /*
  cout << "A = " << endl;
  _printMatrix(_A);

  cout << endl << endl << "b = " << endl;
  _printVector(_b);
  */

  _calculateC();

  /*
  cout << "C = " << endl;
  _printMatrix(_C);
  */

  _solve(tolerance);

  /*
  cout << endl << endl << "h = " << endl;
  _printVector(_prototype);
  */
  
  return _prototype;
}

const gsl_vector* AnalysisOversampledDFTDesign::calcError(bool doPrint)
{
  double a, b, e;

  a = _passbandResponseError();
  b = _inbandAliasingDistortion();
  e =  a + b;

  if (doPrint) {
    printf("eps_p = %f\n", a);
    printf("eps_i = %f\n", b);
  }

  gsl_vector_set(_error, 0, a);
  gsl_vector_set(_error, 1, b);
  gsl_vector_set(_error, 2, e);

  return(_error);
}


#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
void AnalysisOversampledDFTDesign::_solve(double tolerance)
{
  printf("Solving for analysis prototype 'h' ... ");  fflush(stdout);

  // use SVD to solve for the filter prototype
  gsl_matrix_add(_A, _C);
  _svd(_A, _C, _singularVals, _workSpace);
  // setting the singular values below the threshold to zero
  const double largestValue = gsl_vector_get(_singularVals, 0);
  const double threshold = _L * largestValue * tolerance;
  for (int n = 0; n < _L; n++) {
    const double evalue = gsl_vector_get(_singularVals, n);
    if ( evalue > threshold ) {
      gsl_vector_set(_scratch, n, evalue);
    } else {
      printf("%d-th singular value is set to 0 because of %e < %e\n", n, evalue, threshold);
      gsl_vector_set(_scratch, n, 0.0);
    }
  }
  gsl_linalg_SV_solve(_A, _C, _scratch, _b, _prototype);

  printf("Done.\n");  fflush(stdout);
}

double AnalysisOversampledDFTDesign::_passbandResponseError()
{
  double res1, res2;
  gsl_vector *Ah;

  // A * h
  Ah = gsl_vector_calloc( _prototype->size );
  gsl_vector_set_zero(Ah);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _cpA, _prototype, 0.0, Ah);

  // h^T * A * h
  gsl_blas_ddot(_prototype, Ah, &res1);

  // h^T * b
  gsl_blas_ddot(_prototype, _cpb, &res2);

  // h^T * E * h - 2 *  h^T * b + 1
  double eps_p = res1 - 2.0 * res2 + 1.0;
  // fprintf(stderr,"eps_p = %f - 2.0 * %f + 1.0 = %f \n", res1, res2, eps_p);

  gsl_vector_free(Ah);
  return 10 * log10(eps_p);
}

double AnalysisOversampledDFTDesign::_inbandAliasingDistortion()
{
  double beta;
  gsl_vector* Ch;

  // C * h
  Ch = gsl_vector_calloc(_prototype->size);
  gsl_vector_set_zero(Ch);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _cpC, _prototype, 0.0, Ch);

  // h^T * C * h
  gsl_blas_ddot(_prototype, Ch, &beta);

  gsl_vector_free(Ch);

  return 10 * log10(beta);
}

// ----- methods for class 'SynthesisOversampledDFTDesign' -----
//
// @brief make an synthesis filter bank 
// @param const gsl_vector* h [in] an analysis filter bank
// @param int M [in] the number of subbands
// @param int m [in] 
// @param int r [in] the decimation factor
// @param unsigned v [in] a weighting factor to emphasize on either the total response error or the residual aliasing distortion
// @param unsigned wpFactor [in] cut-off frequency factor
// @param int tauT [in] the total analysis-synthesis filter bank delay
// @note This is not a QMF bank. 'M' means the FFT length. In functions where QMF filter is assumed, 'Mx2' represents the number of subbands.
//
SynthesisOversampledDFTDesign::
SynthesisOversampledDFTDesign(const gsl_vector* h, int M, int m, int r, double v, double wp, int tau)
  : PrototypeDesignBase(M, m, r, wp, tau),
    _h(gsl_vector_calloc(h->size)), _v(v),
    _E(gsl_matrix_calloc(_L, _L)), _P(gsl_matrix_calloc(_L, _L)), _f(gsl_vector_calloc(_L)),
    _cpE(gsl_matrix_calloc(_L, _L)), _cpP(gsl_matrix_calloc(_L, _L)), _cpf(gsl_vector_calloc(_L)),
    _error(gsl_vector_calloc(3))
{
  printf("Initializing 'SynthesisOversampledDFTDesign'.\n");  fflush(stdout);
  gsl_vector_memcpy(_h, h);
}

SynthesisOversampledDFTDesign::~SynthesisOversampledDFTDesign()
{
  gsl_vector_free(_h);

  gsl_matrix_free(_E);
  gsl_matrix_free(_P);
  gsl_vector_free(_f);
  gsl_matrix_free(_cpE);
  gsl_matrix_free(_cpP);
  gsl_vector_free(_cpf);
  gsl_vector_free(_error);
}

const gsl_vector* SynthesisOversampledDFTDesign::design(double tolerance)
{
  _calculateEfP();

  /*
  cout << "E = " << endl;
  _printMatrix(_E);

  cout << endl << endl << "f = " << endl;
  _printVector(_f);

  cout << "P = " << endl;
  _printMatrix(_P);
  */

  _solve(tolerance);

  /*
  cout << endl << endl << "g = " << endl;
  _printVector(_prototype);
  */

  return _prototype;
}

const gsl_vector* SynthesisOversampledDFTDesign::calcError(bool doPrint)
{
  double eps_t = _totalResponseError();
  double eps_r = _residualAliasingDistortion();
  double e     =  eps_t + _v * eps_r;

  if (doPrint) {
    printf("eps_t = %f\n", eps_t);
    printf("eps_r = %f\n", eps_r);
  }

  gsl_vector_set(_error, 0, eps_t);
  gsl_vector_set(_error, 1, eps_r);
  gsl_vector_set(_error, 2, e);

  return(_error);
}

void SynthesisOversampledDFTDesign::_calculateEfP()
{
  printf("Calculating 'E', 'P', and 'f' ... ");  fflush(stdout);

  gsl_matrix_set_zero( _E );
  gsl_vector_set_zero( _f );
  gsl_matrix_set_zero( _P );

  int tauT = 2 * _tau;
  for (int m = 0; m < _L; m++) {
    for (int n = 0; n < _L; n++) {

      for (int k = 0; k <= (2*_m) ; k++) {
        const int kM = k * _M;
        if ((kM - m) >= 0 && (kM - m) < _L && (kM - n) >= 0 && (kM - n) < _L ) {
          gsl_matrix_set(_E, m, n, gsl_matrix_get(_E, m, n) +
                         gsl_vector_get(_h, (kM-m)) * gsl_vector_get(_h, (kM-n)));
        }
      }

      int factor = ((m-n) % _D == 0) ? _D - 1 : -1;
      for (int k = -_L; k <= _L; k++) {
        if ((k + n) >= 0 && (k + m) >= 0 && (k + n) < _L && (k + m) < _L ) {
          gsl_matrix_set(_P, m, n, gsl_matrix_get(_P, m, n) +
                         gsl_vector_get(_h, k+n) * gsl_vector_get(_h, k+m) * factor);
        }
      }
    }

    if ((tauT - m) >=0 && (tauT - m) < _L){
      gsl_vector_set(_f, m, gsl_vector_get(_h, tauT - m));
    }
  }

  gsl_matrix_scale(_E, (double) (_M / _D) * (_M / _D));
  gsl_vector_scale(_f, (double) (_M / _D )); // (_M / ( PI * _D ) )); The de Haan's paper has PI in the divisor but it should be removed.
  gsl_matrix_scale(_P, (double) _M / ((double) _D   * (double) _D));

  gsl_matrix_memcpy(_cpE, _E);
  gsl_vector_memcpy(_cpf, _f);
  gsl_matrix_memcpy(_cpP, _P);

  printf("Done.\n");  fflush(stdout);
}

void SynthesisOversampledDFTDesign::_solve(double tolerance)
{
  printf("Solving for synthesis prototype 'g' ... ");  fflush(stdout);

  // use SVD to solve for the filter prototype
  gsl_matrix_scale(_P, _v);
  gsl_matrix_add(_E, _P);
  _svd(_E, _P, _singularVals, _workSpace);
  // setting the singular values below the threshold to zero
  const double largestValue = gsl_vector_get(_singularVals, 0);
  const double threshold = _L * largestValue * tolerance;
  for (int n = 0; n < _L; n++) {
    const double evalue = gsl_vector_get(_singularVals, n);
    if ( evalue > threshold ) {
      gsl_vector_set(_scratch, n, evalue);
    } else {
      printf("%d-th singular value is set to 0 because of %e < %e\n", n, evalue, threshold);
      gsl_vector_set(_scratch, n, 0.0);
    }
  }
  gsl_linalg_SV_solve(_E, _P, _scratch, _f, _prototype);

  printf("Done.\n");  fflush(stdout);
}

double SynthesisOversampledDFTDesign::_totalResponseError()
{

  double res1, res2;
  gsl_vector* Eg;

  /*
  printf("E = \n");
  _printMatrix(_cpE);

  printf("f = \n");
  _printVector(_cpf);
  */

  // E * g
  Eg = gsl_vector_calloc(_prototype->size);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _cpE, _prototype, 0.0, Eg);

  // g^T * A * h
  gsl_blas_ddot(_prototype, Eg, &res1);

  // g^T * b
  gsl_blas_ddot(_prototype, _cpb, &res2);

  // g^T * E * g - 2 *  g^T * b + 1
  double eps_t = res1 - 2.0 * res2 + 1.0;
  // fprintf(stderr,"eps_t = %f - 2.0 * %f + 1.0 = %f \n", res1, res2, eps_t);

  gsl_vector_free(Eg);
  return 10.0 * log10(eps_t);
}

double SynthesisOversampledDFTDesign::_residualAliasingDistortion()
{
  double      delta = 0.0;
  gsl_vector* Pg;

  // P * g
  Pg = gsl_vector_calloc(_prototype->size);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _cpP, _prototype, 0.0, Pg);

  // g^T * P * g
  gsl_blas_ddot(_prototype, Pg, &delta);

  gsl_vector_free(Pg);

  return 10 * log10(delta);
}


// ----- methods for class 'AnalysisNyquistMDesign' -----
//
AnalysisNyquistMDesign::
AnalysisNyquistMDesign(int M, int m, int r, double wpFactor, int tau_h)
  : AnalysisOversampledDFTDesign(M, m, r, wpFactor, tau_h),
    _F(gsl_matrix_calloc(_L, _m)), _d(gsl_vector_calloc(_m)),
    _K(gsl_matrix_calloc(_L, _L + _m)), _dp(gsl_vector_calloc(_L + _m)) { }

AnalysisNyquistMDesign::~AnalysisNyquistMDesign()
{
  gsl_matrix_free(_F);  gsl_vector_free(_d);
  gsl_matrix_free(_K);  gsl_vector_free(_dp);
}

void AnalysisNyquistMDesign::_solve(double tolerance)
{
  cout << "Solving for analysis prototype 'h' ... " << endl;

  _calculateFd();
  if (_condition(_F, _C) < 1.0 / tolerance) {
    cout << "Using alternate solution 4 for analysis prototype ..." << endl;
    _solveNonSingular(_F, _d, _C, tolerance);
  } else {
    cout << "Using alternate solution 3 for analysis prototype ..." << endl;
    _calculateKdp();
    _solveSingular(_A, _b, _K, _dp, tolerance);
  }
  cout << "Done." << endl;
}

// Create the constraint matrix
void AnalysisNyquistMDesign::_calculateFd()
{
  gsl_matrix_set_zero(_F);  gsl_vector_set_zero(_d);
  for (int n = 0; n < _m; n++) {
    gsl_matrix_set(_F, n * _M, n, 1.0);
    if (n == _m / 2) gsl_vector_set(_d, n, 1.0 / _M);
  }

  /*
  cout << "F = " << endl;
  _printMatrix(_F);

  cout << endl << endl << "d = " << endl;
  _printVector(_d);
  */
}

void AnalysisNyquistMDesign::_calculateKdp()
{
  // form the combined constraint matrix K = [F C]
  gsl_matrix_view k1(gsl_matrix_submatrix(_K, /* k1= */ 0, /* k2= */ 0, /* rowN= */ _L, /* colN= */ _m));
  gsl_matrix_memcpy(&k1.matrix, _F);
  gsl_matrix_view k2(gsl_matrix_submatrix(_K, /* k1= */ 0, /* k2= */ _m, /* rowN= */ _L, /* colN= */ _L));
  gsl_matrix_memcpy(&k2.matrix, _C);

  gsl_vector_set_zero(_dp);
  gsl_vector_view dp1(gsl_vector_subvector(_dp, /* k= */ 0, /* size= */ _m));
  gsl_vector_memcpy(&dp1.vector, _d);
}


// ----- methods for class 'SynthesisNyquistMDesign' -----
//
SynthesisNyquistMDesign::
SynthesisNyquistMDesign(const gsl_vector* h, int M, int m, int r,
			double wpFactor, int tau_g)
  : SynthesisOversampledDFTDesign(h, M, m, r, /* v= */ 0.0, wpFactor, tau_g),
    _H(gsl_matrix_calloc(_L, 2*_m)), _c0(gsl_vector_calloc(2*_m)),
    _J(gsl_matrix_calloc(_L, _L + 2*_m)), _cp(gsl_vector_calloc(_L + 2*_m)) { }

SynthesisNyquistMDesign::~SynthesisNyquistMDesign()
{
  gsl_matrix_free(_H);  gsl_vector_free(_c0);
  gsl_matrix_free(_J);  gsl_vector_free(_cp);
}

const gsl_vector* SynthesisNyquistMDesign::calcError(bool doPrint)
{
  double eps_t = _totalResponseError();
  double eps_p = _passbandResponseError();
  double eps_r = _residualAliasingDistortion();

  if (doPrint) {
    printf("eps_t = %f\n", eps_t);
    printf("eps_p = %f\n", eps_p);
    printf("eps_r = %f\n", eps_r);
  }

  gsl_vector_set(_error, 0, eps_t);
  gsl_vector_set(_error, 1, eps_r);

  return(_error);
}

double SynthesisNyquistMDesign::_passbandResponseError()
{
  double res1, res2;

  // A * h
  gsl_vector* Ah = gsl_vector_calloc( _prototype->size );
  gsl_vector_set_zero(Ah);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _cpA, _prototype, 0.0, Ah);

  // h^T * A * h
  gsl_blas_ddot(_prototype, Ah, &res1);

  // h_T * b
  gsl_blas_ddot(_prototype, _cpb, &res2);

  // h^T * E * h + 2 *  h_T * b + 1
  double eps_p = res1 - 2.0 * res2 + 1.0;
  // fprintf(stderr,"eps_p = %f - 2.0 * %f + 1.0 = %f \n", res1, res2, eps_p);

  gsl_vector_free(Ah);
  return 10 * log10(eps_p);
}

void SynthesisNyquistMDesign::_calculateHc0()
{
  gsl_vector_set_zero(_c0);
  gsl_vector_set(_c0, _m, ((double) _D) / _M);

  // a row vector corresponds to h_k in the report
  gsl_matrix_set_zero(_H);
  int m = int(_m);
  int M = int(_M);
  for (int n = 0; n < 2*_m; n++) {
    int minK = max(0, 1+(n-m)*M);
    int maxK = min(n*M, m*M-1);

    for (int k = minK; k <= maxK; k++)
      gsl_matrix_set(_H, k, n, gsl_vector_get(_h, n*M - k));
  }
}

void SynthesisNyquistMDesign::_calculateJcp()
{
  gsl_matrix_view j1(gsl_matrix_submatrix(_J, /* k1= */ 0, /* k2= */ 0, /* rowN= */ _L, /* colN= */ 2*_m));
  gsl_matrix_memcpy(&j1.matrix, _H);
  gsl_matrix_view j2(gsl_matrix_submatrix(_J, /* k1= */ 0, /* k2= */ 2*_m, /* rowN= */ _L, /* colN= */ _L));
  gsl_matrix_memcpy(&j2.matrix, _P);

  gsl_vector_set_zero(_cp);
  gsl_vector_view cp(gsl_vector_subvector(_cp, /* k= */ 0, /* size= */ 2*_m));
  gsl_vector_memcpy(&cp.vector, _c0);
}

void SynthesisNyquistMDesign::_solve(double tolerance)
{
  printf("Solving for synthesis prototype 'g' ... ");  fflush(stdout);

  _calculateAb();  _calculateHc0();
  if (_condition(_H, _P) < 1.0 / tolerance) {
    cout << "Using alternate solution 4 for synthesis prototype ..." << endl;
    _solveNonSingular(_H, _c0, _P, tolerance);
  } else {
    cout << "Using alternate solution 3 for synthesis prototype ..." << endl;
    _calculateJcp();
    _solveSingular(_A, _b, _J, _cp, tolerance);
  }

  printf("Done.\n");  fflush(stdout);
}


// ----- methods for class 'SynthesisNyquistMDesignCompositeResponse' -----
//
SynthesisNyquistMDesignCompositeResponse::
SynthesisNyquistMDesignCompositeResponse(const gsl_vector* h, int M, int m, int r,
					 double wpFactor, int tau)
  : SynthesisNyquistMDesign(h, M, m, r, wpFactor, tau),
    _sincValues(new double[2 * _L])
{
  for (unsigned m = 0; m < 2 * _L; m++)
    _sincValues[m] = -HUGE;
}

SynthesisNyquistMDesignCompositeResponse::
~SynthesisNyquistMDesignCompositeResponse() { delete[] _sincValues; }

void SynthesisNyquistMDesignCompositeResponse::_calculateAb()
{
  printf("Calculating modified 'A' and 'b' ... ");  fflush(stdout);

  int tauT = 2 * _tau;
  for (int m = 0; m < _L; m++) {
    if (m % 10 == 0) { printf("."); fflush(stdout); }
    for (int n = 0; n < _L; n++) {

      // calculate A_{m,n}
      for (int p = 0; p < _L; p++) {
	for (int q = 0; q < _L; q++) {
	  gsl_matrix_set(_A, m, n, gsl_matrix_get(_A, m, n)
			 + gsl_vector_get(_h, p) * gsl_vector_get(_h, q) * _sinc(m-n+q-p));
	}
      }
    }

    // calculate b_{m}
    for (int p = 0; p < _L; p++)
      gsl_vector_set(_b, m, gsl_vector_get(_b, m) + gsl_vector_get(_h, p) * _sinc(tauT-m-p));
  }

  gsl_matrix_scale(_A, (1.0 / _D) * (1.0 / _D));
  gsl_vector_scale(_b, (1.0 / _D));

  gsl_matrix_memcpy(_cpA, _A);
  gsl_vector_memcpy(_cpb, _b);

  printf("Done\n");  fflush(stdout);
}

