#ifndef BLAS1_C_H
#define BLAS1_C_H
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <complex>

using namespace std;

float cabs1 ( complex <float> z );
float cabs2 ( complex <float> z );
void caxpy ( int n, complex <float> ca, complex <float> cx[], 
  int incx, complex <float> cy[], int incy );
void ccopy ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy );
complex <float> cdotc ( int n, complex <float> cx[], int incx, 
  complex <float> cy[], int incy );
complex <float> cdotu ( int n, complex <float> cx[], int incx, 
  complex <float> cy[], int incy );
float cmach ( int job );
void crotg ( complex <float> *ca, complex <float> cb, float *c, 
  complex <float> *s );
void cscal ( int n, complex <float> ca, complex <float> cx[], int incx );
complex <float> csign1 ( complex <float> z1, complex <float> z2 );
complex <float> csign2 ( complex <float> z1, complex <float> z2 );
void csrot ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy, float c, float s );
void csscal ( int n, float sa, complex <float> cx[], int incx );
void cswap ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
int icamax ( int n, complex <float> x[], int incx );
bool lsame ( char ca, char cb );
float r4_abs ( float x );
float r4_sign ( float x );
float scasum ( int n, complex <float> x[], int incx );
float scnrm2 ( int n, complex <float> x[], int incx );
void xerbla ( char *srname, int info );
#endif
