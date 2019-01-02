#ifndef LINPACK_C_H
#define LINPACK_C_H

# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <complex>

using namespace std;

int cchdc ( complex <float> a[], int lda, int p, int ipvt[], int job );
int cchdd ( complex <float> r[], int ldr, int p, complex <float> x[], 
  complex <float> z[], int ldz, int nz, complex <float> y[], float rho[], 
  float c[], complex <float> s[] );
void cchex ( complex <float> r[], int ldr, int p, int k, int l, 
  complex <float> z[], int ldz, int nz, float c[], complex <float> s[], int job );
void cchud ( complex <float> r[], int ldr, int p, complex <float> x[], 
  complex <float> z[], int ldz, int nz, complex <float> y[], float rho[], 
  float c[], complex <float> s[] );
float cgbco ( complex <float> abd[], int lda, int n, int ml, int mu, int ipvt[] );
void cgbdi ( complex <float> abd[], int lda, int n, int ml, int mu, int ipvt[], 
  complex <float> det[2] );
int cgbfa ( complex <float> abd[], int lda, int n, int ml, int mu, int ipvt[] );
void cgbsl ( complex <float> abd[], int lda, int n, int ml, int mu, 
  int ipvt[], complex <float> b[], int job );
float cgeco ( complex <float> a[], int lda, int n, int ipvt[] );
void cgedi ( complex <float> a[], int lda, int n, int ipvt[], 
  complex <float> det[2], int job );
int cgefa ( complex <float> a[], int lda, int n, int ipvt[] );
void cgesl ( complex <float> a[], int lda, int n, int ipvt[], 
  complex <float> b[], int job );
int cgtsl ( int n, complex <float> c[], complex <float> d[], 
  complex <float> e[], complex <float> b[] );
float chico ( complex <float> a[], int lda, int n, int ipvt[] );
void chidi ( complex <float> a[], int lda, int n, int ipvt[], float det[2], 
  int inert[3], int job );
int chifa ( complex <float> a[], int lda, int n, int ipvt[] );
void chisl ( complex <float> a[], int lda, int n, int ipvt[], 
  complex <float> b[] );
float chpco ( complex <float> ap[], int n, int ipvt[] );
void chpdi ( complex <float> ap[], int n, int ipvt[], float det[2], 
  int inert[3], int job );
int chpfa ( complex <float> ap[], int n, int ipvt[] );
void chpsl ( complex <float> ap[], int n, int ipvt[], complex <float> b[] );
float cpbco ( complex <float> abd[], int lda, int n, int m, int *info );
void cpbdi ( complex <float> abd[], int lda, int n, int m, float det[2] );
int cpbfa ( complex <float> abd[], int lda, int n, int m );
void cpbsl ( complex <float> abd[], int lda, int n, int m, complex <float> b[] );
float cpoco ( complex <float> a[], int lda, int n, int *info );
void cpodi ( complex <float> a[], int lda, int n, float det[2], int job );
int cpofa ( complex <float> a[], int lda, int n );
void cposl ( complex <float> a[], int lda, int n, complex <float> b[] );
float cppco ( complex <float> ap[], int n, int *info );
void cppdi ( complex <float> ap[], int n, float det[2], int job );
int cppfa ( complex <float> ap[], int n );
void cppsl ( complex <float> ap[], int n, complex <float> b[] );
void cptsl ( int n, complex <float> d[], complex <float> e[], 
  complex <float> b[] );
void cqrdc ( complex <float> x[], int ldx, int n, int p, 
  complex <float> qraux[], int ipvt[], int job );
int cqrsl ( complex <float> x[], int ldx, int n, int k, complex <float> qraux[], 
  complex <float> y[], complex <float> qy[], complex <float> qty[], 
  complex <float> b[], complex <float> rsd[], complex <float> xb[], int job );
float csico ( complex <float> a[], int lda, int n, int ipvt[] );
void csidi ( complex <float> a[], int lda, int n, int ipvt[], 
  complex <float> det[2], int job );
int csifa ( complex <float> a[], int lda, int n, int ipvt[] );
void csisl ( complex <float> a[], int lda, int n, int ipvt[], complex <float> b[] );
float cspco ( complex <float> ap[], int n, int ipvt[] );
void cspdi ( complex <float> ap[], int n, int ipvt[], complex <float> det[2], int job );
int cspfa ( complex <float> ap[], int n, int ipvt[] );
void cspsl ( complex <float> ap[], int n, int ipvt[], complex <float> b[] );
int csvdc ( complex <float> x[], int ldx, int n, int p, 
  complex <float> s[], complex <float> e[], complex <float> u[], int ldu, 
  complex <float> v[], int ldv, int job );
float ctrco ( complex <float> t[], int ldt, int n, int job );
int ctrdi ( complex <float> t[], int ldt, int n, complex <float> det[2], 
  int job );
int ctrsl ( complex <float> t[], int ldt, int n, complex <float> b[], int job );
float r4_max ( float x, float y );
void srotg ( float *sa, float *sb, float *c, float *s );

#endif
