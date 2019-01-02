#ifndef TAYLORSERIES_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <common/refcount.h>
#include "common/jexception.h"
#include <list>
#include <vector>

using namespace std;

double LogAdd(double x, double y);
double LogSub(double x, double y);
double logFactorial( unsigned int n );

#define NO_FILE_SYMBOL "NONE"
class nonamePdf {
 protected:
  vector<float>  _points; /* @note it must be positive and sorted. */
  int   _maxCoeffs;
  double **_coeffs;       /* [_points.size()][nCoeffs] */
  double **_coeffsLog;

 protected:
  void clear();

 public:
  nonamePdf();
  ~nonamePdf();
  bool loadCoeffDescFile( const String &coefDescfn );

 private:
  bool loadCoeffFiles( char *coeffn, char *logCoeffn, float a );
  int  loadCoeffFile(  char *coeffn, int preNPoints, int idx, bool logProb );
};

class gammaPdf : public nonamePdf {

 public:
  gammaPdf( int numberOfVariate = 2 );
  ~gammaPdf();
  double calcLog( double x, int N );
  double calcDerivative1( double x, int N );
  void   bi( int printLevel = 0);
  void   four( int printLevel = 0 );

  void   printCoeff();
 private:
  void allocate();
  int  indexOfCoeffArray( double x );
  bool interpolate( int printLevel );
};

typedef  refcount_ptr<nonamePdf> nonamePdfPtr;
typedef  Inherit<gammaPdf, nonamePdfPtr> gammaPdfPtr;

#endif
