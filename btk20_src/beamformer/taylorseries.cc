#include"taylorseries.h"
#include"coefficients.cc"

#define LZERO   (-1.0E10)   /* ~log(0) */
#define LSMALL  (-0.5E10)   /* log values < LSMALL are set to LZERO */
#define MINEARG (-708.3)    /* lowest  exp() arg */
#define MAXEARG  (708.3)    /* biggest exp() arg */

static double logTaylorSeries( double x, double *pCoeff, int N, float a );
static double taylorSeriesWithLC( double x, double *pCoeff, int N, float a );
static double derivative1TaylorSeries( double x, double *pCoeff, int N, float a );

nonamePdf::nonamePdf()
{
  _maxCoeffs = 0;
  _coeffs = NULL;
  _points.resize(0);
  _coeffsLog = NULL;
}

nonamePdf::~nonamePdf()
{
  this->clear();
}

void nonamePdf::clear()
{
  if( NULL!=_coeffs ){
    int N = (int) _points.size();
    for(int i=0;i<N;i++)
      free( _coeffs[i] );
    free( _coeffs );
  }
  if( NULL!=_coeffsLog ){
    int N = (int) _points.size();
    for(int i=0;i<N;i++)
      free( _coeffsLog[i] );
    free( _coeffsLog );
  }

  _maxCoeffs = 0;
  _coeffs = NULL; 
  _coeffsLog = NULL; 
  _points.resize(0);
}

/**
   @brief load the coefficient files which list the coefficients of series expansion.

   @param const String &coefDescfn[in] file listing values about where the series is expanded and files which list coefficients.
   @return
 */
bool nonamePdf::loadCoeffDescFile( const String &coefDescfn )
{
  FILE *fp;

  this->clear();
  fp = fopen( coefDescfn.c_str(), "r" );
  if( NULL==fp ){
    fprintf(stderr,"loadCoeffDescFile : cannot open file %s\n", coefDescfn.c_str() );
    return(false);
  }
  
  for(;;){
    float a;
    char    coefFileName[FILENAME_MAX];
    char logCoefFileName[FILENAME_MAX];

    if( 3 != fscanf( fp, "%f %s %s", &a, coefFileName, logCoefFileName ) )
      break;
    if( false==this->loadCoeffFiles( coefFileName, logCoefFileName, a ) ){
      fclose(fp);
      return(false);
    }
  }

  fclose(fp);

  return(true);
}

/**
   @brief load the coefficient file which lists the coefficients of series expansion.
   @param char *coeffn[in] file which lists series coefficients of a pdf.
   @param char *char *logCoeffn[in] file which lists series coefficients of a log-pdf.
   @param float a[in] the point where the series is expanded.
 */
bool nonamePdf::loadCoeffFiles( char *coeffn, char *logCoeffn, float a )
{
  vector<float> tmp( _points );
  int preNPoints = (int)_points.size();
  int idx;
  int maxCoeffs1, maxCoeffs2;

  if( a < 0 ){
    fprintf(stderr,"loadCoeffFiles: Invalid value %f >= 0!\n", a);
    return(false);
  }
  
  /* decide which index in the array we should insert the coefficients */
  idx = preNPoints;
  for(int i=0;i<preNPoints;i++){
    if( a == _points[i] ){
      fprintf(stderr,"loadCoeffFiles: The coefficients of the series expansion about %f are already inserted\n",a);
      return(false);
    }
    if( a < _points[i] ){
      idx = i;
      break;
    }
  }

  // expand the container
  _points.resize( preNPoints + 1 );
  // insert the value which the series are expanded around.
  for(int i=idx+1;i<(int)_points.size();i++){
    _points[i] = tmp[i-1];
  }
  _points[idx] = a;

  maxCoeffs1 = this->loadCoeffFile( coeffn,    preNPoints, idx ,false );
  if( maxCoeffs1 < 0 )
    return(false);
  
  /* If the string represents that there is no coefficient fils, we skip to read it */
  if( strcmp( logCoeffn, NO_FILE_SYMBOL ) != 0 ){
    maxCoeffs2 = this->loadCoeffFile( logCoeffn, preNPoints, idx ,true );
    if( maxCoeffs2 < 0 )
      return(false);
  }
  else{
    /* Since it is extremely time-consuming to calculate a log coefficient, that file is sometimes not created */

    double **oldCoeffs = _coeffsLog;
    double **newCoeffs = (double **)malloc(_points.size()*sizeof(double *));
    for(int i=0;i<preNPoints;i++) newCoeffs[i] = oldCoeffs[i];
    free( oldCoeffs );
    if( NULL==newCoeffs )
      throw jallocation_error("nonamePdf:malloc failed\n");
    for(int i=(int)_points.size()-1;i>idx;i--)
      newCoeffs[i] = newCoeffs[i-1];
    newCoeffs[idx] = NULL;
    _coeffsLog = newCoeffs;
    maxCoeffs2 = 0;
  }

  /* make the array size for log-coefficients the same as one for coefficients */
  if( maxCoeffs2 > maxCoeffs1 ){
    for(int i=0;i<(int)_points.size();i++){
      _coeffs[i] = (double *)realloc(_coeffs[i],_maxCoeffs*sizeof(double));
      if( NULL==_coeffs[i] ){
	throw jallocation_error("nonamePdf:malloc failed\n");
      }
    }
  }

  return(true);
}

/**
   @ read the coefficients from the file and  update members in this class.
 */
int nonamePdf::loadCoeffFile( char *coeffn, int preNPoints, int idx, bool logProb )
{
  list<double> buffer;
  double *insCoeff;
  int  preMaxCoeff;
  double **oldCoeffs;
  double **newCoeffs;

  if( logProb == false )
    oldCoeffs = _coeffs;
  else
    oldCoeffs = _coeffsLog;

  // read series coefficients for a pdf
  {
    FILE *fpCoef = fopen( coeffn, "r" );
    if( NULL==fpCoef ){
      fprintf(stderr,"cannot open file %s\n", coeffn );
      return(-1);
    }
    
    buffer.clear();
    for(;;){// read numbers in a file and put them into a buffer.
      double val;
      if( 1 != fscanf( fpCoef, "%lf", &val ) )
	break;
      buffer.push_back( val );
    }
    fclose(fpCoef);
  }

  // put the coefficients read from the file into the member.
  newCoeffs = (double **)malloc(_points.size()*sizeof(double *));
  for(int i=0;i<preNPoints;i++) newCoeffs[i] = oldCoeffs[i];
  free( oldCoeffs );
  if( NULL==newCoeffs ){
    throw jallocation_error("nonamePdf:malloc failed\n");
  }

  preMaxCoeff = _maxCoeffs;
  if( buffer.size() > _maxCoeffs ){
    _maxCoeffs = buffer.size();
    for(int i=0;i<preNPoints;i++){
      newCoeffs[i] = (double *)realloc(newCoeffs[i],_maxCoeffs*sizeof(double));
      if( NULL==newCoeffs[i] ){
	throw jallocation_error("nonamePdf:malloc failed\n");
      }
      for(int j=preMaxCoeff;j<_maxCoeffs;j++)
	newCoeffs[i][j] = 0.0;
    }
  }

  insCoeff = (double *)malloc(_maxCoeffs*sizeof(double));
  for(int i=0;i<_maxCoeffs;i++) insCoeff[i] = 0.0;
  if( NULL==insCoeff ){
    throw jallocation_error("nonamePdf:calloc failed\n");
  }
  for(int i=0;;i++){
    insCoeff[i] = buffer.front();
    buffer.pop_front();
    if( buffer.size() == 0 )
      break;
  }

  for(int i=(int)_points.size()-1;i>idx;i--){
    newCoeffs[i] = newCoeffs[i-1];
  }
  newCoeffs[idx] = insCoeff;  

  if( logProb == false )
    _coeffs = newCoeffs;
  else
    _coeffsLog = newCoeffs;

  return( _maxCoeffs);
}


/**
   @brief 
   @param int numberOfVariate[in] 2 / 4
 */
gammaPdf::gammaPdf( int numberOfVariate )
{
  if( 2==numberOfVariate )
    this->bi();
  else if( 4==numberOfVariate )
    this->four();
}


gammaPdf::~gammaPdf()
{
}

/**
   @brief set the coefficients of the series expansion for bi-variate gamma pdf.
 */
void gammaPdf::bi( int printLevel )
{
  this->clear();
  this->allocate();

  for(int i=0;i<(int)_points.size();i++)
    _points[i] = s_argdg2[i];

  for(int i=0;i<(int)_points.size();i++){
    for(int j=0;j<s_maxCoeffGA;j++){
      _coeffs[i][j] = s_dg2[i][j];
    }
    for(int j=0;j<s_maxCoeffLGA;j++){
      _coeffsLog[i][j] = s_logdg2[i][j];
    }
  }

}

/**
   @brief set the coefficients of the series expansion for 4-variate gamma pdf.
 */
void gammaPdf::four( int printLevel )
{
  this->clear();
  this->allocate();

  for(int i=0;i<(int)_points.size();i++)
    _points[i] = s_argdg2[i];
	
  for(int i=0;i<(int)_points.size();i++){
    for(int j=0;j<s_maxCoeffGA;j++){
      _coeffs[i][j] = s_dg4[i][j];
    }
    for(int j=0;j<s_maxCoeffLGA;j++){
      _coeffsLog[i][j] = s_logdg4[i][j];
    }
  }

}

void gammaPdf::allocate()
{
  _points.resize(s_nPointsGA);
  _maxCoeffs = (s_maxCoeffGA>s_maxCoeffLGA)? s_maxCoeffGA:s_maxCoeffLGA;
  
  _coeffs = (double **)malloc(_points.size()*sizeof(double *));
  _coeffsLog  = (double **)malloc(_points.size()*sizeof(double *));
  if(  NULL==_coeffs || NULL==_coeffsLog ){
    throw jallocation_error("gammaPdf:malloc failed\n");
  }

  for(int i=0;i<(int)_points.size();i++){
    _coeffs[i]     = (double *)calloc(_maxCoeffs, sizeof(double));
    _coeffsLog[i]  = (double *)calloc(_maxCoeffs, sizeof(double));
    if( NULL==_coeffs[i] || NULL==_coeffsLog[i] ){
      throw jallocation_error("gammaPdf:malloc failed\n");
    }
  }
}

void gammaPdf::printCoeff()
{
  for(int i=0;i<(int)_points.size();i++)
    fprintf(stderr,"%f ",_points[i]);
  fprintf(stderr,"\n");

  for(int i=0;i<(int)_points.size();i++){
    fprintf(stderr,"/* a = %f */\n",_points[i]);
    for(int j=0;j<_maxCoeffs;j++)
      fprintf(stderr,"%f ",_coeffs[i][j]);
    fprintf(stderr,"\n");
  }

  for(int i=0;i<(int)_points.size();i++){
    fprintf(stderr,"/* a = %f */\n",_points[i]);
    for(int j=0;j<_maxCoeffs;j++)
      fprintf(stderr,"%f, ",_coeffsLog[i][j]);
    fprintf(stderr,"\n");
  }
}

/**
   @brief search an array which has the coefficients of series expansion near x;
 */
int gammaPdf::indexOfCoeffArray( double x )
{
  int left  = 0;
  int right = (int)_points.size()-1;
  int mid;
  int ans   = -1;
  double d1,d2;

  while(left<=right){
    mid  = ( left + right ) / 2;
    if( x < _points[mid] ){
      if( mid == 0 )
	return(mid);
      d1 = x - _points[mid-1];
      if( d1 >= 0 ){
	d2 = _points[mid] - x;
	if( d1 < d2 )
	  return(mid-1);
	else
	  return(mid);
      }
      right = mid - 1;
    }
    else if( x > _points[mid]  ){
      if( mid == (int)_points.size()-1 ){
	//if( absx > ( _points.back() + ( _points.back() - _points.at(_points.size()-2) ) / 2 ) )
	//fprintf(stderr,"Add the series coefficients expanded about %f\n", x );
	return(mid);
      }
      d2 = _points[mid+1] - x;
      if( d2 >= 0 ){
	d1 = x - _points[mid];
	if( d1 < d2 )
	  return(mid);
	else
	  if( mid < (int)_points.size()-2 )
	    return(mid+1);
	  else
	    return(mid);
      }
      left = mid + 1;
    }
    else{
      ans = mid;
      return(mid);
    }
  }
  if( x <_points[0] ){
    return(0);
  }
  // if( absx > ( _points.back() + ( _points.back() - _points.at(_points.size()-2) ) / 2 ) )
  // fprintf(stderr,"Add the series coefficients expanded about %f\n", x );
  
  return( (int)_points.size()-1 );
}

/**
   @brief calculate the Taylor series of Gamma pdf as
          y =  sum_{n=0}^{N}  (x-a)^n * g_n(a) / n!.
          In order to avoid the floating error, this will be done in log-domain. 
          Accordingly, this code looks like complicated becauce I ensure that the argument of log() is positive.

   @param double x[in] arg
   @param int N[in] the number of coefficients
   @return the log probability of gamma pdf.
 */
double gammaPdf::calcLog( double x, int N )
{
  int    idx;
  float    a;
  double *pCoeff;

  if( N > _maxCoeffs ){
    fprintf(stderr,"WARN: calcLog: %d > %d\n",N,_maxCoeffs);
    N = _maxCoeffs;
  }
  
  idx = indexOfCoeffArray( x );
  if( idx == -1 ){
    fprintf(stderr,"%f is over the range\n",x);
    return(LZERO);
  }

  a = _points[idx];
  if( _coeffsLog[idx] == NULL ){// We don't have coefficients for log(Meijer G function)
    pCoeff = &_coeffs[idx][0];
    if( idx==_points.size()-1 && x > a ){
      if( _maxCoeffs > 1 ){
	double gval = pCoeff[0] + pCoeff[1] * (x - a);
	if( gval > 0 )
	  return( log( gval ) );
      }
      else
	fprintf(stderr,"The number of coefficients is not enough\n");
      return(LZERO);
    }
    return( logTaylorSeries( x, pCoeff, N, a ) );
  }

  pCoeff = &_coeffsLog[idx][0];
  if( idx==_points.size()-1 && x > a ){
    if( _maxCoeffs > 1 )
      return( pCoeff[0] + pCoeff[1] * (x - a) );
    else
      fprintf(stderr,"The number of coefficients is not enough\n");
    return(LZERO);
  }

  return( taylorSeriesWithLC( x, pCoeff, N, a ) );
}

/**
   @brief calculate the derivative of Taylor series as
          y =  sum_{n=0}^{N}  (x-a)^n * g_n(a) / n!.
          In order to avoid the floating error, this will be done in log-domain. 
          Accordingly, this code looks like complicated becauce I ensure that the argument of log() is positive.

   @param double x
   @param int N[in] the number of coefficients
 */
double gammaPdf::calcDerivative1( double x, int N )
{
  int    idx;
  double *pCoeff;
  float        a;

  if( N > _maxCoeffs ){
    fprintf(stderr,"WARN : calcDerivative1: %d > %d\n",N,_maxCoeffs);
    N = _maxCoeffs;
  }

  idx = indexOfCoeffArray( x );
  if( idx < 0 ){
    fprintf(stderr,"%f is over the range\n",x);
    return(LZERO);
  }
  
  pCoeff = &_coeffs[idx][0];
  a = _points[idx];

  if( idx==_points.size()-1 ){
    if( _maxCoeffs > 1 )
      return( pCoeff[1] );
    else
      fprintf(stderr,"The number of coefficients is not enough\n");
    return(LZERO);
  }
  
  return( derivative1TaylorSeries( x, pCoeff, N, a ) );
}


double LogAdd(double x, double y)
{
  double temp,diff,z;

   if (x<y) {
      temp = x; x = y; y = temp;
   }
   if ( x <= LSMALL && y > LZERO )
     return(y);
   if ( y <= LSMALL && x > LZERO )
     return(x);
   diff = y-x;
   if (diff< -log(-LZERO) ){
     return  (x<LSMALL)?LZERO:x;
   }
   
   z = exp(diff);
   return x+log(1.0+z);
}

double LogSub(double x, double y)
{
  double diff,z;

  if( x < y ){
    fprintf(stderr,"LogSub: %f > %f in order to avoid log(-z).\n",x,y);
    return(LZERO);
  }

  if( y <= LSMALL ){
    return x;
  }
  diff = y-x;
  if (diff< -log(-LZERO) ){
    return  (x<LSMALL)?LZERO:x;
  }else {
    z = exp(diff);
    return x+log(1.0-z);
  }
}

/**
   @brief calculate the log factorial log(n!) = log(n) + ... + log1 . 
 */
double logFactorial( unsigned int n )
{
  unsigned int i;
  double sum = 0.0;

  if( n==0 )
    return 0.0; /* log1 */

  for(i=2;i<=n;i++){
    sum += log((double)i);
  }

  return sum;
}

/**
   @brief calculate Taylor series like
          y =  sum_{n=0}^{N}  (x-a)^n * g_n(a) / n!.
          In order to avoid the floating error, this will be done with expansion series of Meijer G-function in the log-domain. 
          Accordingly, this code looks like complicated becauce I ensure that the argument of log() is positive.
 */
double logTaylorSeries( double x, double *pCoeff, int N, float a )
{
  //int    idx = indexOfCoeffArray( x );
  //double *pCoeff = &s_dg2[idx][0];
  //float        a = s_argdg2[idx];
  float        c = x - a;
  bool isNegative;
  double swG,swC,ltmp;
  double lsumPo = LZERO;
  double lsumNe = LZERO;
  double ly ;

  if( c == 0 ){
    if( pCoeff[0] > 0 )
      return( log(pCoeff[0]) );
    else{
      fprintf(stderr,"This is not pdf because the value becomes negative.\n");
      return( LZERO );
    }
  }
  for(int n=0;n<N;n++){
    // check whether this term is negative or not
    if( n%2 != 0 ){
      if( pCoeff[n] > 0 && c < 0 ){
	isNegative = true;
	swG = pCoeff[n];
	swC = -1 * c;
      }
      else if ( pCoeff[n] < 0 && c > 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
	swC = c;
      }
      else{
	isNegative = false;
	swG = (pCoeff[n]>=0)? pCoeff[n]:-pCoeff[n];
	swC = (c>=0)? c:-c;
      }
    }
    else{/* (x-a)^n > 0 */
      if( pCoeff[n] < 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
      }
      else{
	isNegative = false;
	swG = pCoeff[n];
      }
      swC = (c>0)? c:-c;
    }
    
    ltmp = log( swG ) + n * log( swC ) - logFactorial( (unsigned int)n );
    if( isNegative==false ){
      if( n > 0 )
	lsumPo = LogAdd(lsumPo,ltmp);
      else
	lsumPo = ltmp;
    }
    else{
      if( n > 0 )
	lsumNe = LogAdd(lsumNe,ltmp);
      else
	lsumNe = ltmp;
    }
    
  }
  
  if( lsumPo > LZERO && lsumNe > LZERO ){
    ly = LogSub( lsumPo, lsumNe );
  }
  else if( lsumNe <= LZERO ){
    ly = lsumPo;
  }
  else {
    fprintf(stderr,"This is not pdf because the value becomes negative.\n");
    ly = LZERO;
  }

  return ly;
}

/**
   @brief calculate the Taylor series like
          y =  sum_{n=0}^{N}  (x-a)^n * g_n(a) / n!.
          In order to avoid the floating error, this will be done in log-domain. 
          Accordingly, this code looks like complicated becauce I ensure that the argument of log() is positive.
	  
   @param
   @param
   @param
   @param 
 */
double taylorSeriesWithLC( double x, double *pCoeff, int N, float a )
{
  //int    idx = indexOfCoeffArray( x );
  //double *pCoeff = &s_dg2[idx][0];
  //float        a = s_argdg2[idx];
  float        c = x - a;
  float        swC = (c>0)? c:-c;
  bool isNegative;
  double swG,ltmp;
  double lsumPo = LZERO;
  double lsumNe = LZERO;
  double ly, y;
      
  if( swC < 10e-30 ){
    return( pCoeff[0] );
  }
  for(int n=0;n<N;n++){
    // check whether this term is negative or not
    if( n%2 != 0 ){
      if( pCoeff[n] > 0 && c < 0 ){
	isNegative = true;
	swG = pCoeff[n];
      }
      else if ( pCoeff[n] < 0 && c > 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
      }
      else{
	isNegative = false;
	swG = (pCoeff[n]>=0)? pCoeff[n]:-pCoeff[n];
      }
    }
    else{/* (x-a)^n > 0 */
      if( pCoeff[n] < 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
      }
      else{
	isNegative = false;
	swG = pCoeff[n];
      }
    }

    if( swG > 10e-30 ){
      ltmp = log( swG ) + n * log( swC ) - logFactorial( (unsigned int)n );
    }
    else
      ltmp = LZERO;
    
    if( isNegative==false ){
      if( n > 0 )
	lsumPo = LogAdd(lsumPo,ltmp);
      else
	lsumPo = ltmp;
    }
    else{
      if( n > 0 )
	lsumNe = LogAdd(lsumNe,ltmp);
      else
	lsumNe = ltmp;
    }
  }
  
  if( lsumPo > LZERO || lsumNe > LZERO ){
    if( lsumPo > lsumNe ){
      ly = LogSub( lsumPo, lsumNe );
      if( ly > MINEARG && ly < MAXEARG )
	y = exp( ly );
      else if( ly <= MINEARG )
	y = 0;
      else
	y = DBL_MAX;
    }
    else{
      ly = LogSub( lsumNe, lsumPo );
      if( ly > MINEARG && ly < MAXEARG )
	y = - exp( ly );
      else if( ly <= MINEARG )
	y = 0;
      else
	y = -1 * DBL_MAX;
    }
  }
  else{
    ly = LZERO;
    y = 0;
  }

  return y;
}

/**
   @brief calculate the derivative of the Taylor series as
          y(x) =  sum_{n=0}^{N}  (x-a)^n * g_n(a) / n!.
          In order to avoid the floating error, this will be done in log-domain. 
          Accordingly, this code looks like complicated becauce I ensure that the argument of log() is positive.

   @note In the case of that x is a variable which depends on another variable like x=f(x'), you MUST multiply the derivativeof f(x') with the returned value.
   @param double x
   @param double *pCoeff[in]
   @param int N[in] the number of coefficients
   @param float a[in]
 */
double derivative1TaylorSeries( double x, double *pCoeff, int N, float a )
{
  //int    idx = indexOfCoeffArray( x );
  //double *pCoeff = &s_dg2[idx][0];
  //float        a = s_argdg2[idx];
  float        c = x - a;
  float        swC = (c>0)? c:-c;
  bool isNegative;
  double swG,ltmp;
  double lsumPo = LZERO;
  double lsumNe = LZERO;
  double ly, y;

  if( swC < 10e-30 ){
    return( pCoeff[0] );
  }
  for(int n=1;n<N;n++){
    // check whether this term is negative or not
    if( n%2 != 0 ){
      if( pCoeff[n] > 0 && c < 0 ){
	isNegative = true;
	swG = pCoeff[n];
      }
      else if ( pCoeff[n] < 0 && c > 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
      }
      else{
	isNegative = false;
	swG = (pCoeff[n]>=0)? pCoeff[n]:-pCoeff[n];
      }
    }
    else{/* (x-a)^n > 0 */
      if( pCoeff[n] < 0 ){
	isNegative = true;
	swG = -1 * pCoeff[n];
      }
      else{
	isNegative = false;
	swG = pCoeff[n];
      }
    }
    
    if( swG > 10e-30 )
      if( n > 1 )
	ltmp = log( swG ) + ( n - 1 ) * log( swC ) - logFactorial( (unsigned int)n-1 );
      else
	ltmp = log( swG );
    else
      ltmp = LZERO;

    if( isNegative==false ){
      if( n > 0 )
	lsumPo = LogAdd(lsumPo,ltmp);
      else
	lsumPo = ltmp;
    }
    else{
      if( n > 0 )
	lsumNe = LogAdd(lsumNe,ltmp);
      else
	lsumNe = ltmp;
    }
  }
  
  if( lsumPo > LZERO || lsumNe > LZERO ){
    if( lsumPo > lsumNe ){
      ly = LogSub( lsumPo, lsumNe );
      if( ly > MINEARG && ly < MAXEARG )
	y = exp( ly );
      else if( ly <= MINEARG )
	y = 0;
      else{
	fprintf(stderr,"The gradient is too big\n");
	y = DBL_MAX;
      }
    }
    else{
      ly = LogSub( lsumNe, lsumPo );
      if( ly > MINEARG && ly < MAXEARG )
	y = - exp( ly );
      else if( ly <= MINEARG )
	y = 0;
      else{
	fprintf(stderr,"The gradient is too big\n");
	y = -1 * DBL_MAX;
      }
    }
  }
  else {
    ly = LZERO;
    y = 0;
  }

  return y;
}

