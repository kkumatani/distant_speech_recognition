/*
 * @file lpc.cc
 * @brief Speech recognition front end.
 * @author John McDonough, Matthias Woelfel, Kenichi Kumatani
 */

#include "common/jexception.h"
#include "feature/lpc.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_real.h>


// ----- methods for class `BaseFeature' -----
//
BaseFeature::BaseFeature(unsigned order, unsigned dim)
  : _order(order), _dim(dim), _log2Length((unsigned) ceil(log((double)_dim) / log(2.0))),
    _npoints(0x1 << _log2Length), _npoints2(_npoints / 2),
    _temp(new double[_npoints])
{
  printf("Initializing 'BaseFeature'.\n");  fflush(stdout);
}

BaseFeature::~BaseFeature()
{
  delete[] _temp;
}

void BaseFeature::fftPower(float* power)
{
  for (unsigned i = 0; i < _dim; i++)
    _temp[i] = (double) power[i];

  for (unsigned i = _dim; i < _npoints; i++)
    _temp[i] = 0.0;

  gsl_fft_real_radix2_transform(_temp, /*stride=*/ 1, _npoints);

  // convert the complex data to power
  power[0]         = _temp[0] * _temp[0];
  power[_npoints2] = _temp[_npoints2] * _temp[_npoints2];

  for (unsigned i = 1; i < _npoints2; i++)
    power[i] = _temp[i] * _temp[i] + _temp[_npoints - i] * _temp[_npoints - i];
}


// ----- methods for class `WarpFeature' -----
//
WarpFeature::WarpFeature(unsigned order, unsigned dim)
  : BaseFeature(order, dim),
    _K(new float[_order + 1]), _R(new float[order + 1]),
    _WX(new float[_dim + 1]), _WXTEMP(new float[dim+1]),
    _tmpA(gsl_matrix_float_calloc(_order + 1, _order + 1))
{
  printf("Initializing 'WarpFeature'.\n");  fflush(stdout);
}

WarpFeature::~WarpFeature()
{
  delete[] _K;  delete[] _R;
  delete[] _WX; delete[] _WXTEMP;
  gsl_matrix_float_free(_tmpA);
}

void WarpFeature::autoCorrelation(const float* X, float* LP, float* E, float warp)
{
  // calculate warped LP
  float sum = 0.0;
  for (unsigned i = 0; i < _dim; i++)
    sum += X[i] * X[i];
  _R[0] = sum;

  for (unsigned j = 0; j < _dim; j++)
    _WX[j] = X[j];

  for (unsigned i = 1; i <= _order; i++) {
    // filter
    for (unsigned j = 0; j < _dim; j++)
      _WXTEMP[j] = _WX[j];

    _WX[0] = -warp * _WXTEMP[0];
    for (unsigned j = 1; j < _dim; j++)
      _WX[j] = warp * (_WX[j-1] - _WXTEMP[j]) + _WXTEMP[j-1];

    sum = 0.0;
    for (unsigned j = 0; j < _dim; j++)
      sum += X[j] * _WX[j];
    _R[i] = sum;
  }

  E[0] = _R[0];
  gsl_matrix_float_set(_tmpA, 1, 0, 1.0);
  for (unsigned i = 1; i <= _order; i++) {
    // eq. 8.68 (eq. 3.51a)
    _K[i] = _R[i];
    for (unsigned j = 1; j < i; j++)
      _K[i] -= gsl_matrix_float_get(_tmpA, j, i-1) * _R[i-j];

    if (E[i-1] != 0) {
      _K[i] /= E[i-1];
    } else {
      _K[i] = 1000000000;
      /* fprintf(stderr,"WARNING: in LPautocorrelation division by zero\n"); */
    }

    // eq. 8.69 (eq. 3.51b)
    gsl_matrix_float_set(_tmpA, i, i, _K[i]);

    // eq. 8.70 (eq. 3.51c)
    for (unsigned j = 1; j <= i-1; j++) {
      double val = gsl_matrix_float_get(_tmpA, j, i-1) - _K[i] * gsl_matrix_float_get(_tmpA, i-j, i-1);
      gsl_matrix_float_set(_tmpA, j, i, val);
    }

    // eq. 8.71 (eq. 3,51d)
    E[i]=(1 - _K[i] * _K[i]) * E[i-1];
  }

  // important for second step
  LP[0]=1.0;
  for (unsigned i = 1; i <= _order; i++)
    LP[i] = -gsl_matrix_float_get(_tmpA, i, _order);
}


// ----- methods for class `BurgFeature' -----
//
BurgFeature::BurgFeature(unsigned order, unsigned dim)
  : BaseFeature(order, dim),
    _EF(new float[_dim]), _EB(new float[_dim]), _EFP(new float[_dim]), _EBP(new float[_dim]),
    _A_flip(new float[_order+1]), _K(new float[_order+1])
{
}

BurgFeature::~BurgFeature()
{
  delete[] _EF;     delete[] _EB;
  delete[] _EFP;    delete[] _EBP;
  delete[] _A_flip; delete[] _K;
}

void BurgFeature::autoCorrelation(const float *X, float *A, float *E, float warp)
{
  E[0]=0.0;
  for (unsigned i = 0; i < _dim; i++)
    E[0] += X[i]*X[i];

  for (unsigned i = 0; i <= _order; i++) {
    _A_flip[i] = 0.0;
    A[i] = 0.0;
  }

  for (unsigned i = 0; i < _dim; i++) {
    _EF[i] = X[i];
    _EB[i] = X[i];
  }

  for (unsigned i = 0; i < _order; i++) {

    // Calculate the next order reflection (parcor) coefficient

    for (unsigned j = 0; j < _dim-i-1; j++) {
      _EFP[j] = _EF[j+1];
      _EBP[j] = _EB[j];
    }

    double num = 0.0;
    double den = 0.0;
    for (unsigned j = 0; j < _dim-i-1; j++) {
      num -= 2 * _EBP[j] * _EFP[j];
      den += _EFP[j] * _EFP[j] + _EBP[j] * _EBP[j];
    }

    _K[i] = (float) num/den;

    // Update the forward and backward prediction errors

    for (unsigned j = 0; j < _dim-i-1; j++) {
      _EF[j] = _EFP[j] + _K[i] * _EBP[j];
      _EB[j] = _EBP[j] + _K[i] * _EFP[j];
    }

    // Update the AR coeff.

    A[0] = 1.0;
    for (unsigned j = 0; j <= i+1; j++)
      _A_flip[j] = A[i-j+1];
    for (unsigned j = 1; j <= i+1; j++)
      A[j] += _K[i] * _A_flip[j];
  }
}


// ----- methods for class `WarpedTwiceFeature' -----
//
WarpedTwiceFeature::WarpedTwiceFeature(unsigned order, unsigned dim )
  : BaseFeature(order, dim),
    _K(new float[_order + 1]), _R(new float[order + 1]),
    _WX(new float[_dim + 1]), _WXTEMP(new float[dim+1])
{
  printf("Initializing 'WarpedTwiceFeature'.\n");  fflush(stdout);

  float *fptr = new float[(order+1)*(order+1)];
  _tmpA = (float **)malloc((order+1)*sizeof(float *));
  if( NULL==_tmpA ){
    jallocation_error("WarpedTwiceFeature: could not allocate image\n");
  }
  for(unsigned i=0;i<=_order;i++)
    _tmpA[i] = &fptr[i*(order+1)];
}

WarpedTwiceFeature::~WarpedTwiceFeature()
{
  delete[] _K;  delete[] _R;
  delete[] _WX; delete[] _WXTEMP;
  delete[] _tmpA[0];
  free(_tmpA);
}

/**
   @brief Compensate for warp like Nakamoto et al. INTERSPEECH 2004
   @note New warped-LPC implementation
         Calculate LP through autocorrelation based on
	 Durbin's Recursion; see  Rabiner pp. 411
	 (or Applications of DSP editor Oppenheimer pp. 152)    
	 Results in LP[1..order]=-A[1..order][order]
*/
void WarpedTwiceFeature::autoCorrelation(const float *X, float *LP, float *E, float warp, float rewarp )
{
  float temp;
  float g1, gj, sum;
  float a0, a1;
  
  /* calculate warped LP */
  temp = 0.0;
  for (unsigned j=0;j<_dim;j++)
    temp += X[j]*X[j];
  _R[0] = temp;
  
  for (unsigned j=0;j<_dim;j++)
    _WX[j] = X[j];
  
  /*  for compensation we need R[order+1], otherwise not needed */
  for (unsigned i=1;i<=_order+1;i++) {
    /* filter */
    for (unsigned j=0;j<_dim;j++) 
      _WXTEMP[j] = _WX[j];
    
    _WX[0] = - warp * _WXTEMP[0];
    for (unsigned j=1;j<_dim;j++) 
      _WX[j] = warp * ( _WX[j-1] - _WXTEMP[j]) + _WXTEMP[j-1];
    
    temp=0.0;
    for (unsigned j=0;j<_dim;j++)
      temp += X[j] * _WX[j];
    _R[i] = temp;
  }
  
  /* compensate for the warp value */
  a0 = (warp+rewarp)/(1+warp*rewarp);
  gj = 1.0 - a0 * a0;
  a1 = a0 / gj;
  a0 = ( 1.0 + a0 * a0 ) / gj;
 
  g1 = _R[0];
  _R[0] = a0 * _R[0] + 2.0 * a1 * _R[1];
  
  for (unsigned i = 1; i <= _order; ++i) {
    sum = a0 * _R[i] + a1 * ( g1 + _R[i+1] );
    g1 = _R[i];
    _R[i] = sum;
  }
 
  E[0] = _R[0];
  _tmpA[1][0] = 1;
  
  for (unsigned i=1;i<=_order;i++) {
    float **A = _tmpA;
    /* eq. 8.68 (eq. 3.51a) */
    _K[i] = _R[i];
    for (unsigned j=1;j<i;j++) 
      _K[i]-= A[j][i-1] * _R[i-j];
    
    if ( E[i-1] != 0 ) {
      _K[i] /= E[i-1];
    } else {
      _K[i] = 1000000000;
      /* fprintf(stderr,"WARNING: in LPautocorrelation division by zero\n"); */
    }
    
    /* eq. 8.69 (eq. 3.51b) */
    A[i][i] = _K[i];
    
             /* eq. 8.70 (eq. 3.51c) */
    for (unsigned j=1;j<=i-1;j++)
      A[j][i]= A[j][i-1] - _K[i] * A[i-j][i-1];
    
    /* eq. 8.71 (eq. 3,51d) */
    E[i] = ( 1-_K[i] * _K[i] ) * E[i-1];
  }
  
  /* important for second step */
  LP[0]=1;
  for (unsigned i=1;i<=_order;i++) {
    float **A=_tmpA;
    LP[i]=-A[i][_order];
  }  
}

// ----- methods for class `WarpedTwiceMVDRFeature' -----
//

/**
   @brief 

   @param const VectorFloatFeatureStreamPtr& src
   @param unsigned order
   @param unsigned correlate 
   @param float warp
   @param int warpFactorFixed
   @param float sensibility
*/
WarpedTwiceMVDRFeature::WarpedTwiceMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp, bool warpFactorFixed, float sensibility, const String& nm)
  : VectorFeatureStream(src->size()/2+1, nm),
    WarpedTwiceFeature(order, src->size()),
    _src(src),
    _correlate(correlate),
    _temp_order(2*order+1),
    _warp(warp),
    _tmpR(new float[order+1]),
    _tmpA(new float[order+1]),
    _tmpPC(new float[_temp_order]),
    _tmpPA(new float[WarpedTwiceFeature::_dim+1]),
    _E(new float[order+1]),
    _warpFactorFixed(warpFactorFixed),
    _sensibility(sensibility)
{
  if (_correlate < 10) _correlate = WarpedTwiceFeature::_dim;
  if (WarpedTwiceFeature::_order >= WarpedTwiceFeature::_dim/2+1)
    throw jparameter_error("Order (%d) and dimension (%d) do not match.", WarpedTwiceFeature::_order, WarpedTwiceFeature::_dim/2+1);
  
  if( _warpFactorFixed == true ){ //Case("fixwarp")
    float warp_var = _sensibility + _warp;
    _rewarp = ( warp_var - _warp) / (1 - warp_var * _warp );
  }
  
}

WarpedTwiceMVDRFeature::~WarpedTwiceMVDRFeature()
{
  delete[] _tmpR;  delete[] _tmpA;  delete[] _tmpPC;  delete[] _tmpPA;  delete[] _E;
}

/*
  @brief This function uses a long chain of first order
         allpass elements to warp a signal               
  @usage trans_longchain(signal,len,lam, tim, xm)   
  @param signal[in]: the input of the filter 
  @param tim[in] the length of the signal
  @param xm[in]: the ouput of the filter  
  @param len[in] the length of the produced warped signal ( len < length(signal) )
  @param lam[in] the warping parameter 
*/
void trans_longchain(float *signal, int len, float lam, float *xm, int tim)
{
  float x,tmpr;
  int w,e, ind = 0;
  
  for(w=0;w<tim;w++) {
    x=signal[ind++];
    for(e=0; e <len; e++) {
      /* The difference equation */
      tmpr=xm[e]+lam*xm[e+1]-lam*x;
      xm[e]=x;
      x=tmpr;
    }
  }
  return;
}
 

float R1R0( const float *X, int dim ) 
{
  int i,j;
  float temp;
  float R[2];
  
  for (i=0;i<=1;i++) {
    temp=0.0;
    for (j=0;j<dim-i;j++) { temp += X[j]*X[j+i]; }
    R[i]=temp;
  }
  
  temp = fabs( R[1]/R[0] );

  return temp;
}
 
const gsl_vector* WarpedTwiceMVDRFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = _src->next(frame_no_ + 1);
  increment_();

  const float* X  = block->data;
  float* A  = _tmpA;
  float* PC = _tmpPC;
  float* PA = _tmpPA;

  if(_warpFactorFixed==false){ // Case("varwarp")
    float warp_var = _sensibility * ( R1R0( X, _correlate ) - 0.5 ) + _warp;
    _rewarp = ( warp_var - _warp ) / ( 1 - warp_var * _warp );
  }

  WarpedTwiceFeature::autoCorrelation(X, _tmpA, _E, _warp, _rewarp );

  for(unsigned i=0;i<=WarpedTwiceFeature::_order;i++) {
    float temp = 0;

    for (unsigned ii=0;ii<=(WarpedTwiceFeature::_order-i);ii++)
      temp += (float)( WarpedTwiceFeature::_order + 1 - i -2 * ii ) * A[ii] * A[ii+i];
    if ( _E[0] > 0 ) {
      PC[WarpedTwiceFeature::_order+i] = -temp;
      /* instead of -temp/E[order] in the paper by B. Musicus IEEE 1985 */
    } else {
      PC[WarpedTwiceFeature::_order+i] = 10000000;
    }
  }

  for (unsigned i=1;i<=WarpedTwiceFeature::_order;i++) PC[WarpedTwiceFeature::_order-i] = PC[WarpedTwiceFeature::_order+i];

  for (unsigned i=0;i<=WarpedTwiceFeature::_dim;i++) PA[i] = 0.0;
  trans_longchain( PC, WarpedTwiceFeature::_dim, -_rewarp, PA, _temp_order );

  /* ------------------------------------------------------------------------
     Calculate MVDR Power (also called Capon Max. Likelihood Method)*/
  WarpedTwiceFeature::fftPower( PA );
  for (unsigned i=0;i<=WarpedTwiceFeature::_dim/2;i++) {
    double temp = sqrt(PA[i]);
    if (temp > 0) {
      temp = _E[0]/temp;
    } else {
      temp = 10000000;
    }
    gsl_vector_set(vector_, i, temp);
  }
  
  return vector_;
}


// ----- methods for class `SpectralSmoothing' -----
//
SpectralSmoothing::SpectralSmoothing(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom, const String& nm)
  : VectorFeatureStream(adjustTo->size(), nm), _adjustTo(adjustTo), _adjustFrom(adjustFrom), _R(new float[size()])
{
  if (adjustTo->size() != adjustFrom->size())
    throw jdimension_error("Feature sizes (%d vs. %d) do not match.", adjustTo->size(), adjustFrom->size());
}

SpectralSmoothing::~SpectralSmoothing()
{
  delete[] _R;
}

const gsl_vector* SpectralSmoothing::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* _maxSpec = _adjustTo->next(frame_no_ + 1);
  const gsl_vector* _maxFFT  = _adjustFrom->next(frame_no_ + 1);
  increment_();

  // corresponds to blur = 2
  _R[0] = 0.0;
  _R[1] = 0.0;
  for (unsigned i = 2; i < size()-2; i++) {
    _R[i] =  gsl_vector_get(_maxFFT, i-2) / 9.0
      + 2.0 * gsl_vector_get(_maxFFT, i-1) / 9.0
      + gsl_vector_get(_maxFFT, i) / 3.0
      + 2.0 * gsl_vector_get(_maxFFT, i+1) / 9.0
      + gsl_vector_get(_maxFFT, i+2) / 9.0;
  }
  _R[size()-2] = 0.0;
  _R[size()-1] = 0.0;

  float maxFFT  = 0.0;
  float maxSPEC = 0.0;
  for (unsigned i = 0; i < size(); i++)
    if (maxFFT < _R[i]) maxFFT = _R[i];
  for (unsigned i = 0; i < size(); i++) {
    float val = gsl_vector_get(_maxSpec, i);
    if (maxSPEC < val) maxSPEC = val;
  }

  float mult;
  if (maxSPEC < 0.01)
    mult = 100 * maxFFT;
  else
    mult = maxFFT / maxSPEC;

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_set(vector_, i, mult * gsl_vector_get(_maxSpec, i));

  return vector_;
}
