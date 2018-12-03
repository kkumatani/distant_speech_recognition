/*
 * @file mcc_localizer.cc
 * @brief grid search based localization
 * @author Kenichi Kumatani
 */

#include "mcc_localizer.h"
#include <math.h>
#include <gsl/gsl_blas.h>
#include "localization.h"

#define SSPEED 343740.0
//double  sspeed = 343740.0;
#define TPI 6.28318530717958647692 /* 2 pi */

SearchGridBuilder::SearchGridBuilder( int nChan, bool isFarField, unsigned int samplingFreq ):
  _isFarField(isFarField),_samplingFreq(samplingFreq),_maxTimeDelay(-1)
{
  _mpos = gsl_matrix_calloc( nChan, 3 );
  _hypopos = gsl_vector_calloc( 3 );
  _delays  = gsl_vector_calloc( nChan );
}

SearchGridBuilder::~SearchGridBuilder()
{
  gsl_matrix_free( (gsl_matrix*)_mpos );
  gsl_vector_free( (gsl_vector *)_hypopos );
  gsl_vector_free( (gsl_vector *)_delays );
}

void SearchGridBuilder::reset()
{
  gsl_vector_set(_hypopos,0,0.0);
  gsl_vector_set(_hypopos,1,0.0);
  gsl_vector_set(_hypopos,2,0.0);
}

const gsl_vector *SearchGridBuilder::nextSearchGridNF()
{
  fprintf(stderr,"need to be implemented\n");
  return NULL;
}

SGB4LinearArray::SGB4LinearArray( int nChan, bool isFarField, unsigned int samplingFreq ):
  SearchGridBuilder( nChan, isFarField, samplingFreq )
{
}

/**
   @brief set the geometry of the linear array with equsi-spaced sensors.
   @param float distance[in] the distance between microphones
   @note unit is milli-meter.
 */
void SGB4LinearArray::setDistanceBtwMicrophones( float distance )
{
  size_t nMic = _mpos->size1;

  for(size_t micX=0;micX<nMic;micX++){
    gsl_matrix_set( _mpos, micX, 0, 0.0 );
    gsl_matrix_set( _mpos, micX, 1, micX * distance );
    gsl_matrix_set( _mpos, micX, 2, 0.0 );
  }

  _constV = 0.99 * SSPEED / ( (nMic-1) * distance * _samplingFreq );
  _maxTimeDelay = (nMic-1) *distance / SSPEED;
} 

void SGB4LinearArray::setPositionsOfMicrophones( const gsl_matrix* mpos )
{
  size_t nMic = _mpos->size1;
  if( nMic != mpos->size1 ){
    throw jdimension_error("The size of the matrix for the geometry of the array should be %lu x %lu\n", nMic, _mpos->size2 );
  }
  
  float maxDist=-1, pos0[3];
  for(size_t micX=1;micX<nMic;micX++){
    double x = gsl_matrix_get( mpos, micX, 0 );
    double y = gsl_matrix_get( mpos, micX, 1 );
    double z = gsl_matrix_get( mpos, micX, 2 );
    gsl_matrix_set( _mpos, micX, 0, x );
    gsl_matrix_set( _mpos, micX, 1, y );
    gsl_matrix_set( _mpos, micX, 2, z );
    if( micX > 0 ){
      float dx = pos0[0] - (float)x;
      float dy = pos0[1] - (float)y;
      float dz = pos0[2] - (float)z;
      float dist = sqrtf(  dx * dx + dy * dy + dz * dz );
      if( dist > maxDist )
	maxDist = dist;
    }
    else{
      pos0[0] = (float)x;
      pos0[1] = (float)y;
      pos0[2] = (float)z;
    }
  }

  _constV =  0.99 * SSPEED / ( maxDist * _samplingFreq );
  _maxTimeDelay = maxDist / SSPEED;
}

bool SGB4LinearArray::nextSearchGrid()
{
  const gsl_vector *hypo;

  if( true==_isFarField ){
    hypo = nextSearchGridFF();
  }
  else{
    hypo = nextSearchGridNF();
  }
  
  if( NULL == hypo )
    return false;

  return true;
}

const gsl_vector *SGB4LinearArray::nextSearchGridFF()
{  
  float azimuth = (float)gsl_vector_get(_hypopos,1);
  float oldSin = sinf( azimuth );
  float newAzimuth;
  float newSin;
  
  /* search a source position from 0 to pi/2 and 3pi/2 to 2pi. */
  if( azimuth < M_PI_2 ){/* [0,pi/2] */
    newSin = oldSin + _constV;
    if( newSin >= 1 ){ 
      newAzimuth = M_PI_2;
    }
    else
      newAzimuth = asinf( newSin );
  }
  else if( azimuth < ( 3 * M_PI_2 ) ){/* skip the search region [pi/2,3pi/2]*/
    newAzimuth = 3 * M_PI_2;
  }
  else{/* [3*pi/2,2*pi] */
    newSin = oldSin + _constV;
    if( ( newSin + _constV/2.0 )>= 0 )
      return NULL;
    newAzimuth = TPI + asinf( newSin );
  }
  /*printf("%f %f %f\n",newSin, newAzimuth);*/
  gsl_vector_set(_hypopos,1,newAzimuth);
    
  return (const gsl_vector *)_hypopos;
}

const gsl_vector *SGB4LinearArray::getTimeDelays()
{
  calcDelaysOfLinearMicrophoneArray( gsl_vector_get(_hypopos,1),
				     (const gsl_matrix*)_mpos, 
				     _delays);
#if 0
  for (size_t i=0; i<_delays->size; i++){
    gsl_vector_set(_delays, i, - gsl_vector_get(_delays, i ) );
  }#
#endif
  return (const gsl_vector *)_delays;
}

SGB4CircularArray::SGB4CircularArray( int nChan, bool isFarField, unsigned int samplingFreq ):
  SearchGridBuilder( nChan, isFarField, samplingFreq )
{
}

/**
   @brief set the geometry of the circular array with equsi-spaced sensors.
   @param float radius[in] the radius of the array
   @param float height[in]
   @note unit is milli-meter.
 */
void SGB4CircularArray::setRadius( float radius, float height )
{
  size_t nMic = _mpos->size1;
  float bias = TPI / (float)nMic;

  for(size_t micX=0;micX<nMic;micX++){
    gsl_matrix_set( _mpos, micX, 0, radius * cosf( micX * bias ) );
    gsl_matrix_set( _mpos, micX, 1, radius * sinf( micX * bias ) );
    gsl_matrix_set( _mpos, micX, 2, height );
  }
  
  _constV = SSPEED / ( 2 *radius * _samplingFreq );
  _maxTimeDelay = 2 *radius / SSPEED;
}


bool SGB4CircularArray::nextSearchGrid()
{
  const gsl_vector *hypo;

  if( true==_isFarField ){
    hypo = nextSearchGridFF();
  }
  else{
    hypo = nextSearchGridNF();
  }
  
  if( NULL == hypo )
    return false;

  return true;
}

const gsl_vector *SGB4CircularArray::nextSearchGridFF()
{
  float azimuth = (float)gsl_vector_get(_hypopos,1);
  float polarAngle = (float)gsl_vector_get(_hypopos,2);
  float newAzimuth, newPolarAngle;
  float val1, val2;

  if( azimuth >= TPI && polarAngle >= M_PI )
    return NULL;// the search should be finished
  // calculate the delta of the polar angle
  if( ( azimuth >= M_PI_4 && azimuth < (3*M_PI_4) ) ||
      ( azimuth >= (5*M_PI_4) && azimuth < (7*M_PI_4) ) ){
    val1 = _constV / sinf(azimuth);
  }
  else{
    val1 = _constV / cosf(azimuth);
  }
  newPolarAngle = (val1<1)? asinf(val1):M_PI_2;
  if( ( newPolarAngle + polarAngle ) < M_PI ){
    // increase the polar angle value while keeping the same azimuth for the next search
    newPolarAngle += polarAngle;
    newAzimuth = azimuth;
  }
  else{
    val2 = _constV / sinf(newPolarAngle);
    newAzimuth = (val2<1)? acos( _constV/val2 ):M_PI;
    newAzimuth += azimuth;
  }
  
  gsl_vector_set(_hypopos,1,newAzimuth);
  gsl_vector_set(_hypopos,2,newPolarAngle);
  if( newAzimuth >= TPI && newPolarAngle >= M_PI )
    return NULL;// the search should be finished
    
  return (const gsl_vector *)_hypopos;
}

const gsl_vector *SGB4CircularArray::getTimeDelays()
{
  
  calcDelaysOfCircularMicrophoneArray( gsl_vector_get(_hypopos,1), 
				       gsl_vector_get(_hypopos,2),
				       (const gsl_matrix*)_mpos, 
				       _delays);
#if 0
  for (size_t i=0; i<_delays->size; i++){
    gsl_vector_set(_delays, i, - gsl_vector_get(_delays, i ) );
  }
#endif
  return (const gsl_vector *)_delays;
}

MCCLocalizer::MCCLocalizer(SearchGridBuilderPtr &sgbPtr,  size_t maxSource, const String&nm )
  :VectorFeatureStream( 3, nm ), _sgbPtr(sgbPtr)
{
  size_t chanN = sgbPtr->chanN();

  _tau = new int[chanN];
  _blockList.resize( chanN );
  _R = gsl_matrix_alloc( chanN, chanN );
  _Rcopy = gsl_matrix_alloc( chanN, chanN );
  _x = gsl_vector_alloc( chanN );
  _workspace = (void *)gsl_eigen_symm_alloc(chanN);
  _eigenvalues = gsl_vector_alloc( chanN );

  unsigned int samplingFrequency = _sgbPtr->samplingFrequency();
  size_t maxSampleDelay = (size_t)( samplingFrequency * _sgbPtr->maxTimeDelay() );
  _shPtr = new SampleHolder( chanN, maxSampleDelay );

  _sourceCandidates.resize( maxSource );
  for(size_t i=0;i<maxSource;i++){
    _sourceCandidates[i] = new SourceCandidate( chanN );
  }
}

MCCLocalizer::~MCCLocalizer()
{
  delete [] _tau;
  _blockList.clear();
  gsl_matrix_free( _R );
  gsl_matrix_free( _Rcopy );
  gsl_vector_free( _x );
  gsl_eigen_symm_free((gsl_eigen_symm_workspace *)_workspace);
  gsl_vector_free( _eigenvalues );

  for(size_t i=0;i<_sourceCandidates.size();i++){
    delete _sourceCandidates[i];
  }
  _sourceCandidates.clear();
}

void MCCLocalizer::setChannel(VectorFloatFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

/*
  @brief get a block of data at the current frame.
*/
bool MCCLocalizer::setIncomingData( int frame_no )
{
  bool endOfSample = false;
  unsigned chanX = 0;

  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++ ) {
    _blockList[chanX++] = (*itr)->next(frame_no);    
    //if( true==(*itr)->isEnd() ) endOfSample = true;
  }
  
  return endOfSample;
}

/*
  @brief calculate the covariance matrix from a block of data.
*/
gsl_matrix *MCCLocalizer::calcCovarianceMatrix()
{ 
  const gsl_vector *delays = _sgbPtr->getTimeDelays();
  unsigned int samplingFrequency = _sgbPtr->samplingFrequency();
  size_t maxSampleDelay = (size_t)( samplingFrequency * _sgbPtr->maxTimeDelay() );
  size_t chanN = _sgbPtr->chanN();

  if( _blockList[0]->size < 2 * maxSampleDelay ){
    throw j_error("Data samples are insufficient\n");
  }

  for(size_t chanX=0;chanX<chanN;chanX++){
    float tau_l = samplingFrequency * gsl_vector_get(delays,chanX);
    _tau[chanX] = (int)( tau_l );
    _shPtr->setSamples( chanX, (gsl_vector_float *)_blockList[chanX] );
  }

  gsl_matrix_set_zero( _R );
  int nSamples = 0;
  int frameS = -_shPtr->nFilled() + (int)maxSampleDelay;
  int frameN = (int)_blockList[0]->size - (int)maxSampleDelay; 
  for (int frame_no=frameS;frame_no<frameN;frame_no++,nSamples++){
    for(size_t chanX=0;chanX<chanN;chanX++){
      int frameY = frame_no + _tau[chanX];
      float val;
      if( frameY >= 0 ){
	val = gsl_vector_float_get( (gsl_vector_float *)_blockList[chanX], frameY );
      }
      else{ // get samples from the previous data block
	val = _shPtr->getSample( chanX, frameY );
      }
      gsl_vector_set( _x, chanX, (double)val );
    }
    gsl_blas_dsyr(CblasLower,1.0,_x,_R);
  }
  gsl_matrix_scale( _R, 1.0 / (double)nSamples );

  return _R;
}

void MCCLocalizer::doEigenValueDecomposition()
{
  gsl_matrix_memcpy(_Rcopy, _R);
  gsl_eigen_symm(_Rcopy, _eigenvalues, (gsl_eigen_symm_workspace *)_workspace );
}

double MCCLocalizer::calcObjectiveFunction( bool normalizeVariance )
{
  double ldetR = 0;
  double lnrm = 0;
  double cf;
  int numMinus = 0;
  bool isThereZeroEigenVal = false;
  // make sure that the eigenvalues are positive,
  // calculate the determinant of the covariance matrix and 
  // calculate the power of the sample.
  for(size_t i=0;i<_eigenvalues->size;i++){
    double eval = gsl_vector_get(_eigenvalues,i);
    if( eval < 0 ){
      numMinus += 1;
      //fprintf(stderr,"eigenvlaue is negative %e\n",eval);
      eval = -eval;
      gsl_vector_set(_eigenvalues,i,eval);
    }
    else if( eval <= 0 ){
      fprintf(stderr,"eigenvlaue is zero %e\n",eval);
      isThereZeroEigenVal = true;
      break;
    }

    ldetR += log(eval);
    double lr_ii = log( gsl_matrix_get( _R, i , i ) );
    lnrm += lr_ii; // the power of the sample
  }  
  if( isThereZeroEigenVal == true )
    return 0.0;

  _detR = exp( ldetR );
  if( ( numMinus % 2 ) != 0 )
    _detR = - _detR;
    
  if( normalizeVariance == false ) lnrm = 0.0;
#ifdef _NO_LOG_
  cf = exp( ldetR - lnrm );
#else
  cf = ldetR - lnrm;
#endif
  
  return cf;
}

gsl_vector* MCCLocalizer::search( int frame_no )
{
  size_t maxSrc = _sourceCandidates.size();
  _sgbPtr->reset();
#define TAKE_MAX_IN_ONE_UTT
#ifdef  TAKE_MAX_IN_ONE_UTT
  for(size_t i=0;i<maxSrc;i++){_sourceCandidates[i]->setConstV();}
#endif  

  while(1){
    double costV;
    const gsl_vector *position = _sgbPtr->getSearchPosition();

    calcCovarianceMatrix();
    doEigenValueDecomposition();
    costV = calcObjectiveFunction();
    //fprintf(stderr,"POS %e %e\n",gsl_vector_get(position,1),costV);

    if( costV <  _sourceCandidates[maxSrc-1]->_costV ){// keep the N-best candidates
      SourceCandidate *hook = _sourceCandidates[maxSrc-1];// take this pointer since it will be removed from the stack.
      for(size_t i=0;i<maxSrc;i++){
	if( costV < _sourceCandidates[i]->_costV ){
	  for(size_t j=maxSrc-1;j>i;j--){
	    _sourceCandidates[j] = _sourceCandidates[j-1];
	  }
	  hook->setSourceInfo( _tau, position, costV, _eigenvalues );
	  _sourceCandidates[i] = hook;
	  break;
	}
      }
    }
    if( false== _sgbPtr->nextSearchGrid() )
      break;
  }
  //_minMCCC = 1 - minCostV;
  gsl_vector_memcpy(vector_, _sourceCandidates[0]->_position);

  return vector_;
}

const gsl_vector* MCCLocalizer::next(int frame_no )
{
  setIncomingData( frame_no );
  vector_ = search( frame_no );
  return vector_;
}

void MCCLocalizer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
   (*itr)->reset();
  VectorFeatureStream::reset(); 
  _sgbPtr->reset();
  //_endOfSample = false;
}

MCCCalculator::MCCCalculator( SearchGridBuilderPtr &sgbPtr, bool normalizeVariance, const String&nm )
 :MCCLocalizer( sgbPtr,  1, nm ), 
  _delays(NULL),
  _normalizeVariance(normalizeVariance)
{
}

MCCCalculator::~MCCCalculator()
{
  if( NULL != _delays ){
    gsl_vector_free( _delays );
  }
}

void MCCCalculator::setTimeDelays( gsl_vector *delays )
{
  if( NULL != _delays ){
    gsl_vector_free( _delays );
  }
  _delays = gsl_vector_alloc( delays->size );
  gsl_vector_memcpy( _delays, delays );
}

/*
  @brief calculate the covariance matrix from a block of data.
*/
gsl_matrix *MCCCalculator::calcCovarianceMatrix( gsl_vector *delays )
{ 
  unsigned int samplingFrequency = _sgbPtr->samplingFrequency();
  size_t maxSampleDelay = (size_t)( samplingFrequency * _sgbPtr->maxTimeDelay() );
  size_t chanN = _sgbPtr->chanN();

  if( _blockList[0]->size < 2 * maxSampleDelay ){
    throw j_error("Data samples are insufficient\n");
  }

  for(size_t chanX=0;chanX<chanN;chanX++){
    float tau_l = samplingFrequency * gsl_vector_get(delays,chanX);
    _tau[chanX] = (int)( tau_l );
    _shPtr->setSamples( chanX, (gsl_vector_float *)_blockList[chanX] );
  }

  gsl_matrix_set_zero( _R );
  int nSamples = 0;
  int frameS = -_shPtr->nFilled() + (int)maxSampleDelay;
  int frameN = (int)_blockList[0]->size - (int)maxSampleDelay; 
  for (int frame_no=frameS;frame_no<frameN;frame_no++,nSamples++){
    for(size_t chanX=0;chanX<chanN;chanX++){
      int frameY = frame_no + _tau[chanX];
      float val;

      if( frameY >= 0 ){
	val = gsl_vector_float_get( (gsl_vector_float *)_blockList[chanX], frameY );
      }
      else{ // get samples from the previous data block
	val = _shPtr->getSample( chanX, frameY );
      }
      gsl_vector_set( _x, chanX, (double)val );
      //fprintf(stderr,"%e ", val);
    }
    //fprintf(stderr,"\n");
    gsl_blas_dsyr(CblasLower,1.0,_x,_R);
  }
  gsl_matrix_scale( _R, 1.0 / (double)nSamples );

  return _R;
}

const gsl_vector* MCCCalculator::next(int frame_no )
{
  double costV;
  //_sgbPtr->reset();
  //const gsl_vector *position = _sgbPtr->getSearchPosition();
  if( NULL == _delays ){
    throw j_error("set time delays with setTimeDelays()\n");
  }

  setIncomingData( frame_no );
  calcCovarianceMatrix( _delays );
  doEigenValueDecomposition();
  costV = calcObjectiveFunction( _normalizeVariance );
  gsl_vector_set(vector_, 0, costV );
  _sourceCandidates[0]->setSourceInfo( _tau, vector_, costV, _eigenvalues );

  return vector_;
}

double MCCCalculator::getMCCC()
{
  double costV = gsl_vector_get(vector_,0);
#ifdef _NO_LOG_
  return ( 1 - costV );
#else
  return( 1.0 - exp(costV) );
#endif
}

double MCCCalculator::getCostV(){ 
  return gsl_vector_get(vector_, 0);
}

void MCCCalculator::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
   (*itr)->reset();
  VectorFeatureStream::reset(); 
  _sgbPtr->reset();
  //_endOfSample = false;
}


RMCCLocalizer::RMCCLocalizer( SearchGridBuilderPtr &sgbPtr, float lambda, size_t maxSource, const String&nm )
  :MCCLocalizer( sgbPtr, maxSource, nm )
{
  size_t chanN = sgbPtr->chanN();

  _invR = gsl_matrix_alloc( chanN, chanN );
  _kd = gsl_vector_alloc( chanN );

  //_old_x = gsl_vector_alloc( chanN );
  _old_R = gsl_matrix_alloc( chanN, chanN );
  _old_invR = gsl_matrix_alloc( chanN, chanN );
  _old_kd = gsl_vector_alloc( chanN );
}

RMCCLocalizer::~RMCCLocalizer()
{
  gsl_matrix_free( _invR );
  gsl_vector_free( _kd );

  //  gsl_vector_free( _old_x );
  gsl_matrix_free( _old_R );
  gsl_matrix_free( _old_invR );
  gsl_vector_free( _old_kd );
}

void RMCCLocalizer::calcInverseMatrix()
{
}


void RMCCLocalizer::updateParameters()
{
}

const gsl_vector* RMCCLocalizer::next(int frame_no )
{
  return vector_;
}

void RMCCLocalizer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
   (*itr)->reset();
  VectorFeatureStream::reset(); 
  _sgbPtr->reset();
  //_endOfSample = false;
}
