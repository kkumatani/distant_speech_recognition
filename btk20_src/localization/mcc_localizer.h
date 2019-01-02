/**
 * @file mcc_localizer.h
 * @brief grid search based localization
 * @author Kenichi Kumatani
 */

#ifndef MCC_LOCALIZER_H
#define MCC_LOCALIZER_H
#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "feature/feature.h"
//#include "modulated/modulated.h"

/**
   @brief  construct a search grid for source localization and return the time delay corresponding to each position on the grid.
   @note In order to do efficient grid-search for the source position which provides the maximum/minimum objective function, each cell size of a search grid has to be "reasonable".
   @usage
   After you construct an instance, you have to 
   1. set the geometry of the microphone array with setRadius(), setDistanceBtwMicrophones() or setPositionsOfMicrophones().
   2. obtain the time delays at the source position with getTimeDelays(), and
   3. go to the next search candidate on the grid with nextSearchGrid().
 */
class SearchGridBuilder {
 public:
  SearchGridBuilder( int nChan, bool isFarField, unsigned int samplingFreq=16000);
  virtual ~SearchGridBuilder();

  const gsl_vector *getSearchPosition(){return(const gsl_vector *)_hypopos;}
  float maxTimeDelay(){return _maxTimeDelay;}
  size_t chanN(){return(_mpos->size1);}
  unsigned int samplingFrequency(){return(_samplingFreq);}
  void reset();

  virtual const gsl_vector *getTimeDelays(){return NULL;}
  virtual bool nextSearchGrid(){return false;}

protected:
  virtual const gsl_vector *nextSearchGridFF(){return NULL;}
  const gsl_vector *nextSearchGridNF();

protected:
  const bool _isFarField; /* if _isFarField = true, the far-filed is assumed. Otherwise, the near-field is assumed. */
  unsigned int _samplingFreq;
  float _maxTimeDelay;
  gsl_matrix* _mpos;
  gsl_vector *_hypopos;
  gsl_vector *_delays;  
  float _constV;
};

typedef refcount_ptr<SearchGridBuilder> SearchGridBuilderPtr;

class SGB4LinearArray : public SearchGridBuilder{
public:
  SGB4LinearArray( int nChan, bool isFarField, unsigned int samplingFreq=16000);
  void setDistanceBtwMicrophones( float distance );
  void setPositionsOfMicrophones( const gsl_matrix* mpos );
  virtual const gsl_vector *getTimeDelays();
  virtual bool nextSearchGrid();

protected:
  const gsl_vector *nextSearchGridFF();
};

typedef Inherit<SGB4LinearArray, SearchGridBuilderPtr> SGB4LinearArrayPtr;

class SGB4CircularArray : public SearchGridBuilder {
public:
  SGB4CircularArray( int nChan, bool isFarField, unsigned int samplingFreq=16000 );
  void setRadius( float radius, float height=0.0 );
  virtual const gsl_vector *getTimeDelays();
  virtual bool nextSearchGrid();

protected:
  const gsl_vector *nextSearchGridFF();
};

typedef Inherit<SGB4CircularArray, SearchGridBuilderPtr> SGB4CircularArrayPtr;

/**
   @brief keeps samples in the previous block. 
          this object is used in order to fill the gap between previous and current block data.
*/
class SampleHolder {
public:
  SampleHolder( size_t chanN, size_t maxSampleDelay ):
    _filledNum(0),_buffer(gsl_matrix_float_calloc(chanN,maxSampleDelay))
  {}

  ~SampleHolder()
  {
    gsl_matrix_float_free(_buffer);
  }
  
  void setSamples( size_t chanX, gsl_vector_float *samples )
  {
    size_t maxSampleDelay = _buffer->size2;

    if( samples->size >= maxSampleDelay ){
      size_t frameY = samples->size - maxSampleDelay;
      for(size_t frameX=0;frameX<maxSampleDelay;frameX++,frameY++){
	gsl_matrix_float_set( _buffer, chanX, frameX, gsl_vector_float_get( samples, frameY ) );
      }
      _filledNum = maxSampleDelay;
    }
    else{
      // shift the elements of the buffer
      size_t frameN = maxSampleDelay - samples->size;
      for(size_t frameX=0;frameX<frameN;frameX++){
	gsl_matrix_float_set( _buffer, chanX, frameX, gsl_matrix_float_get( _buffer, chanX, frameX + samples->size ) );
      }
      // hold the samples of the current block
      size_t frameY = 0;
      for(size_t frameX=frameN;frameX<maxSampleDelay;frameX++,frameY++){
	gsl_matrix_float_set( _buffer, chanX, frameX, gsl_vector_float_get( samples, frameY ) );
      }
      _filledNum += samples->size;
      if(  _filledNum > maxSampleDelay ) 
	_filledNum = maxSampleDelay;
    }
  }

  float getSample( int chanX, int minusFrameX )
  {
    return gsl_matrix_float_get( _buffer, chanX, _buffer->size2 + minusFrameX );
  }

  int nFilled(){ return _filledNum;}

private:
  int  _filledNum;
  gsl_matrix_float *_buffer; /* the number of channels X samples */
};

typedef refcount_ptr<SampleHolder> SampleHolderPtr;

/**
   @brief hold information for a position estimate.
 */
class SourceCandidate {
public:
  SourceCandidate(size_t chanN, double costV=100000):
    _costV(costV), _chanN(chanN){
    _sampledelay = new int[chanN];
    _position = gsl_vector_calloc( 3 );
    _eigenvalues = gsl_vector_alloc( chanN );
  }
  ~SourceCandidate(){
    delete [] _sampledelay;
    gsl_vector_free(_position);
    gsl_vector_free( _eigenvalues );
  }
  void setSourceInfo( int *tau, const gsl_vector *position, double costV, gsl_vector *eigenvalues ){
    for(size_t i=0;i<_chanN;i++){_sampledelay[i]=tau[i];}
    gsl_vector_memcpy(_position, (gsl_vector *)position);
    _costV = costV;
    gsl_vector_memcpy(_eigenvalues, eigenvalues);
  }
  void setConstV( double costV=100000 ){
    _costV = costV;
  }

  double _costV; /* value of the cost function */
  int *_sampledelay;
  gsl_vector *_position;
  gsl_vector *_eigenvalues;
private:
  size_t _chanN;
};

/**
   @class estimate the source positions which provide the larger multi-channel cross correlation values. 
   @usage
   1. setChannel()
   2. next()
   3. 

   @note The algorithm implemented here is described in:
          J. Chen, J. Benesty and Y. Huang, "Robust Time Delay Estimation Exploiting Redundancy Among Multiple Microphones", IEEE Trans. SAP, vol.11, Sep. 2003.
*/

class MCCLocalizer : public VectorFeatureStream {
public:
  MCCLocalizer( SearchGridBuilderPtr &sgbPtr, size_t maxSource=1, const String& nm= "MCCSourceLocalizer" );
  ~MCCLocalizer();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();
  //void setChannel(SampleFeaturePtr& chan);
  void setChannel(VectorFloatFeatureStreamPtr& chan);
  /*@brief obtain a relative delay at a microphone which corresponds to the best candidate */
  int getDelayedSample( int chanX ){return _sourceCandidates[0]->_sampledelay[chanX];}
  /*@brief obtain the maximum MCCC value */
  double getMaxMCCC(){/* mistake: it was getMinMCCC() */
#ifdef _NO_LOG_
    return( 1 -_sourceCandidates[0]->_costV );
#else
    return( 1 -exp(_sourceCandidates[0]->_costV) );
#endif
  }
  /*@brief obtain the best candidate of the position estimates */
  const gsl_vector* getPosition(){return((const gsl_vector* )_sourceCandidates[0]->_position);}

  /*@brief obtain a relative delay at a microphone which corresponds to the N-th best candidate */
  int getNthBestDelayedSample( int nth, int chanX ){return _sourceCandidates[nth]->_sampledelay[chanX];}
  /*@brief obtain the N-th best MCCC value */
  double getNthBestMCCC( int nth ){
#ifdef _NO_LOG_
    return( 1 -_sourceCandidates[nth]->_costV );
#else
    return( 1 - exp(_sourceCandidates[nth]->_costV) );
#endif
  }
  /*@brief obtain theN-th  best candidate of the position estimates */
  const gsl_vector* getNthBestPosition( int nth ){return((const gsl_vector* )_sourceCandidates[nth]->_position);}

  const gsl_vector* getEigenValues(){return ((const gsl_vector*)_sourceCandidates[0]->_eigenvalues);}
  gsl_matrix *getR(){return _R;}

protected:
  bool setIncomingData( int frameX );
  void doEigenValueDecomposition();
  double calcObjectiveFunction( bool normalizeVariance = true );
private:
  gsl_matrix *calcCovarianceMatrix();

private:
  gsl_vector* search( int frameX = -5 );

protected:
  typedef list<VectorFloatFeatureStreamPtr>   _ChannelList; //typedef list<SampleFeaturePtr>   _ChannelList;
  typedef _ChannelList::iterator   _ChannelIterator;
  _ChannelList                     _channelList;

  SearchGridBuilderPtr _sgbPtr;
  vector<const gsl_vector_float *> _blockList;
  int *_tau; /* sample delays */
  gsl_vector *_x;
  gsl_matrix *_R; /* covariance matrix */
  gsl_matrix *_Rcopy;
  double _detR;
  void *_workspace;
  gsl_vector *_eigenvalues;
  SampleHolderPtr _shPtr; /* keep samples of the block processed at a prevous frame */
  vector<SourceCandidate *> _sourceCandidates; /* _sourceCandidates[n] is the N-th best candidate */
};

typedef Inherit<MCCLocalizer, VectorFeatureStreamPtr> MCCLocalizerPtr;

/**
   @class calculate multichannel cross correlation coefficient givin a position or time delays.
*/
class MCCCalculator : public MCCLocalizer {
public:
  MCCCalculator( SearchGridBuilderPtr &sgbPtr, bool normalizeVariance=true, const String& nm= "MCCCalculator" );
  ~MCCCalculator();
  void setTimeDelays( gsl_vector *delays );
  double getMCCC();
  double getCostV();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();
private:
  gsl_matrix *calcCovarianceMatrix( gsl_vector *delays );

private:
  gsl_vector *_delays;
  bool        _normalizeVariance;
};

typedef Inherit<MCCCalculator, MCCLocalizerPtr> MCCCalculatorPtr;

/**
   @class recursively estimate the source position which provides the multi-channel cross correlation.
   @usage

   @note The algorithm implemented here is described in:
         J. Benesty, J. Chen and Y. Huang, "Time-Delay Estimation via Linear Interpolation and Cross Correlation", IEEE Trans. SAP, vol.12, Sep. 2004.
*/
class RMCCLocalizer : public MCCLocalizer {
public:
  RMCCLocalizer( SearchGridBuilderPtr &sgbPtr, float lambda, size_t maxSource=1, const String& nm= "RMCCSourceLocalizer" );
  ~RMCCLocalizer();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();

private:
  void calcInverseMatrix();
  void updateParameters();

  gsl_matrix *_invR; /* an inverse matrix of the covariance matrix */
  gsl_vector *_kd; /* a priori Kalman gain vector */

  //gsl_vector *_old_x;
  gsl_matrix *_old_R; /* covariance matrix at a previous frame */
  gsl_matrix *_old_invR;
  gsl_vector *_old_kd; /* a priori Kalman gain vector at a previous frame */
};

typedef Inherit<RMCCLocalizer, MCCLocalizerPtr> RMCCLocalizerPtr;

#endif
