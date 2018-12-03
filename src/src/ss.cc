#include <stdio.h>
#include <limits.h>
#include <getopt.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "modulated/modulated.h"
#include "postfilter/spectralsubtraction.h"

using namespace sndfile;
using namespace std;


gsl_vector **readFilterCoeffs( const char *fn );

class myOption {

  unsigned _M;
  unsigned _m;
  unsigned _r;
  unsigned _D;
  int _filterLen;
  std::string _coeffFile;
  gsl_vector **_coeffSet;
  bool isNoiseTrainingMode;

public:
  std::string _audioFile;
  std::string _outputFile;
  std::string _noiseFile;
  std::string _outputNoiseFile;
  float _alpha;
  float _floorV;
  int   _frameN;
  float _normFactor;
  int   _samplingRate;
  int   _delayCompensationType;

  myOption(){
    _audioFile = "test.wav";  //list of input audio files
    _coeffFile  = "./M256-m4-r1"; // coefficients of analysis-and-synthesis prototype
    _outputNoiseFile = "outNoiseModel.txt";
    _coeffSet = NULL;
    _noiseFile = "";
    _outputFile = ""; // output wave file
    _M = 256;
    _m = 4;
    _r = 1;
    _alpha = 1.5;
    _floorV = 0.1;
    _frameN = 100;
    _normFactor = 1.0;
    recalcFilterParameter();
    _delayCompensationType = 2;
  }
  
  ~myOption(){
    if( NULL != _coeffSet ){
      gsl_vector_free(_coeffSet[0]);
      gsl_vector_free(_coeffSet[1]);
      free(_coeffSet);
    }
  }

  gsl_vector **setCoeffFile( const char *fn ){
    _coeffFile = fn;
    _coeffSet = readFilterCoeffs( fn );
    //gsl_vector *h_fb = coeffSet[0]; // coeffs of an analysis prototype
    //gsl_vector *g_fb = coeffSet[1]; // coeffs of a synthesis prototype 
    return _coeffSet;
  }

  gsl_vector *getCoeffsOfAnalysisPrototype()
  {
    if( NULL!=_coeffSet ){
      return _coeffSet[0];
    }
    return NULL;
  }

  gsl_vector *getCoeffsOfSynthesisPrototype()
  {
    if( NULL!=_coeffSet ){
      return _coeffSet[1];
    }
    return NULL;
  }
			   
  void setSubbandN( unsigned M ){
    _M = M;
  }
  
  void setFilterLengthFactor( unsigned m ){
    _m = m;
  }

  void setDecimationFactor( unsigned r ){
    _r = r;
  }

  unsigned getSubbandN(){
    return _M;
  }

  unsigned getm(){
    return _m;
  }

  unsigned getr(){
    return _r;
  }

  unsigned getD(){
    recalcFilterParameter();
    return _D;
  }

  unsigned getFilterLength(){
    recalcFilterParameter();
    return _filterLen;
  }

private:
  void recalcFilterParameter(){
    _filterLen = _M * _m;
    _D = _M / powi( 2, _r );
  }
  
};

/**
   @brief read coefficients of analysis and synthesis prototypes.
   @param const char *fn[in] file name of the filter prototype.
   @return gsl_vector *coeffSet[2] 
   *coeffSet[0] : coefficients for the analysis prototype.
   *coeffSet[1] : coefficients for the synthesis prototype.
   @note You must free a pointer returned by this function
 */
gsl_vector **readFilterCoeffs( const char *fn )
{
  FILE *fp;
  int counter = 0;
  list<double> coeffL;
  list<double>::iterator itrL;
  
  fp = fopen( fn, "rt" );
  if( fp == NULL ){
    fprintf( stderr, "cannot open a filter-coefficient file %s\n", fn );    
    return NULL;
  }
  coeffL.clear();
  while(1){
    float coeff;
    if( fscanf( fp, "%f", &coeff ) != 1 )
      break;
    if( ferror(fp ) ){
      fprintf( stderr, "find an error in %s\n", fn );    
      return NULL;      
    }
    coeffL.push_back( coeff );
    counter++;
  }  
  fprintf(stderr,"%d (%d) coefficients in total are loaded\n",(int)coeffL.size(),counter);
  fclose(fp);

  gsl_vector **coeffSet;
  gsl_vector *coeffVA; // coefficients for an analysis filter
  gsl_vector *coeffVS; // coefficients for a synthesis filter
  int filterLen = coeffL.size()/2;

  coeffSet = (gsl_vector **)malloc(2*sizeof(gsl_vector *));
  if( NULL == coeffSet ){
    fprintf( stderr, "could not allocate memory\n" );
    return NULL;
  }
  coeffVA = gsl_vector_alloc( (size_t)filterLen );
  coeffVS = gsl_vector_alloc( (size_t)filterLen );
  itrL  = coeffL.begin();
  for (int i=0; i<filterLen; itrL++,i++ )
    gsl_vector_set( coeffVA, i, *itrL ); 
  for (int i=0; i<filterLen; itrL++,i++ )
    gsl_vector_set( coeffVS, i, *itrL ); 

  coeffSet[0] = coeffVA;
  coeffSet[1] = coeffVS;
  return coeffSet;
}


bool doSS( myOption &myOpt )
{
  const char *audioFile = myOpt._audioFile.c_str();
  gsl_vector *h_fb = myOpt.getCoeffsOfAnalysisPrototype(); // coeffs of an analysis prototype
  gsl_vector *g_fb = myOpt.getCoeffsOfSynthesisPrototype(); // coeffs of a synthesis prototype 
  unsigned DFACTOR = myOpt.getD();
  unsigned fftLen  = myOpt.getSubbandN();
  unsigned mFACTOR = myOpt.getm();
  unsigned rFACTOR = myOpt.getr();
  float    alpha   = myOpt._alpha;
  float    floorV  = myOpt._floorV; 
  int frameN = myOpt._frameN;
  const char *outputFile = myOpt._outputFile.c_str();
  float nf = myOpt._normFactor;
  const char *noiseFile = myOpt._noiseFile.c_str();
  const char *outputNoiseFile = myOpt._outputNoiseFile.c_str();
  int samplingRate =  myOpt._samplingRate;
  int delayCompensationType = myOpt._delayCompensationType;

  SampleFeaturePtr sampleP = new SampleFeature( "", DFACTOR, DFACTOR, true );
  OverSampledDFTAnalysisBankPtr analysisP = new OverSampledDFTAnalysisBank( (VectorFloatFeatureStreamPtr&)sampleP, h_fb, fftLen, mFACTOR, rFACTOR, delayCompensationType );
  SpectralSubtractorPtr ssP = new SpectralSubtractor( fftLen, false, alpha, floorV );
  OverSampledDFTSynthesisBankPtr synthesisP = new OverSampledDFTSynthesisBank((VectorComplexFeatureStreamPtr&)ssP, g_fb, fftLen, mFACTOR, rFACTOR, delayCompensationType );
		
  sampleP->read( audioFile, samplingRate );
  printf("audio %s\n",audioFile );
  ssP->setChannel( (VectorComplexFeatureStreamPtr&)analysisP );

  float max = -10000;
  list<float> dataFL;
  const gsl_vector_float *data;

  if( strlen( noiseFile ) > 0 ){
    if( false == ssP->readNoiseFile( noiseFile, 0 ) ){
      fprintf(stderr,"ssP->readNoiseFile( ) failed\n");
      return false;
    }
    fprintf(stderr,"read a noise model file %s\n",noiseFile);
  }
  else{
    unsigned frameX;

    for(frameX=0;;frameX++){
      try{
	data = synthesisP->next();
      }
      catch(  jiterator_error& e) { 
	break;
      }
      for(unsigned i=0;i<DFACTOR;i++){
	float tmp = gsl_vector_float_get( data, i );
	dataFL.push_back( tmp * nf );
	if( fabs(tmp)  > max )
	  max = fabs(tmp);
      }
      if( sampleP->isEnd() == true )
	break;
      if( frameN > 0 && frameX >= frameN )
	break;
    }
    ssP->stopTraining();
    if( false == ssP->writeNoiseFile( outputNoiseFile, 0 ) ){
      fprintf(stderr,"ssP->writeNoiseFile( ) failed\n");
      return false;
    }
    if( frameN > 0 )
      fprintf(stderr,"finish training a noise model\n");
    else
      fprintf(stderr,"trained a noise model with %d frames\n",frameX);
  }

  for(unsigned frameX=0;;frameX++){
    try{
      data = synthesisP->next();
    }
    catch(  jiterator_error& e) { 
      break;
    }
    for(unsigned i=0;i<DFACTOR;i++){
      float tmp = gsl_vector_float_get( data, i );
      dataFL.push_back( tmp * nf );
      if( fabs(tmp)  > max )
	max = fabs(tmp);
    }
    if( sampleP->isEnd() == true )
      break;
  }
  fprintf(stderr,"MAX %e\n",max);

  sndfile::SF_INFO sfinfo;
  sfinfo.samplerate = 16000;
  sfinfo.channels = 1;
  sfinfo.format = ( 0x010000 | 0x0002 );	//0x0006 0x0002
  sndfile::SNDFILE* waveFP = sf_open( outputFile, sndfile::SFM_WRITE, &sfinfo);
  //sf_command (waveFP, SFC_SET_NORM_FLOAT, NULL, SF_FALSE );
  //sf_command (sndfile, SFC_SET_SCALE_INT_FLOAT_WRITE, NULL, SF_TRUE) ;

  list<float>::iterator itrFL  = dataFL.begin();
  list<float>::iterator itrFLE = dataFL.end();
  for(;itrFL!=itrFLE;itrFL++){    
    //float val = *itrFL;
    //sf_writef_float( waveFP, &val, 1 ); // use this only if the output format flag has 0x0006
    short val = (short)*itrFL;
    sf_writef_short( waveFP, &val, 1 ); // use this only if the output format flag has 0x0002
#ifdef _DEBUG_
    fprintf(stderr,"%e ",val);
#endif
  }
  sf_close(waveFP);
  
  return true;
}

 int main( int argc, char **argv )
 {
   int opt = -1, longopt_index = 0;
   const char *optstring = "a:f:A:C:O:M:m:r:N:n:W:s:d:h";
   struct option long_options[] = {
     { "help", 0, 0, 'h' },
     { "alpha", 1, 0, 'a' },
     { "floorV", 1, 0, 'f' },
     { "audioFile", 1, 0, 'A' },
     { "coeffFile", 1, 0, 'C' },
     { "outputFile", 1, 0, 'O' },
     { "M", 1, 0, 'M' },
     { "m", 1, 0, 'm' },
     { "r", 1, 0, 'r' },
     { "N", 1, 0, 'N' },
     { "normalizationFactor", 1, 0, 'n' },
     { "outNoiseModel", 1, 0, 'W' },
     { "inNoiseModel", 1, 0, 's' },
     { "delayCompensationType", 1, 0, 'd' },
     { 0, 0, 0, 0 }
   };

   myOption myOpt;

   while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case 'A':
      /* getopt signals end of '-' options */
      myOpt._audioFile = optarg;
      break;
    case 'C':
      myOpt.setCoeffFile( optarg );
      break;
    case 'W':
      myOpt._outputNoiseFile = optarg;
      break;
    case 'O':
      myOpt._outputFile = optarg;
      break;
    case 'M':
      myOpt.setSubbandN( atoi(optarg) );
      break;
    case 'm':
      myOpt.setFilterLengthFactor( atoi(optarg) );
      break;
    case 'r':
      myOpt.setDecimationFactor( atoi(optarg) );
      break;
    case 'N':
      myOpt._frameN = atoi(optarg);
      break;
    case 'a':
      myOpt._alpha = atof(optarg);
      break;
    case 'f':
      myOpt._floorV = atof(optarg);
      break;
    case 'n':
      myOpt._normFactor = atof(optarg);
      break;
    case 's':
      myOpt._noiseFile = optarg;
      break;
    case 'd':
      myOpt._delayCompensationType = atoi(optarg);
      break;
    default:
      break;
    }
  }
  
   doSS( myOpt );
}
