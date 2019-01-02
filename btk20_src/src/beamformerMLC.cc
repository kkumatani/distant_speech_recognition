#include <stdio.h>
#include <getopt.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "modulated/modulated.h"
#include "postfilter/postfilter.h"
#include "beamformer/beamformer.h"

using namespace sndfile;
using namespace std;
#define SOUNDSPEED 343740.0

/**
   @brief read coefficients of analysis and synthesis prototypes.
   @param const char *fn[in] file name of the filter prototype.
   @return gsl_vector *coeffSet[2] 
   *coeffSet[0] : coefficients for the analysis prototype.
   *coeffSet[1] : coefficients for the synthesis prototype.
   @note You must free a pointer returned by this function
 */
gsl_vector **getFilterCoeffs( const char *fn )
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

/**
   @brief read positions of microphones
   @param const char *micPosFn[in] file containing the geometry of the microphone array.
   
   @return the geometry of the microphone array
 */
float **getGeometryOfArray( const char *micPosFn, int *chanNP )
{
  float **micPos;
  int   chanN;

  { // read positions of microphones
    // format
    // the-number-of-microphones
    // x1 y1 z1
    // ...
    FILE *fp = fopen( micPosFn, "rt" );
    if( fp == NULL ){
      fprintf( stderr, "cannot open a microphone-position file %s\n", micPosFn );
      return NULL;
    }
    fscanf( fp, "%d", &chanN );
    fprintf(stderr,"The number of microphones is %d\n",chanN);
    micPos = (float **)malloc(chanN*sizeof(float *));
    if( NULL==micPos ){
      fprintf(stderr,"cannot allocate memory\n");
      return NULL;
    }    

    for(int i=0;i<chanN;i++){
      micPos[i] = (float *)malloc(3*sizeof(float));
      if( NULL==micPos[i] ){
	fprintf(stderr,"cannot allocate memory for the %d-th mic\n",i);
	return NULL;
      }
      fscanf( fp, "%f", &micPos[i][0] ); // x of the ith mic
      fscanf( fp, "%f", &micPos[i][1] ); // y of the ith mic
      fscanf( fp, "%f", &micPos[i][2] ); // z of the ith mic
      fprintf(stderr,"%d-th mic %e %e %e\n",i,micPos[i][0],micPos[i][1],micPos[i][2]);
    }
    fclose(fp);
  }

  *chanNP = chanN;
  return micPos;
}


bool calcTimeDelays( int target_index, const char *micPosFn, const char *srcPosFile, gsl_vector** delaysTP, gsl_matrix** delaysJP )
{
  int chanN;
  float **micPos = getGeometryOfArray( micPosFn, &chanN );

  int   sourceN;     // the number of sources
  float **positions; // positions[sourceN][2]

  // read position estimates of sources
  FILE *fp = fopen( srcPosFile, "rt" );  
  if( fp == NULL ){
    fprintf( stderr, "cannot open %s\n", srcPosFile );    
    return false;
  }

  char buffer[FILENAME_MAX];
  sourceN = 0;
  while( fgets( buffer, FILENAME_MAX, fp)!=NULL ){
    sourceN++;
    int id;
    if( sscanf( buffer, "%d", &id ) != 1 )
      break;
    if( ferror(fp ) )
      break;
  }

  fprintf( stderr, "The number of active sources is %d\n", sourceN );
  fseek( fp, 0 , SEEK_SET );
  positions = (float **)malloc(sourceN*sizeof(float *));
  if( NULL==positions ){
      fprintf(stderr,"cannot allocate memory\n");
      return false;
  }
  for(int i=0;i<sourceN;i++){
    int id;
    positions[i] = (float *)malloc(2*sizeof(float));
    if( NULL==positions[i] ){
      fprintf(stderr,"cannot allocate memory for the %d-th source\n",i);
      return false;
    }
    fscanf( fp, "%d", &id ); // 
    fscanf( fp, "%f", &positions[i][0] ); // azimuth
    fscanf( fp, "%f", &positions[i][1] ); // elevation
    fprintf(stderr,"%d-th source %e %e\n",i,positions[i][0],positions[i][1]);
  }
  fclose(fp);

  { // time delays for a target source
    *delaysTP = gsl_vector_alloc( chanN );
    
    float azimuth = positions[target_index][0];
    float elevation = positions[target_index][1];
    float c_x = - sin( elevation ) * cos( azimuth );
    float c_y = - sin( elevation ) * sin( azimuth );
    float c_z = - cos( elevation );
    for( int i=0;i<chanN;i++){
      float t = (c_x * micPos[i][0] + c_y * micPos[i][1] + c_z * micPos[i][2]) / SOUNDSPEED;
      gsl_vector_set( *delaysTP, i, t );
    }
  }

  { // time delays for a interference signals
    *delaysJP = NULL; 
    int NC = sourceN;      // the number of constraints ( the number of interference signals )
    
    if( NC > 1 ){
      *delaysJP = gsl_matrix_alloc( NC-1, chanN );
      for( int i=0,idx=0;i<sourceN;i++){
	if( i==target_index  )
	  continue;
	for( int j=0;j<chanN;j++){
	  
	  float azimuth = positions[i][0];
	  float elevation = positions[i][1];
	  float c_x = - sin( elevation ) * cos( azimuth );
	  float c_y = - sin( elevation ) * sin( azimuth );
	  float c_z = - cos( elevation );
	  float t = (c_x * micPos[j][0] + c_y * micPos[j][1] + c_z * micPos[j][2]) / SOUNDSPEED;
	  gsl_matrix_set( *delaysJP, idx, j, t );
	}
	idx++;
      }
    }// if( NC > 1 )
  
  }
  
  for( int i=0;i<chanN;i++)
    free( micPos[i] );
  free( micPos );

  for(int i=0;i<sourceN;i++)
    free( positions[i] );
  free( positions );

  return true;
}

/**
   @brief beamform the multi-channel data
   @param const char *audioList : a file listing input audio files
   @param const char *coeffFile : a file of the filter prototypes
   @param unsigned fftLen : the number of subbands / FFT bins
   @param int pf : a type of post-filtering
   @param double alpha : 
   @param int D : frame shift
   @param int m : filter length factor
   @param int r : decimation factor
 */
bool doBeamforming( const char *audioList, const char *coeffFile, 
		    gsl_vector* delaysT, gsl_matrix* delaysJ, double sampleRate, 
		    unsigned fftLen, int pf, double alpha, unsigned D, unsigned m, unsigned r, const char *outputFile )
{
  FILE *fp;
  char fn[FILENAME_MAX];
  gsl_vector **coeffSet = getFilterCoeffs( coeffFile );
  gsl_vector *h_fb = coeffSet[0]; // coeffs of an analysis prototype
  gsl_vector *g_fb = coeffSet[1]; // coeffs of a synthesis prototype 
  list<SampleFeaturePtr> sampleFeaturePL;
  list<OverSampledDFTAnalysisBankPtr> analysisFBPL;
  SubbandGSCPtr beamformerP = new SubbandGSC( fftLen, false );
  ZelinskiPostFilterPtr output = new ZelinskiPostFilter((VectorComplexFeatureStreamPtr&)beamformerP, fftLen, alpha, pf );
  OverSampledDFTSynthesisBankPtr synthesisFBP = new OverSampledDFTSynthesisBank((VectorComplexFeatureStreamPtr&)output, g_fb, fftLen, m, r );

  fp = fopen( audioList, "rt" );
  if( fp == NULL ){
    fprintf( stderr, "cannot open %s\n", audioList );    
    return false;
  }
  while(1){
    if( fscanf( fp, "%s", fn ) != 1 )
      break;
    fprintf(stderr,"read %s\n",fn);// read an audio file.
    if( ferror(fp ) ){
      fprintf( stderr, "find an error in %s\n", audioList );    
      return false;      
    }
    SampleFeaturePtr sampleFeatureP = new SampleFeature( "", D, D, true );
    sampleFeatureP->read( fn, sampleRate);
    OverSampledDFTAnalysisBankPtr analysisFBP = new OverSampledDFTAnalysisBank( (VectorFloatFeatureStreamPtr&)sampleFeatureP, h_fb, fftLen, m, r );
    beamformerP->setChannel( (VectorComplexFeatureStreamPtr&)analysisFBP );
    sampleFeaturePL.push_back( sampleFeatureP );
    analysisFBPL.push_back( analysisFBP );
  }  
  fclose(fp);
  if( delaysJ == NULL ){
     beamformerP->calcGSCWeights( sampleRate, delaysT );
  }
  else{
    unsigned NC = delaysJ->size1 + 1;
    beamformerP->calcGSCWeightsN( sampleRate, delaysT, delaysJ, NC );
  }

  float max = -10000;
  list<float> dataFL;

  output->setBeamformer( beamformerP );
  for(;;){
    const gsl_vector_float *data;
    try{
      data = synthesisFBP->next();
    }
    catch(  jiterator_error& e) { 

      break;
    }
    if( true == synthesisFBP->isEnd() ){
      printf("synthesisFBP->isEnd()\n");
      break;
    }
    for(int i=0;i<D;i++){
      float tmp = gsl_vector_float_get( data, i );
      dataFL.push_back( tmp );
      if( fabs(tmp)  > max )
	max = fabs(tmp);
    }
  }

  fprintf(stderr,"output wave file %s\n",outputFile);
  SF_INFO sfinfo;
  sfinfo.samplerate = 16000;
  sfinfo.channels = 1;
  sfinfo.format = ( 0x010000 | 0x0006 );//0x0006 0x0002 
  SNDFILE* waveFP = sf_open( outputFile, SFM_WRITE, &sfinfo);

  list<float>::iterator itrFL  = dataFL.begin();
  list<float>::iterator itrFLE = dataFL.end();
  for(;itrFL!=itrFLE;itrFL++){    
    float val = *itrFL / max;
    sf_writef_float( waveFP, &val, 1 );
#ifdef _DEBUG_
    fprintf(stderr,"%e ",val);
#endif
  }
  sf_close(waveFP);
  
  gsl_vector_free(coeffSet[0]);
  gsl_vector_free(coeffSet[1]);
  free(coeffSet);

  return true;
}

int main( int argc, char **argv )
{
  int opt = -1, longopt_index = 0;
  const char *optstring = "A:P:C:O:S:M:i:h";
  struct option long_options[] = {
    { "help", 0, 0, 'h' },
    { "audioList", 1, 0, 'A' },
    { "micPosFile", 1, 0, 'P' },
    { "coeffFile", 1, 0, 'C' },
    { "outputFile", 1, 0, 'O' },
    { "srcPosFile", 1, 0, 'S' },
    { "M", 1, 0, 'M' },
    { "target_index", 1, 0, 'i' },
    { 0, 0, 0, 0 }
  };

  std::string audioList = "./testL"; // list of input audio files
  std::string micPosFile = "./array.txt"; // mic. positions
  std::string coeffFile  = "./M256-m4-r1"; // coefficients of analysis-and-synthesis prototype
  std::string srcPosFile = "./source_position.txt";
  std::string outputFile = "./beamformed.wav"; // output wave file
  unsigned M = 256, m=4, r=1;
  int Len = M * m;
  double sampleRate = 16000.0;
  int pf = 2;
  double alpha = 0.6;
  unsigned D = M / powi( 2, r );
  unsigned target_index = 0;

  while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case 'A':
      /* getopt signals end of '-' options */
      audioList = optarg;
      break;
    case 'P':
      micPosFile = optarg;
      break;
    case 'C':
      coeffFile = optarg;
      break;
    case 'O':
      outputFile = optarg;
      break;
    case 'S':
      srcPosFile = optarg;
      break;
    case 'M':
      M = atoi(optarg);
      break;
    case 'i':
       target_index = atoi(optarg);
      break;
    default:
      break;
    }
  }
  gsl_vector* delaysT;
  gsl_matrix* delaysJ;

  if( calcTimeDelays( target_index, micPosFile.c_str(), srcPosFile.c_str(), &delaysT, &delaysJ )==false ){
    fprintf(stderr,"calcTimeDelays() failed\n");
    return -1;
  }
  doBeamforming( audioList.c_str(), coeffFile.c_str(), delaysT, delaysJ, sampleRate, M, pf, alpha, D, m, r, outputFile.c_str() );

  gsl_vector_free( delaysT );
  gsl_matrix_free( delaysJ );
}
