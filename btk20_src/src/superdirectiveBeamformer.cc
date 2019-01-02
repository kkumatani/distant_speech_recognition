#include <stdio.h>
#include <getopt.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
//#include "modulated/modulated.h"
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

gsl_matrix *readMicrophonePositionFile( const char *micPosFn )
{
  gsl_matrix *micPos;
  int chanN;

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

    micPos = gsl_matrix_alloc(chanN,3);

    for(int i=0;i<chanN;i++){
      float x,y,z;
      fscanf( fp, "%f", &x ); // x of the ith mic
      fscanf( fp, "%f", &y ); // y of the ith mic
      fscanf( fp, "%f", &z ); // z of the ith mic
      gsl_matrix_set( micPos, i, 0, x );
      gsl_matrix_set( micPos, i, 1, y );
      gsl_matrix_set( micPos, i, 2, z );
      //fprintf(stderr,"%d-th mic %e %e %e\n",i,micPos[i][0],micPos[i][1],micPos[i][2]);
    }
    fclose(fp);
  }

  return micPos;
}

/**
   @brief calculate time delay of arrival (TDOA) in the case of a circular microphone array.
   @param float azimuth[in] direction of the arriving signal 
   @param loat elevation[in] direction of the arriving signal 
   @param const char *micPosFn[in] file containing the geometry of the microphone array.
   
   @return time delays
 */
gsl_vector *calcDelaysPolar2( float azimuth, float elevation, gsl_matrix *micPos )
{
  gsl_vector *delays;
  int   chanN = micPos->size1;

  delays = gsl_vector_alloc( chanN );

  float c_x = - sin( elevation ) * cos( azimuth );
  float c_y = - sin( elevation ) * sin( azimuth );
  float c_z = - cos( elevation );
  for( int i=0;i<chanN;i++){
    float x = gsl_matrix_get( micPos, i, 0 );
    float y = gsl_matrix_get( micPos, i, 1 );
    float z = gsl_matrix_get( micPos, i, 2 );
    float t = (c_x * x + c_y * y + c_z * z ) / SOUNDSPEED;
    gsl_vector_set( delays, i, t ); 
  }

  return delays;
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
		    const char *micPosFile,
		    float azimuth, float elevation,
		    double sampleRate, unsigned fftLen, int pf, double alpha, 
		    unsigned D, unsigned m, unsigned r, const char *outputFile )
{
  FILE *fp;
  char fn[FILENAME_MAX];
  gsl_vector **coeffSet = getFilterCoeffs( coeffFile );
  gsl_vector *h_fb = coeffSet[0]; // coeffs of an analysis prototype
  gsl_vector *g_fb = coeffSet[1]; // coeffs of a synthesis prototype 
  vector<SampleFeaturePtr> sampleFeaturePL;
  vector<OverSampledDFTAnalysisBankPtr> analysisFBPL;
  SubbandMVDRPtr beamformerP   = new SubbandMVDR( fftLen, false );
  ZelinskiPostFilterPtr output = new ZelinskiPostFilter((VectorComplexFeatureStreamPtr&)beamformerP, fftLen, alpha, pf );
  OverSampledDFTSynthesisBankPtr synthesisFBP = new OverSampledDFTSynthesisBank((VectorComplexFeatureStreamPtr&)output, g_fb, fftLen, m, r );

  gsl_matrix* micPos = readMicrophonePositionFile( micPosFile );
  gsl_vector *delays = calcDelaysPolar2( azimuth, elevation, micPos );

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

  beamformerP->calcArrayManifoldVectors( sampleRate, delays );
  beamformerP->setDiffuseNoiseModel( micPos, sampleRate, SOUNDSPEED );
  beamformerP->divideAllNonDiagonalElements( 0.01 );
  beamformerP->calcMVDRWeights( sampleRate, 1.0E-8 );

  float max = -10000;
  list<float> dataFL;

  output->setBeamformer( beamformerP );
  for(;;){
    const gsl_vector_float *data;
    try{
      data = synthesisFBP->next();
      //if( true== sampleFeaturePL[0]->isEnd() ){
      if( true== analysisFBPL[0]->isEnd() ){
	fprintf(stderr,"end\n");
	break;
      }
    }
    catch(  jiterator_error& e) { 
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
  gsl_matrix_free( micPos );
  gsl_vector_free( delays );

  return true;
}

int main( int argc, char **argv )
{
  int opt = -1, longopt_index = 0;
  const char *optstring = "A:P:C:O:M:a:e:h";
  struct option long_options[] = {
    { "help", 0, 0, 'h' },
    { "audioList", 1, 0, 'A' },
    { "micPosFile", 1, 0, 'P' },
    { "coeffFile", 1, 0, 'C' },
    { "outputFile", 1, 0, 'O' },
    { "M", 1, 0, 'M' },
    { "azimuth", 1, 0, 'a' },
    { "elevation", 1, 0, 'e' },
    { 0, 0, 0, 0 }
  };

  std::string audioList = "./testL"; // list of input audio files
  std::string micPosFile  = "./position"; // mic. positions
  std::string coeffFile  = "./M256-m4-r1"; // coefficients of analysis-and-synthesis prototype
  std::string outputFile  = "./beamformed.wav"; // output wave file
  unsigned M = 256, m=4, r=1;
  int Len = M * m;
  float azimuth = 0.0;
  float elevation = 0.0;
  double sampleRate = 16000.0;
  int pf = 2;
  double alpha = 0.6;
  unsigned D = M / powi( 2, r );

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
    case 'M':
      M = atoi(optarg);
      break;
    case 'a':
      azimuth = atof(optarg);
      break;
    case 'e':
      elevation = atof(optarg);
      break;
    default:
      break;
    }
  }

  doBeamforming( audioList.c_str(), coeffFile.c_str(), micPosFile.c_str(), azimuth, elevation, sampleRate, M, pf, alpha, D, m, r, outputFile.c_str() );

}
