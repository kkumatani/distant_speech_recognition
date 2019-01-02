#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "modulated/modulated.h"
#include "postfilter/spectralsubtraction.h"

#ifdef __JACK__
#include <jack/jack.h>
#include <jack/ringbuffer.h>
#endif

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

public:
  std::string _connectionName;
  std::string _outputFile;
  std::string _noiseFile;
  std::string _outputNoiseFile;
  float _alpha;
  float _floorV;
  int   _frameN;
  float _normFactor;

  myOption(){
    _connectionName = "system:capture_1";  //list of input audio files
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


#ifdef __JACK__

void jackoutput_error (const char *desc){
  fprintf (stderr, "JACK output error: %s\n", desc);
}

class JackOutput {
public:
  JackOutput( const char *client_name = "myjackout" );
  ~JackOutput();
  void addPort( unsigned buffersize, const char *port_name="output");
  void output( const gsl_vector_float *samples, float nf = 1.0 );
private:
  int process_callback (jack_nframes_t nframes);
  static int _process_callback(jack_nframes_t nframes, void *arg) {
    return static_cast<JackOutput *> (arg)->process_callback (nframes);
  }
  void shutdown_callback (void);
  static void _shutdown_callback(void *arg) {
    static_cast<JackOutput *> (arg)->shutdown_callback();
  }
  int srate_callback (jack_nframes_t nframes );
  static int _srate_callback( jack_nframes_t nframes, void *arg ){
    static_cast<JackOutput *> (arg)->srate_callback(nframes);
  }

  jack_client_t *_client;
  jack_nframes_t _sr;   /*The current sample rate*/
  vector<jack_channel_t*> _channel;
  jack_channel_t *_tail_channel;
};

void JackOutput::shutdown_callback(void)
{
  throw j_error("JACK shutdown");
}

int JackOutput::process_callback(jack_nframes_t nframes)
{
  /*grab our output buffer*/
  size_t bytes;
  jack_default_audio_sample_t *out;

  for (unsigned i = 0; i < _channel.size(); i++){

    out = (jack_default_audio_sample_t *) jack_port_get_buffer (_channel[i]->port, nframes);
    bytes = jack_ringbuffer_read(_channel[i]->buffer, (char *) out, sizeof(jack_default_audio_sample_t) * nframes);
  }

  return 0;      
}

int JackOutput::srate_callback(jack_nframes_t nframes)
{
  fprintf (stderr,"the sample rate is now %lu/sec\n", nframes);
  _sr = nframes;
  return 0;
}

/**
   @brief a constructor for playing audio data through the Jack interface.
   @param const char *client_name[in] a name of this Jack client 
 */
JackOutput::JackOutput( const char *client_name )
{
  jack_set_error_function(jackoutput_error);

  _client = jack_client_new(client_name);
  if ( _client == NULL ) {
    fprintf (stderr, "jack server %s not running?\n",client_name);
    throw jio_error("Jack server not running?");
  }

  jack_set_process_callback(_client, _process_callback, this);
  jack_set_sample_rate_callback( _client, _srate_callback, this);
  jack_on_shutdown( _client, _shutdown_callback, this);
  _sr = jack_get_sample_rate (_client);

}

JackOutput::~JackOutput()
{
  jack_client_close(_client);
  for (unsigned i=0; i<_channel.size(); i++) {
    jack_ringbuffer_free(_channel[i]->buffer);
    delete _channel[i];
  }
}

/**
   @brief connect ports between your client module and audio device.
   @param unsigned buffersize[in]
   @param const char *connection[in]
 */
void JackOutput::addPort( unsigned buffersize, const char *connection )
{
  const char **ports;
  jack_channel_t* ch = new (jack_channel_t);
  ch->buffersize = buffersize;

  ch->port = jack_port_register( _client, connection, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
  if( ch->port == NULL ){
    delete ch;
    ch = NULL;
    printf("cannot register a jack output port\n");
    throw jio_error("cannot registor a jack output port\n");
  }

  if( jack_activate(_client) ){
    delete ch;
    ch = NULL;
    printf("cannot activate client %s\n",connection);
    throw jio_error ("cannot activate client %s\n",connection);
  }
  
   /* connect the ports*/
   ports = jack_get_ports( _client, NULL, NULL, JackPortIsPhysical|JackPortIsInput );
   if( ports == NULL ){
     delete ch;
     ch = NULL;
     printf("Cannot find any physical playback ports\n");
     throw jio_error( "Cannot find any physical playback ports\n");
   }

   for(int i=0;ports[i]!=NULL;i++){
     if( jack_connect(_client, jack_port_name(ch->port), ports[i]) ){
       fprintf(stderr, "cannot connect output ports\n");
     }
   }

   if(ch){
     ch->buffer = jack_ringbuffer_create( sizeof(jack_default_audio_sample_t) * buffersize );
     ch->can_process = true;
     _channel.push_back(ch);
     _tail_channel = ch;
   }
   
  free(ports);
}

void JackOutput::output( const gsl_vector_float *samples, float nf )
{
  size_t spsize;
  jack_default_audio_sample_t frame;
  unsigned s = sizeof(jack_default_audio_sample_t)*(samples->size);

#define MYMAXLOOP 480000
  for(unsigned i=0;i< MYMAXLOOP;i++){
    spsize =jack_ringbuffer_write_space( _tail_channel->buffer );
    //printf("%d:%d\n",i,spsize);
    if( spsize >= s )
      break;
  }

  for (unsigned i = 0; i < samples->size; i++) {
    frame =  gsl_vector_float_get( samples, i ) * nf;
    jack_ringbuffer_write( _tail_channel->buffer, (char*) &frame, sizeof(frame));
  }
}

bool doSS( myOption &myOpt )
{
  gsl_vector *h_fb = myOpt.getCoeffsOfAnalysisPrototype(); // coeffs of an analysis prototype
  gsl_vector *g_fb = myOpt.getCoeffsOfSynthesisPrototype(); // coeffs of a synthesis prototype 
  unsigned DFactor = myOpt.getD();
  unsigned fftLen  = myOpt.getSubbandN();
  unsigned mFACTOR = myOpt.getm();
  unsigned rFACTOR = myOpt.getr();
  float    alpha   = myOpt._alpha;
  float    floorV  = myOpt._floorV; 
  const char *connectionName = myOpt._connectionName.c_str();
  unsigned frameN = myOpt._frameN;
  const char *outputFile = myOpt._outputFile.c_str();
  float nf = myOpt._normFactor;
  const char *noiseFile = myOpt._noiseFile.c_str();
  const char *outputNoiseFile = myOpt._outputNoiseFile.c_str();

  JackPtr jackP = new Jack("rtss");
  JackFeaturePtr jackFeatP = new JackFeature( jackP, DFactor, DFactor*100, connectionName, "myjack2" );

  OverSampledDFTAnalysisBankPtr analysisP = new OverSampledDFTAnalysisBank( (VectorFloatFeatureStreamPtr&)jackFeatP, h_fb, fftLen, mFACTOR, rFACTOR );
  SpectralSubtractorPtr ssP = new SpectralSubtractor( fftLen, false, alpha, floorV );
  OverSampledDFTSynthesisBankPtr synthesisP = new OverSampledDFTSynthesisBank((VectorComplexFeatureStreamPtr&)ssP, g_fb, fftLen, mFACTOR, rFACTOR );
  ssP->setChannel( (VectorComplexFeatureStreamPtr&)analysisP );

  if( strlen( noiseFile ) > 0 ){
    if( false == ssP->readNoiseFile( noiseFile, 0 ) ){
      fprintf(stderr,"ssP->readNoiseFile( ) failed\n");
      return false;
    }
    fprintf(stderr,"read a noise model file\n");
  }
  else{
    for(unsigned frameX=0;frameX<frameN;frameX++){
      jackP->start();
      try{
	synthesisP->next();
      }
      catch(  jiterator_error& e) { 
	break;
      }
    }
    ssP->stopTraining();
    if( false == ssP->writeNoiseFile( outputNoiseFile, 0 ) ){
      fprintf(stderr,"ssP->writeNoiseFile( ) failed\n");
      return false;
    }
    ssP->clearNoiseSamples();

    fprintf(stderr,"finish training a noise model\n");
  }

  ssP->startNoiseSubtraction();
#ifdef REALTIME_OUTPUT
  JackOutput *jackoutP = new JackOutput( "jo" );
  jackoutP->addPort( DFactor, "system::playback_1" );
#endif

  float max = -10000;
  list<float> dataFL;
  for(unsigned frameX=0;frameX <= 4800/2 ;frameX++){
    jackP->start();
    const gsl_vector_float *data;

    try{
      data = synthesisP->next();
    }
    catch(  jiterator_error& e) { 
      break;
    }
    for(unsigned i=0;i<DFactor;i++){
      float tmp = gsl_vector_float_get( data, i );
      dataFL.push_back( tmp * nf );
      if( fabs(tmp)  > max )
	max = fabs(tmp);
    }
#ifdef REALTIME_OUTPUT
    jackoutP->output( data, nf );
#endif
  }
  fprintf(stderr,"MAX %e\n",max);
#ifdef REALTIME_OUTPUT
  delete jackoutP;
#endif

  // write a wave file
  if( strlen(outputFile) > 0 ){
    sndfile::SF_INFO sfinfo;
    sfinfo.samplerate = (int)jackP->getSampleRate();
    sfinfo.channels = 1;
    sfinfo.format = ( 0x010000 | 0x0006 ); /* 32 bit float data 0x0006, Signed 16 bit data 0x0002 */
    sndfile::SNDFILE* waveFP = sf_open( outputFile, sndfile::SFM_WRITE, &sfinfo);
  
    list<float>::iterator itrFL  = dataFL.begin();
    list<float>::iterator itrFLE = dataFL.end();
    for(;itrFL!=itrFLE;itrFL++){    
      float val = *itrFL;
      sf_writef_float( waveFP, &val, 1 );
#ifdef _DEBUG_
      fprintf(stderr,"%e ",val);
#endif
    }
    sf_close(waveFP);
  }
#if 1
  {
    JackOutput *jackoutP = new JackOutput( "jo" );
    gsl_vector_float *samples = gsl_vector_float_alloc( dataFL.size() );
    list<float>::iterator itrFL  = dataFL.begin();
    list<float>::iterator itrFLE = dataFL.end();

    for(unsigned i=0;itrFL!=itrFLE;itrFL++,i++){    
      gsl_vector_float_set( samples, i, *itrFL );
    }
    jackoutP->addPort( dataFL.size(), "system::playback_1" );
    jackoutP->output( samples );

    sleep(5);
    delete jackoutP;
    gsl_vector_float_free( samples );
  }
#endif
  return true;
}

#endif

 int main( int argc, char **argv )
 {
   int opt = -1, longopt_index = 0;
   const char *optstring = "J:a:f:C:O:M:N:n:W:h";
   struct option long_options[] = {
     { "help", 0, 0, 'h' },
     { "jackportname", 1, 0, 'J' },
     { "alpha", 1, 0, 'a' },
     { "floorV", 1, 0, 'f' },
     { "coeffFile", 1, 0, 'C' },
     { "outputFile", 1, 0, 'O' },
     { "M", 1, 0, 'M' },
     { "N", 1, 0, 'N' },
     { "normalizationFactor", 1, 0, 'n' },
     { "noiseModel", 1, 0, 'W' },
     { 0, 0, 0, 0 }
   };
   myOption myOpt;

   while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case 'J':
      myOpt._connectionName = optarg;
      break;
    case 'W':
      myOpt._outputNoiseFile = optarg;
      break;
    case 'C':
      myOpt.setCoeffFile( optarg );
      break;
    case 'O':
      myOpt._outputFile = optarg;
      break;
    case 'M':
      myOpt.setSubbandN( atoi(optarg) );
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
    default:
      break;
    }
  }

#ifdef __JACK__
  doSS( myOpt );
#else
  fprintf(stderr,"Jack SDK is not installed\n");
#endif
 }
