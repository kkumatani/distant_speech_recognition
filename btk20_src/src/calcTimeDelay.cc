#include <stdio.h>
#include <limits.h>
#include <getopt.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "common/jpython_error.h"
#include "TDEstimator/CCTDE.h"

using namespace sndfile;
using namespace std;

class myOption {
public:
  std::string _audioFile1;
  std::string _audioFile2;
  int   _chanX1;
  int   _chanX2;
  int   _windowLen;
  int   _shiftLen1;
  int   _shiftLen2;
  int   _from1;
  int   _to1;
  int   _from2;
  int   _to2;
  int   _nCands;
  bool  _useAllSamples;

  myOption(){
    _chanX1 = 1;
    _chanX2 = 1;
    _windowLen = 16384; //2097152;
    _shiftLen1 = _windowLen / 2;
    _shiftLen2 = _windowLen / 2;
    _from1 = 0;
    _to1 = -1;
    _from2 = 0;
    _to2 = -1;
    _nCands = 5;
    _useAllSamples = false;
  }
  
  ~myOption(){
  }

};

bool calcTimeDelay( myOption &myOpt )
{
  SampleFeaturePtr samplefeat1 = new SampleFeature( "", myOpt._windowLen, myOpt._shiftLen1, true );
  SampleFeaturePtr samplefeat2 = new SampleFeature( "", myOpt._windowLen, myOpt._shiftLen2, true );

  samplefeat1->read( myOpt._audioFile1, 0, 16000, myOpt._chanX1, myOpt._from1, myOpt._to1 );
  samplefeat2->read( myOpt._audioFile2, 0, 16000, myOpt._chanX2, myOpt._from2, myOpt._to2 );
  printf("compare %s with %s\n", myOpt._audioFile1.c_str(), myOpt._audioFile2.c_str() ); fflush(stdout);

  CCTDEPtr ccTDE = new CCTDE( samplefeat1, samplefeat2, myOpt._windowLen, myOpt._nCands );
  unsigned frameX1, frameX2;
  unsigned frameN1 = samplefeat1->samplesN() / myOpt._shiftLen1 - 1;
  unsigned frameN2 = samplefeat2->samplesN() / myOpt._shiftLen2 - 1;
  int sampleRate = samplefeat1->getSampleRate();
  const unsigned *timeDelays;

  if( false==myOpt._useAllSamples ){
    try {
      while(true){
	ccTDE->next();
	for(frameX1=0;frameX1<frameN1;frameX1++){
	  //const gsl_vector* vec0 = ccTDE->nextX( 0, frameX1 );
	  for(frameX2=1;frameX2<frameN2;frameX2++){
	    int sampleDiff = frameX2 * myOpt._shiftLen2 - frameX1 *myOpt._shiftLen1;

	    printf("#Base Delay (sample) : (msec)\n");
	    printf("%d : %f\n", sampleDiff, 1000.0*sampleDiff/sampleRate); fflush(stdout);
	    const gsl_vector* vec1 = ccTDE->nextX( 1, frameX2 );	    
	  }	  
	  samplefeat1->next();
	  samplefeat2->reset();	  
	}
      }
    } catch (j_error& e) {
      if (e.getCode() == JPYTHON) {
	jpython_error *pe = static_cast<jpython_error*>(&e);
	throw jpython_error();
      }
    } catch (...) {
      printf("Finshed\n");fflush(stdout);
      throw;
    }
  }
  else{
    ccTDE->allsamples();
    timeDelays = ccTDE->getSampleDelays();
  }

  return true;
}

 int main( int argc, char **argv )
 {
   int opt = -1, longopt_index = 0;
   const char *optstring = "A:1:2:w:x:y:a:b:c:d:e:f:n:h";
   struct option long_options[] = {
     { "help", 0, 0, 'h' },
     { "allsamp", 1, 0, 'A' },
     { "audio1", 1, 0, '1' },
     { "audio2", 1, 0, '2' },
     { "windowLen", 1, 0, 'w' },
     { "chanX1", 1, 0, 'x' },
     { "chanX2", 1, 0, 'y' },
     { "shiftLen1", 1, 0, 'a' },
     { "shiftLen2", 1, 0, 'b' },
     { "from1", 1, 0, 'c' },
     { "from2", 1, 0, 'd' },
     { "to", 1, 0, 'e' },
     { "to2", 1, 0, 'f' },
     { "n", 1, 0, 'n' },
     { 0, 0, 0, 0 }
   };

   myOption myOpt;

   while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case 'A':
      myOpt._useAllSamples = (bool)atoi(optarg);
    case '1':
      /* getopt signals end of '-' options */
      myOpt._audioFile1 = optarg;
      break;
    case '2':
      myOpt._audioFile2 = optarg;
      break;
    case 'w':
      myOpt._windowLen = atoi(optarg);
      break;
    case 'a':
      myOpt._shiftLen1 = atoi(optarg);
    case 'b':
      myOpt._shiftLen2 = atoi(optarg);
      break;
    case 'n':
      myOpt._nCands = atoi(optarg);
      break;
    default:
      break;
    }
  }
  
   calcTimeDelay( myOpt );
}
