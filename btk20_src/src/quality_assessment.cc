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
#include "objectiveMeasure/objectiveMeasure.h"

int main( int argc, char **argv )
{
  unsigned fftLen = 64;
  unsigned r = 1;
  unsigned windowType = 1;
  int begin=0, end=-1;
  int opt = -1, longopt_index = 0;
  int normalizationOption = 0;
  const char *optstring = "M:r:w:1:2:b:e:n:h";
  const char *originalSpeechFile;
  const char *enhancedSpeechFile;
  struct option long_options[] = {
     { "help", 0, 0, 'h' },
     { "M", 1, 0, 'M' },
     { "r", 1, 0, 'r' },
     { "w", 1, 0, 'w' },
     { "b", 1, 0, 'b' },
     { "e", 1, 0, 'e' },
     { "normalization", 1, 0, 'n' },
     { "1", 1, 0, '1' },
     { "2", 1, 0, '2' },
     { 0, 0, 0, 0 }
   };

  while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case 'w':
      windowType = atoi( optarg );
      break;
    case 'M':  
      fftLen = atoi( optarg );
      break;
    case 'r':  
      r = atoi( optarg );
      break;
    case 'b':  
      begin = atoi( optarg );
      break;
    case 'e':  
      end = atoi( optarg );
      break;
    case 'n':  
      normalizationOption = atoi( optarg );
      // 1 mean normalization
      // 2 scaling normalization with max peaks
      // 4 scaling normalization with standard deviations
      break;
    case '1':  
      originalSpeechFile = optarg;
      break;
    case '2':  
      enhancedSpeechFile = optarg;
      break;
    default:
      break;
    }
  }

  float val;
  SNRPtr snrP = new SNR();
  ItakuraSaitoMeasurePSPtr ISDistP = new ItakuraSaitoMeasurePS( fftLen,  r, windowType );

  val = snrP->getSNR( originalSpeechFile, enhancedSpeechFile, normalizationOption, 1, 16000, begin, end );
  printf("SNR %f\n",val);

  unsigned shiftLen = ISDistP->frameShiftLength();
  val = ISDistP->getDistance( originalSpeechFile, enhancedSpeechFile, 1, 16000, (int)begin/shiftLen, (int)end/shiftLen );
  printf("IS  %f\n",val);
}
