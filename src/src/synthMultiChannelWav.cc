#include <stdio.h>
#include <limits.h>
#include <getopt.h>
#include <list>
#include <math.h>
#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "common/jpython_error.h"

using namespace sndfile;

#define NINPUTFILES 2
class myOption {
public:
  std::string _audioFile[NINPUTFILES];
  std::string _outFile;

  myOption(){
    _outFile = "new.wav";
  }
  
  ~myOption(){
  }

};

void usage( char *progn )
{
  printf("NAME %s\n", progn);
  printf("\n");  
  printf("SYNOPSIS\n");
  printf("\t %s -1 infile1 -2 infile2 -o outfile\n",progn);
  printf("\n");
  printf("DESCRIPTION\n");
  printf("\t Read multiple audio files and synthesize a multi-channel audio file with all the input channels\n");
}

bool synthMultiChannelWav( myOption &myOpt )
{
  sndfile::SNDFILE* outfile;
  sndfile::SF_INFO  outinfo;
  sndfile::SNDFILE* sndfile[NINPUTFILES];
  sndfile::SF_INFO    sfinfo[NINPUTFILES];
  sndfile::sf_count_t sfcnt[NINPUTFILES];
  sndfile::sf_count_t nsamples = 4096;
  int nChannels;

  //sfinfo1.channels   = channels;
  //sfinfo1.samplerate = samplerate;
  //sfinfo1.format     = format;
  sndfile[0] = sndfile::sf_open( myOpt._audioFile[0].c_str(), sndfile::SFM_READ, &sfinfo[0] );
  cout << "Reading sound file " << myOpt._audioFile[0].c_str() << endl;
  sndfile[1] = sndfile::sf_open( myOpt._audioFile[1].c_str(), sndfile::SFM_READ, &sfinfo[1] );
  cout << "Reading sound file " << myOpt._audioFile[1].c_str() << endl;
  
  if ( sndfile[0] == NULL ){
    fprintf( stderr, "Could not open file %s.", myOpt._audioFile[0].c_str() );
    return false;
  }
  if ( sndfile[1] == NULL ){
    fprintf( stderr, "Could not open file %s.", myOpt._audioFile[1].c_str() );
    return false;
  }
  if ( sndfile::sf_error( sndfile[0] ) ) {
    fprintf( stderr, "Detect an error in %s.", myOpt._audioFile[0].c_str() );
    sndfile::sf_close( sndfile[0] );
    return false;
  }
  if ( sndfile::sf_error( sndfile[1] ) ) {
    fprintf( stderr, "Detect an error in %s.", myOpt._audioFile[1].c_str() );
    sndfile::sf_close( sndfile[1] );
    return false;
  }

  if( sfinfo[0].samplerate != sfinfo[1].samplerate ){
    fprintf(stderr,"Sampleing rates %d and %d have to be the same\n", 
	    sfinfo[0].samplerate, sfinfo[1].samplerate );
  }
  if( sfinfo[0].channels != sfinfo[1].channels ){
    fprintf(stderr,"Sampleing rates %d and %d have to be the same\n", 
	    sfinfo[0].samplerate, sfinfo[1].samplerate );
  }

  cout << "channels: "   << sfinfo[0].channels   << sfinfo[1].channels   << endl;
  cout << "frames: "     << sfinfo[0].frames     << sfinfo[1].frames     << endl;
  cout << "samplerate: " << sfinfo[0].samplerate << endl;
  
  outinfo.channels = 0;
  for(int fileX=0;fileX<NINPUTFILES;fileX++)
    outinfo.channels += sfinfo[fileX].channels;
  outinfo.format     = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_FLOAT;
  outinfo.samplerate = sfinfo[0].samplerate;
  outfile = sndfile::sf_open( myOpt._outFile.c_str(), sndfile::SFM_WRITE, &outinfo );
  float **bufs = (float **)malloc( NINPUTFILES * sizeof(float *) );
  float *obuf = new float[nsamples*outinfo.channels];

  if( NULL == bufs )
    fprintf(stderr,"could not allocate memory\n");
  for(int fileX=0;fileX<NINPUTFILES;fileX++)
    bufs[fileX] = new float[nsamples*sfinfo[fileX].channels];
  
  while(true){
    int idx1;
    sndfile::sf_count_t maxcnt = 0;

    for(int fileX=0;fileX<NINPUTFILES;fileX++){
      sfcnt[fileX] = sndfile::sf_readf_float( sndfile[fileX], bufs[fileX], nsamples );
      if( sfcnt[fileX] > maxcnt )
	maxcnt = sfcnt[fileX];
    }
    for (int frX=0; frX < nsamples; frX++){
      //idx1 = frX * outinfo.channels;
      int idxY = frX * outinfo.channels;
      int chY  = 0;
      for(int fileY=0;fileY<NINPUTFILES;fileY++){
	int idxX = frX * sfinfo[fileY].channels;
	for(int chX=0;chX<sfinfo[fileY].channels;chX++,chY++){
	  obuf[idxY+chY] = bufs[fileY][idxX+chX];
	}
      }
    }
    sndfile::sf_writef_float( outfile, obuf, nsamples );
    
    if( sfcnt[0] < nsamples || sfcnt[1] < nsamples ){
      break;
    }
  }

  for(int fileX=0;fileX<NINPUTFILES;fileX++)
    delete [] bufs[fileX];
  free(bufs);
  delete [] obuf;
  
  return true;
}

int main( int argc, char **argv )
{
   int opt = -1, longopt_index = 0;
   const char *optstring = "1:2:o:h";
   struct option long_options[] = {
     { "help", 0, 0, 'h' },
     { "audio1", 1, 0, '1' },
     { "audio2", 1, 0, '2' },
     { "output", 1, 0, 'o' },
     { 0, 0, 0, 0 }
   };

   myOption myOpt;

   while ((opt = getopt_long(argc, argv, optstring, long_options, &longopt_index)) != -1) {
    switch (opt) {
    case '1':
      /* getopt signals end of '-' options */
      myOpt._audioFile[0] = optarg;
      break;
    case '2':
      myOpt._audioFile[1] = optarg;
      break;
    case 'o':
      myOpt._outFile = optarg;
    default:
      break;
    }
   }
  
   synthMultiChannelWav( myOpt );
}
