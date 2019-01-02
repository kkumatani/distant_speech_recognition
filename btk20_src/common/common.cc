/**
 * @file common.cc
 * @brief Common operation
 * @author Fabian Jakobs
 */

#include "common/mlist.h"
#include <ctype.h>
#include <list>
#include "common/jexception.h"

using namespace std;

#ifdef WINDOWS
  #include <fcntl.h>

  /* set file open (fopen) mode to binary and not to text */
  int _fmode = _O_BINARY;

#endif

void hello() {
  printf( "\n");
  printf( "BTK2.0: Beyond linear processing\n");
  printf( "\n");
}

void split_list(const String& line, std::list<String>& out)
{
  String tmp = "";
  bool inWord = false;
  bool writeWord = false;
  int  parendepth = 0;

  String::const_iterator iter = line.begin();
  while (iter != line.end()) {

    if (!inWord) {
      if (!isspace(*iter)) inWord=true;
    }
    if (inWord) {
      if (parendepth == 0) {
	if (isspace(*iter)) {
	  inWord = false;
	  writeWord = true;
	} else if (*iter == '{') {
	  parendepth++;
	  iter++;
	  continue;
	} else { // iter not space and not '{'
	  tmp += *iter;
	}
      } else if (parendepth > 0) {
	if (*iter == '{') parendepth++;
	if (*iter == '}') parendepth--;
	if (parendepth == 0) {
	  inWord = false;
	  writeWord = true;
	} else {
	  tmp += *iter;
	}
      }
      if (parendepth < 0)
	throw jparse_error("'}' expected in line %s!", line.c_str());
    }
    if (writeWord) {
      out.push_back(tmp);
      tmp = "";
      writeWord = false;
    }
    iter++;
  }
  if (inWord) {
    out.push_back(tmp);
  }
}

char* date_string() { time_t t=time(0); return (ctime(&t)); }


// fileOpen: open a file for reading/writing
//
FILE* btk_fopen(const char* fileName, const char* mode)
{
  int   retry = 20;
  int   count = 0;
  int   l     = strlen(fileName);
  int   pipe  = 0;
  FILE* fp    = NULL;
  char itfBuffer[500];

  // if (strchr(mode,'w')) itfMakePath(fileName,0755);

  if        (! strcmp( fileName + l - 2, ".Z")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"zcat %s",       fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"compress > %s", fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;

  } else if (! strcmp( fileName + l - 3, ".gz")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"gzip -d -c  '%s'", fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"gzip -c >  '%s'",  fileName);
    else if (! strcmp( mode, "a")) sprintf(itfBuffer,"gzip -c >> '%s'",  fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;

  } else if (! strcmp( fileName + l - 4, ".bz2")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"bzip2 -cd    '%s'", fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"bzip2 -cz >  '%s'", fileName);
    else if (! strcmp( mode, "a")) sprintf(itfBuffer,"bzip2 -cz >> '%s'", fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;
  }

  while (count <= retry) {
    if (! (fp = ( pipe) ? popen( itfBuffer, mode) :
                          fopen( fileName,  mode))) {
      sleep(5); count++;
    }
    else break;
  }
  if ( count > retry)
    throw jio_error("'fileOpen' failed for %s.\n", fileName);

  return fp;
}

// fileClose:  close previously openend file
//
void btk_fclose(const char* fileName, FILE* fp)
{
  int l = strlen(fileName);

  fflush( fp);

  if      (! strcmp( fileName + l - 2, ".Z"))   pclose( fp);
  else if (! strcmp( fileName + l - 3, ".gz"))  pclose( fp);
  else if (! strcmp( fileName + l - 4, ".bz2")) pclose( fp);
  else                                          fclose( fp);
}

static int        line_;
static const char* file_;

int setErrLine_(int line, const char* file)
{
  line_ = line; file_ = file;
  
  return 1;
}

void warnMsg_(const char* message, ...)
{
   va_list ap;    /* pointer to unnamed args */
   FILE* f = stdout;

   fflush(f);    /* flush pending output */
   va_start(ap, message);
   fprintf(f, " >>> Warning: ");
   vfprintf(f, message, ap);
   va_end(ap);
   fprintf(f,"\n");
   fprintf(f," >>>          at line %d of \'%s\'\n", line_, file_);
   fprintf(f," >>> Continuing ... \n\n");
}
