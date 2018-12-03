/*
 * @file utils.cc
 * @brief Utility functions
 * @author John McDonough
 */

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>

#include "utils.h"

// ------------------------------------------------------------------------
//  fileLock      opens a file handle and locks it for writing
// ------------------------------------------------------------------------

int fileLock(const char* name, int mode)
{
  struct flock lock;
  int          fd  = -1;
  int          ret = -1;

  while ( ret < 0) {
    if ((fd = open( name, mode, 0666)) < 0) {
      printf("<ITF,COF>Can't open file %s.\n",name);
      return -1;
    }
    lock.l_type   = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start  = 0;
    lock.l_len    = 0;

    if ((ret = fcntl( fd, F_SETLK, &lock)) < 0) { close(fd); sleep(1); }
  }
  return fd;
}

// ------------------------------------------------------------------------
//  fileUnlock   removes the write lock and closes the file
// ------------------------------------------------------------------------

static void fileUnlock( int fd)
{
  struct flock lock;

  lock.l_type   = F_UNLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start  = 0;
  lock.l_len    = 0;

  if (fcntl(fd,F_SETLK,&lock) < 0) {
    printf("<ITF,CUL>Could not unlock file.\n");
  }
}

const char* fgets(const char* name, const char* log, int tstamp)
{
  FILE* fp       =  NULL;
  int   fd       =  0;
  char c;
  static char  line[1024];
  static const char* EmptyString = "";

  if ((fd = fileLock( name, O_RDWR)) < 0) return EmptyString;
  else fp = fdopen(fd,"r+");

  // find the first line in the file which is not empty and has no
  // preceeding comment character #

  do {
    do { c = getc(fp); } while ((c=='\n' || c=='\t' || c==' ') && !feof(fp));

    if (feof(fp)) break;
    if (c != '#') break;

    do { c = getc(fp); } while ( c != '\n' && !feof(fp));
  } while (1);

  if (! feof(fp)) {
    int i;

    fseek(fp, (long int)-1,SEEK_CUR);

  fputc( (int)'#',fp); fflush(fp);

    line[i=0] = c;
    do { c = getc(fp);
         assert( i < 1024);
         if ( ! feof(fp) && c != '\n') line[++i] = c;
    } while ( ! feof(fp) && c != '\n');
    line[++i] = 0;

    fileUnlock(fd);  fclose(fp);

    return line;
  }

  fileUnlock(fd);  fclose(fp);

  return EmptyString;
}
