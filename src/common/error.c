
#include "common.h"
#include <stdarg.h>
#include <string.h>

/* ------------------------------------------------------------------------
    Local Variables
   ------------------------------------------------------------------------ */

#define MAXERRACCU 8192
char error_msg_[MAXERRACCU];
static char *errorFileName;
static int   errorLine;
static int   errorType;
static int   errorMode;

/* ------------------------------------------------------------------------
   msg_handler_ptr
   ------------------------------------------------------------------------ */

MsgHandler* msg_handler_ptr(char* file, int line, int type, int mode)
{
  errorFileName = file;
  errorLine     = line;
  errorType     = type;
  errorMode     = mode;
  return &error_msg_handler;
}

/* ------------------------------------------------------------------------
    error_msg_handler
   ------------------------------------------------------------------------ */

int error_msg_handler( char *format, ... )
{

  va_list  ap;
  char     buf[MAXERRACCU] = "", *format2 = format;

  if ( format) {
    va_start(ap,format);
    vsnprintf(buf, MAXERRACCU, format2, ap);
    va_end(ap);
  }

  snprintf(error_msg_, MAXERRACCU, "(%s,%d): %s", errorFileName, errorLine, buf);
  return 0;
}

char* get_error_msg(void) {
  return error_msg_;
}

/*
 *  @brief initialize an error handler
 */

int init_error_handler(void)
{
  static int errorInitialized = 0;

  if (! errorInitialized) {
    errorInitialized = 1;
    sprintf(error_msg_, "No error");
  }
  return 0;
}
