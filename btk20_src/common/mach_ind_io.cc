#include <stdio.h>
#include <stdlib.h>
#include <string.h>                     /* for memcpy, which is used instead of bcopy */
#include <assert.h>
#include "common/common.h"
#include "common/mach_ind_io.h"
#include "common/jexception.h"


/*#############################################################################################################################
 #
 #  LOCAL DEFINITIONS  (Stuff for reading/writing canonical binary files on any type of machine)
 #
 ############################################################################################################################*/

#define RT_SUN   1    /* RT or SUN machine type */
#define DEC      2    /* DEC machine type         (= PMAX) ... everybody knows what a DEC is! */
#define VAX      3    /* VAX machine type */

union unionT {        /* union for type conversions */
    float         fval;      /*    4-byte float */
    int           ival;      /*    4-byte int */
    unsigned char cval [4];  /*    4-byte string */
};

/*#############################################################################################################################
 #
 #  LOCAL VARIABLES
 #
 ############################################################################################################################*/

static int machine = 0;                 /* what type of machine are we currently running on? */


/*#############################################################################################################################
 #
 #  LOCAL MACROS
 #
 ############################################################################################################################*/

/*-----------------------------------------------------------------------------------------------------------------------------
 | Some useful macros:
 |
 |  SWAP_SHORT:       simply swaps high and low bytes of a two byte short
 |  REVERSE_4BYTE:    reverses order of 4byte sequence (float or int storage)
 |  FLOAT_VAX_TO_STD: swaps first and second bytes, swaps third and fourth bytes, subtracts
 |                        one from resulting first byte, unless byte is already zero
 |  FLOAT_STD_TO_VAX: swaps first and second bytes, swaps third and fourth bytes, adds one
 |        to resulting second byte (with no carry into first byte)
 |
 | NOTE:    I (arthurem) still don't understand these VAX conversions, but I made sure these macros are
 |      doing the exact same thing that jmt's old routines were doing for the VAX.
 |
 | COMMENT: I (Tilo Sloboda) didn't understand them either (sigh!).
 |
 +---------------------------------------------------------------------------------------------------------------------------*/

#define SWAP_SHORT(x)    (short)((short)(0xFF00U & (short)((short)(x)<<8)) | (short)(0x00FFU & (short)((short)(x)>>8)))

#define REVERSE_4BYTE(x)  ((((x)<<24)&0xFF000000U) | (((x)<<8)&0x00FF0000U) | \
         (((x)>>8) & 0x0000FF00U) | (((x)>>24)&0x000000FFU))


#define FLOAT_VAX_TO_STD(x)  (((((x)<<8)&0xFF00FF00U)  | (((x)>>8)&0x00FF00FFU)  ) \
           - (((x)&0x0000FF00U) ? 0x00000001U : 0 )   )

#define FLOAT_STD_TO_VAX(x)  (((((x)<<8)+0x00000100)&0xFF00FF00U) | (((x)>>8)&0x00FF00FFU))


/*#############################################################################################################################
 #
 #  GLOBAL FUNCTIONS
 #
 ############################################################################################################################*/

/*=============================================================================================================================
 | init_mach_ind_io:  Figures out what type of machine we're running on.  Saves machine indicator in static variable 
 |         "machine".  Also checks that float, short, and long are correct number of bytes on current machine.
 |
 | PARAMETERS: none.
 |
 | RETURNS:    none.
 |
 | NOTE:       aborts the program if the machine type is unknown or the size of the basic types is not as expected!
 |             (because we can't assure that the files are read/written correctly)
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified so returns success/error flag
 |   16 Jul 93  sloboda   "unknown machine type" and unexpected size of basic types are fatal errors!
 |                        (because we can't assure that the files are read/written correctly)
 +===========================================================================================================================*/
void init_mach_ind_io( void )
{
  union unionT u;

  if (sizeof(float) != 4)
  {
    fprintf (stderr,"\n\n >>> FATAL ERROR: size of float = %d\n", (int)sizeof(float));
    exit(-1);
  }
  if (sizeof(int) != 4)
  {
    fprintf (stderr,"\n\n >>> FATAL ERROR: size of int = %d\n", (int)sizeof(int));
    exit(-1);
  }
  if (sizeof(short) != 2)
  {
    fprintf (stderr,"\n\n >>> FATAL ERROR: size of short = %d\n", (int)sizeof(short));
    exit(-1);
  }

  u.fval = 123.456;        /* assign a known float.  How is it represented? */
  if      (u.cval[3] == 121)  machine = RT_SUN;  /* if 121 is in byte #3, we're on a RT or SUN */
  else if (u.cval[0] == 121)  machine = DEC;  /* if 121 is in byte #0, we're on a DEC */
  else if (u.cval[2] == 121)  machine = VAX;  /* if 121 is in byte #2, we're on a VAX */
  else
  {    /* ERROR, couldn't determine machine type */
    fprintf(stderr, "\n\n >>> FATAL ERROR: unknown float representation on current machine\n");
    exit(-1);
  }
}

/*=============================================================================================================================
 | float_to_ubyte:  converts a float in the range [0..1] into a floatbyte in the range [0..255]
 |
 | PARAMETERS:
 |   f  = float to write, assumed to be between 0.0 and 1.0.
 |
 | RETURNS: floatbyte
 |
 | HISTORY:
 |    2 Aug 93   sloboda   Created.
 +===========================================================================================================================*/
UBYTE float_to_ubyte( float f )
{
  if (f < 0.0) 
       { fprintf (stderr, "\n\n >>> WARNING in float_to_ubyte: %f being truncated to 0.0\n", f);
         f = 0.0;
       }
  else if (f > 1.0)
       { fprintf (stderr, "\n\n >>> WARNING in float_to_ubyte: %f being truncated to 1.0\n", f);
         f = 1.0;
       }

  return( (UBYTE) (f * 255.0 + 0.5) );              /* convert 0..1 to 0..255 */
}

/*=============================================================================================================================
 | ubyte_to_float:  converts a floatbyte in the range [0..255] into a float in the range [0..1]
 |
 | PARAMETERS:
 |   u  = floatbyte
 |
 | RETURNS: float
 |
 | HISTORY:
 |    2 Aug 93   sloboda   Created.
 +===========================================================================================================================*/
float ubyte_to_float( UBYTE u )
{
  return( (float) ((float)u / 255.0) );                    /* convert 0..255 to 0..1 */
}

/*=============================================================================================================================
 | read_float:  Reads a float from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |
 | RETURNS:
 |   The next float in the file.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
float read_float ( FILE *fp )
{
  union unionT u,v;

  u.fval=0.0;
  fread (&u.fval, sizeof(float), 1, fp);    /* read the next 4 bytes */

  switch (machine) {
    case RT_SUN: v.fval = u.fval;       /* RT or SUN:  already in correct format */
                 break;

    case DEC:    v.ival = REVERSE_4BYTE(u.ival);        /* DEC:  reverse the bytes */
                 break;

    case VAX:    v.ival = FLOAT_STD_TO_VAX(u.ival);  /* VAX: shuffle the bytes */
                 break;

    default:     throw jio_error("Error: unknown machine type %d.", machine);
   }
  return(v.fval);
}

/*=============================================================================================================================
 | read_floats:  Reads floats from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp      = binary file pointer.  (Assumed to be already open.)
 |   whereto = pointer to beginning (float) location to start storing floats that will be read (already allocated)
 |   count   = number of floats to read and store
 |
 | RETURNS:
 |   Number of floats successfully read
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to read any number of floats, and to return number read
 +===========================================================================================================================*/
int read_floats( FILE *fp, float *whereto, int count )
{
  union unionT v,u;    /* union variables for converting values */
  float *ptr = whereto;    /* temp pointer to step through destination locations */
  int i;      /* counter for number of floats converted */
  int number_read;    /* number read from fread */

  number_read = fread (whereto, sizeof(float), count, fp);    /* read all the floats */

  for (i=0; i<number_read; i++)
  {
    u.fval = *ptr;

    switch (machine)
    {
      case RT_SUN: v.fval = u.fval;       /* RT or SUN:  already in correct format */
      break;

      case DEC:    v.ival = REVERSE_4BYTE(u.ival);    /* DEC:  reverse the bytes */
      break;

      case VAX:    v.ival = FLOAT_STD_TO_VAX(u.ival);  /* VAX: shuffle the bytes */
      break;

      default:     throw jio_error("Error: unknown machine type %d.", machine);

      v.fval = u.fval;
    }

    *ptr = v.fval;
    ptr++;

  }
  return(number_read);
}

/*=============================================================================================================================
 | read_floatbytes:  Reads floats from a file, in which each float (0..1) is represented by a byte (0..255).
 |
 | PARAMETERS:
 |   fp      = binary file pointer.  (Assumed to be already open.)
 |   whereto = pointer to beginning (float) location to start storing floats that will be read (already allocated)
 |   count   = number of floats to read and store
 |
 | RETURNS:
 |   Number of floats successfully read
 |
 | HISTORY:
 |  15 Jul 93  sloboda   Created.
 |  24.Nov 93  sloboda   added extra parentheses
 +===========================================================================================================================*/
int read_floatbytes( FILE *fp, float *whereto, int count )
{
    UBYTE *tmpA;    /* temporary array to hold all data for conversion */
    UBYTE *fromP;    /* temp pointer to step through source array */
    float  *toP = whereto;      /* temp pointer to step through target array */
    int     i;      /* counter for number of floats converted */
    int     number_read;  /* number read from fread */


  /* If bad input, don't bother */
  if (count <= 0) return(0);

  /* Get a temp array to modify the byte order */
  tmpA = (UBYTE *)malloc((int)sizeof(UBYTE)*count);
  if (tmpA == NULL)
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in read_floatbytes\n");
    return(0);
  }

  /* Start at beginning of temp array */
  fromP = tmpA;
  number_read = fread (fromP, sizeof(UBYTE), count, fp);  /* read all the floatbytes */

  for (i=0; i<number_read; i++)
    *toP++ = (float) ((float)(*fromP++) / 255.0);      /* convert 0..255 to 0..1 */

  free(tmpA);
  return(number_read);
}

/*=============================================================================================================================
 | read_floatbyte:  Reads a float from a file, in which a float (0..1) is represented by a byte (0..255).
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |
 | RETURNS:
 |   The next float in the file.
 |
 | HISTORY:
 |   21.Jan 93   jmt   Created.
 +===========================================================================================================================*/
float read_floatbyte( FILE *fp )
{
  UBYTE    b;
  float     val;

  fread (&b, 1, 1, fp);          /* read the next byte */
  val = (float) b / 255.0;        /* convert 0..255 to 0..1 */
  return (val);            /* return float */
}

/*=============================================================================================================================
 | read_int:  Reads an integer from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |
 | RETURNS:
 |   The next integer in the file.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
int read_int( FILE *fp )
{
  int   i=0;

  fread (&i, sizeof(int), 1, fp);	// read the next 4 bytes
  switch (machine) {
  case RT_SUN:
    return(i);				// RT or SUN: already in correct format

  case DEC:
  case VAX:
    return( REVERSE_4BYTE(i) );		// DEC or VAX: reverse the bytes

  default:
    throw jio_error("Error: unknown machine type %d.", machine);
  }

  return 0;
}

/*=============================================================================================================================
 | read_ints:  Reads integers (4 bytes) from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp      = binary file pointer.  (Assumed to be already open.)
 |   whereto = pointer to beginning (integer) location to start storing ints that will be read (already allocated)
 |   count   = number of ints to read and store
 |
 | RETURNS:
 |   Number of ints successfully read
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to read any number of ints, and to return number read
 +===========================================================================================================================*/
int read_ints( FILE *fp, int *whereto, int count )
{
  union unionT v,u;            /* union variables for converting values */
  int         *ptr = whereto;    /* temp pointer to step through destination locations */
  int          i;      /* counter for number of ints converted */
  int          number_read;    /* number read from fread */

  number_read = fread (whereto, sizeof(int), count, fp);    /* read all the floats */

  for (i=0; i<number_read; i++)
  {
    u.ival = *ptr;

    switch (machine)
    {
      case RT_SUN:   v.ival = u.ival;       /* RT or SUN:  already in correct format */
            break;

      case DEC:            /* DEC or VAX:  reverse the bytes */
      case VAX:     v.ival = REVERSE_4BYTE(u.ival);
            break;

      default:
        throw jio_error("Error: unknown machine type %d.", machine);
    }
    *ptr = v.ival;
    ptr++;
  }
  return(number_read);
}

/*=============================================================================================================================
 | read_short:  Reads a short integer from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |
 | RETURNS:
 |   The next short integer in the file.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
short read_short( FILE *fp )
{
  short  s=0;

  fread (&s, sizeof(short), 1, fp);	// read the next 2 bytes

  switch (machine) {
  case RT_SUN:
    return(s);       			// RT or SUN: already in correct format

  case DEC:
  case VAX:
    return(SWAP_SHORT(s));		// DEC or VAX: reverse the bytes

  default:
    throw jio_error("Error: unknown machine type %d.", machine);
  }

  return 0;
}

/*=============================================================================================================================
 | read_shorts:  Reads shorts (2 bytes) from a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp      = binary file pointer.  (Assumed to be already open.)
 |   whereto = pointer to beginning (short) location to start storing shorts that will be read (already allocated)
 |   count   = number of shorts to read and store
 |
 | RETURNS:
 |   Number of shorts successfully read
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to read any number of shorts, and to return number read
 +===========================================================================================================================*/
int read_shorts(FILE *fp, short *whereto, int count)
{
  union unionT u;            /* union variables for converting values */
  short       *ptr = whereto;    /* temp pointer to step through destination locations */
  int         i;      /* counter for number of shorts converted */
  int         number_read;    /* number read from fread */

  number_read = fread (whereto, sizeof(short), count, fp);    /* read all the floats */

  for (i=0; i<number_read; i++)
  {
    u.ival = *ptr;

    switch (machine)
    {
      case RT_SUN:              /* RT or SUN:  already in correct format */
            break;

      case DEC:            /* DEC or VAX:  reverse the bytes */
      case VAX:     *ptr = SWAP_SHORT(*ptr);
            break;

      default:
        throw jio_error("Error: unknown machine type %d.", machine);
    }
    ptr++;
  }
  return(number_read);
}

/*=============================================================================================================================
 | read_string:  Reads a string from a binary file, where it is prefixed with its length and terminated with EOS.
 |
 | PARAMETERS:
 |   f   = binary file to read from.  The file must be already open.
 |   str = address of destination string.
 |   
 | RETURNS:
 |    1 if successful, 0 if error occurs
 | HISTORY:
 |   12.Oct 89   jmt       Created.
 |   19.Jan 90   jmt       Use read_short, not fread, to read the length.
 |   14.Jun 91   arthurem  modified to give return value
 |   16 Jul 93   sloboda   changed return value for the case of an empty string.
 +===========================================================================================================================*/
int read_string(FILE *fp, char *str)
{
  int    status;
  short  len;

  len = read_short(fp);        /* read length of string */
            /* len==0  for an empty string */
  status = fread(str, len+1, 1, fp);    /* read that many characters (plus EOS) into string */
  if (status == 0) return(0);
  return(1);
}

/*=============================================================================================================================
 | write_float:  Writes a float into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |   f  = float to write.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
void write_float (FILE *fp, float f)
{
    union unionT u,v;

  u.fval = f;
  switch (machine) {
    case RT_SUN: v.fval = u.fval;                    /* RT or SUN:  already in correct format */
                 break;

    case DEC:    v.ival = REVERSE_4BYTE(u.ival);  /* DEC:  reverse the bytes */
                 break;

    case VAX:    v.ival = FLOAT_VAX_TO_STD(u.ival);  /* VAX:  shuffle the bytes */
                 break;

    default:    throw jio_error("Error: unknown machine type %d.", machine);
  }

  if ( fwrite( &v.fval , sizeof(float), 1, fp) != 1)
    { fprintf(stderr, "\n\n >>> FATAL ERROR: couldn't write to file in write_float()\n");
      exit(-1);
    }
}

/*=============================================================================================================================
 | write_floats:  Writes floats into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp        = binary file pointer.  (Assumed to be already open.)
 |   wherefrom = pointer to beginning (float) location to start writing floats from
 |   count     = number of floats to write to file
 |
 | RETURNS:
 |   Number of floats successfully written
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to write any number of floats, and to return number written
 |   14.Aug 93  sloboda   replaced bcopy() by memcpy() which is defined in both ANSI C and K&R C
 |   24.Nov 93  sloboda   removed superfluous memcpy() call
 |    4.Mar 96  maier     in case of memory shortage write single floats
 +===========================================================================================================================*/
int write_floats (FILE *fp, float *wherefrom, int count)
{
    union unionT u,v;
    float       *tmpA;      /* temporary array to hold all data for conversion */
    float       *toP;                   /* temp pointer to step through wherefrom array */
    float       *fromP;                 /* temp pointer to step through temp array */
    int          i;      /* counter for number of floats converted */
    int          number_written = 0;  /* number read from fread */
    float       f;

  /* If bad input, don't bother */
  if (count <= 0) return(0);

  /* Get a temp array to modify the byte order */
  tmpA = (float *)malloc(sizeof(float)*count);
  if (tmpA == NULL) {
    fprintf(stderr, "\n\n >>> WARN: Couldn't malloc more space in write_floats\n");
    fprintf(stderr, " >>> WARN: Will write single floats to file\n");
    toP = &f;
  }
  else  toP = tmpA;

  /* Start at beginning of temp array */
  fromP = wherefrom;

  for (i=0; i<count; i++, fromP++)
  {
    u.fval = *fromP;  /* Put value in union structure */

    switch (machine)
    {
      case RT_SUN:   v.fval = u.fval;       /* RT or SUN:  already in correct format */
      break;

      case DEC:     v.ival = REVERSE_4BYTE(u.ival);    /* DEC:  reverse the bytes */
      break;

      case VAX:      v.ival = FLOAT_VAX_TO_STD(u.ival);  /* VAX: shuffle the bytes */
      break;

      default:       throw jio_error("Error: unknown machine type %d.", machine);

    }
    *toP = v.fval;  /* Write new value from union structure back into temp array */
    if (tmpA) toP++;
    else number_written += fwrite (toP, sizeof(float), 1, fp);    /* write one float */
  }

  if (tmpA) {
    number_written = fwrite (tmpA, sizeof(float), count, fp);    /* write all the floats */
    free(tmpA);
  }
  return(number_written);
}

/*=============================================================================================================================
 | write_floatbyte:  Writes a float into a file, representing a float (0..1) by a byte (0..255).
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |   f  = float to write, assumed to be between 0.0 and 1.0.
 |
 | HISTORY:
 |   21 Jan 93   jmt       Created.
 |   15 Jul 93   sloboda   minor change
 +===========================================================================================================================*/
void write_floatbyte (FILE *fp, float f)
{
     UBYTE  b;

  if (f < 0.0)
       { fprintf(stderr, "\n\n >>> WARNING in write_floatbyte: %f being truncated to 0.0\n", f);
         f = 0.0;
       }
  else if (f > 1.0)
       { fprintf(stderr, "\n\n >>> WARNING in write_floatbyte: %f being truncated to 1.0\n", f);
         f = 1.0;
       }

  b = (UBYTE) (f * 255.0 + 0.5);              /* convert 0..1 to 0..255 */
  fwrite (&b, sizeof(UBYTE), 1, fp);        /* write byte */
}

/*=============================================================================================================================
 | write_floatbytes:  Writes floats into a file, representing each float (0..1) by a byte (0..255).
 |
 | PARAMETERS:
 |   fp        = binary file pointer.  (Assumed to be already open.)
 |   wherefrom = pointer to beginning (float) location to start writing floats from
 |   count     = number of floats to write to file
 |
 | RETURNS:
 |   Number of floatbytes successfully written
 |
 | HISTORY:
 |   15 Jul 93   sloboda   created from Joe's routine "write_floatbyte()"
 |   14.Aug 93   sloboda   replaced bcopy() by memcpy() which is defined in both ANSI C and K&R C
 |   24.Nov 93   sloboda   removed memcpy() -- it was superfluous! not a bug as I thought first... schwitz!
 +===========================================================================================================================*/
int write_floatbytes(FILE *fp, float *wherefrom, int count)
{
    UBYTE  *tmpA;      /* temporary array to hold all data for conversion */
    UBYTE  *toP;            /* temp pointer to step through target array */
    float  *fromP = wherefrom;          /* temp pointer to step through source array */
    int     i;                          /* counter for number of floats converted */
    int     number_written;    /* number read from fread */
    int     wcount = 0;
    int     wcmax  = 5;

  /* If bad input, don't bother */
  if (count <= 0) return(0);

  /* Get a temp array to store the converted floatbytes */
  tmpA = (UBYTE *)malloc(sizeof(UBYTE)*count);
  if (tmpA == NULL)
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in write_floatbytes\n");
    return(0);
  }

  /* Start at beginning of temp array */
  toP = tmpA;

  for (i=0; i<count; i++)
  {
    if (*fromP < 0.0)
       { if (wcount < wcmax) fprintf(stderr, "\n\n >>> WARNING in write_floatbytes: %f being truncated to 0.0\n", *fromP);
         *fromP = 0.0; wcount++;
       }
    else if (*fromP > 1.0)
       { if (wcount < wcmax) fprintf(stderr, "\n\n >>> WARNING in write_floatbytes: %f being truncated to 1.0\n", *fromP);
         *fromP = 1.0; wcount++;
       }

    *toP++ = (UBYTE) (*fromP++ * 255.0 + 0.5);                  /* convert 0..1 to 0..255 and copy to tmpA */
  }
  if (wcount >= wcmax)
    fprintf(stderr, "\n\n >>> WARNING in write_floatbytes: ... and %d other coefficients truncated\n",wcount-wcmax+1);

  number_written = fwrite (tmpA, sizeof(UBYTE), count, fp);  /* write all the floatbytes from tmpA to file */

  free(tmpA);
  return(number_written);
}

/*=============================================================================================================================
 | write_int:  Writes an integer into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |   i  = integer to write.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
void write_int( FILE *fp, int i)
{
    union unionT u;

  switch (machine) {
    case RT_SUN: u.ival = i;        /* RT or SUN:  already in correct format */
                 break;
    case DEC:
    case VAX:
                 u.ival = REVERSE_4BYTE(i);          /* DEC or VAX:  reverse the bytes */
                 break;

    default:     throw jio_error("Error: unknown machine type %d.", machine);
  }
  fwrite (&u.ival, sizeof(int), 1, fp);
}

/*=============================================================================================================================
 | write_ints:  Writes ints into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp        = binary file pointer.  (Assumed to be already open.)
 |   wherefrom = pointer to beginning (int) location to start writing ints from
 |   count     = number of ints to write to file
 |
 | RETURNS:
 |   Number of ints successfully written
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to write any number of ints, and to return number written
 |   14.Aug 93  sloboda   replaced bcopy() by memcpy() which is defined in both ANSI C and K&R C
 |   24.Nov 93  sloboda   removed superfluous call of memcpy()
 |    6.Dec 93  sloboda   bug removed for HPs and SUNs
 +===========================================================================================================================*/
int write_ints(FILE *fp, int *wherefrom, int count)
{
    int *tmpA;      /* temporary array to hold all data for conversion */
    int *toP;      /* temp pointer to step through temp array */
    int *fromP;                 /* temp pointer to step through wherefrom array */
    int  i;      /* counter for number of floats converted */
    int  number_written;  /* number read from fread */

  /* If bad input, don't bother */
  if (count <= 0) return(0);

  /* Get a temp array to modify the byte order */
  tmpA = (int *)malloc(sizeof(int)*count);
  if (tmpA == NULL)
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in write_ints\n");
    return(0);
  }

  /* start at the beginning of temp array */
  toP = tmpA;
  fromP = wherefrom;

  for (i=0; i<count; i++, fromP++, toP++)
    switch (machine)
    {
      case RT_SUN:   *toP = *fromP;      /* RT or SUN:  already in correct format */
      break;

      case DEC:            /* DEC or VAX:  reverse the bytes */
      case VAX:     *toP = REVERSE_4BYTE(*fromP);
      break;

      default:      throw jio_error("Error: unknown machine type %d.", machine);
    }

  number_written = fwrite (tmpA, sizeof(int), count, fp);    /* write all the ints */

  free(tmpA);
  return(number_written);
}

/*=============================================================================================================================
 | write_short:  Writes a short integer into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp = binary file pointer.  (Assumed to be already open.)
 |   s  = short integer to write.
 |
 | HISTORY:
 |   20.Oct 89  jmt  Created.
 +===========================================================================================================================*/
void write_short(FILE *fp, short s)
{
  short  t;

  switch (machine) {
    case RT_SUN: t = s;                /* RT or SUN:  already in correct format */
                 break;

    case DEC:
    case VAX:    t = SWAP_SHORT(s);      /* DEC or VAX:  reverse the bytes */
         break;

    default:     throw jio_error("Error: unknown machine type %d.", machine);
  }
  fwrite (&t, sizeof(short), 1, fp);
}

/*=============================================================================================================================
 | write_shorts:  Writes shorts into a binary file.  Works on any machine.
 |
 | PARAMETERS:
 |   fp        = binary file pointer.  (Assumed to be already open.)
 |   wherefrom = pointer to beginning (short) location to start writing shorts from
 |   count     = number of shorts to write to file
 |
 | RETURNS:
 |   Number of shorts successfully written
 |
 | HISTORY:
 |   20.Oct 89  jmt       Created.
 |   14.Jun 91  arthurem  Modified to write any number of shorts, and to return number written
 |   16 Jul 93  sloboda   changed header to ANSI-C and changed type of count to int.
 |   14.Aug 93  sloboda   replaced bcopy() by memcpy() which is defined in both ANSI C and K&R C
 |   24.Nov 93  sloboda   removed superfluous call of memcpy()
 |    6.Dec 93  sloboda   bug removed for HPs and SUNs
 +===========================================================================================================================*/
int write_shorts(FILE *fp, short *wherefrom, int count)
{
  short *tmpA;      /* temporary array to hold all data for conversion */
  short *fromP;                       /* temp pointer to step through wherefrom array */
  short *toP;                         /* temp pointer to step through temp array */
  int    i;                           /* counter for number of floats converted */
  int    number_written;    /* number read from fread */

  /* If bad input, don't bother */
  if (count <= 0) return(0);

  /* Get a temp array to modify the byte order */
  tmpA = (short *)malloc(sizeof(short)*count);
  if (tmpA == NULL)
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in write_shorts\n");
    return(0);
  }

  /* Start at beginning of temp array */
  toP = tmpA;
  fromP = wherefrom;

  for (i=0; i<count; i++, fromP++, toP++)
    switch (machine)
    {
      case RT_SUN:  *toP = *fromP;      /* RT or SUN:  already in correct format */
      break;
      case DEC:            /* DEC or VAX:  reverse the bytes */
      case VAX:     *toP = SWAP_SHORT(*fromP);
      break;

      default:      throw jio_error("Error: unknown machine type %d.", machine);
    }

  number_written = fwrite (tmpA, sizeof(short), count, fp);    /* write all the shorts */

  free(tmpA);
  return(number_written);
}

/*========================================================================================================================== */
/* change_short, change_int, change_float: This routines will change one short, int or float, respectively, into machine     */
/*  independent format. This routines can be used to convert just one data item, or to convert a structure to make it        */
/*  suitable for socket connections across system boundaries.                                                                */
/*  Nov 1994   tk created                                                                                                    */
/*========================================================================================================================== */
int change_short(short *x)
{
  short temp;
  switch(machine){
  case RT_SUN:      break;                              /* RT or SUN:  already in correct format */
  case DEC:                                             /* DEC or VAX: reverse the bytes         */
  case VAX:         temp = SWAP_SHORT(*x); *x = temp; break;
  default:          fprintf(stderr, "\n\n >>> FATAL ERROR: unknown machine type %d\n", machine); 
    exit(-1);
  }
  return 0;
}

int change_int(int *x)
{
  int temp;
  switch(machine){
  case RT_SUN:      break;                              /* RT or SUN:  already in correct format */
  case DEC:                                             /* DEC or VAX: reverse the bytes         */
  case VAX:         temp = REVERSE_4BYTE(*x); *x = temp; break;
  default:          fprintf(stderr, "\n\n >>> FATAL ERROR: unknown machine type %d\n", machine); 
    exit(-1);
  }
  return 0;
}

int change_float(float *x)
{
  union unionT u,v;
  u.fval = v.fval = *x;
  switch(machine){
  case RT_SUN:      break;                                       /* RT or SUN:  already in correct format */
  case DEC:         v.ival = REVERSE_4BYTE(u.ival); break;       /* DEC: reverse the bytes         */
  case VAX:         v.ival = FLOAT_STD_TO_VAX(u.ival); break;    /* VAX: shuffle the bytes         */
  default:          fprintf(stderr, "\n\n >>> FATAL ERROR: unknown machine type %d\n", machine);
    exit(-1);
  }
  *x = v.fval;
  return 0;
}


/*========================================================================================================================== */
/*  short_memorychange: Changes an array of shorts into machine-independent format in memory. After this change,             */
/*                      the data can be sent by a socket to another machine and can be re-changed there by another           */
/*                      call to short_memorychange(). write_shorts() can be simulated with short_memorychange() and          */
/*                      a subsequent fwrite() system call; read_shorts() with fread() and subsequent short_memorychange()    */
/*                                                                                                                           */
/*  Nov 1994    tk created                                                                                                   */
/*========================================================================================================================== */

int short_memorychange(short *buf, int bufN)
{
  switch(machine){
  case RT_SUN:      break;                              /* RT or SUN:  already in correct format */
  case DEC:                                             /* DEC or VAX: reverse the bytes         */
  case VAX:         buf_byte_swap(buf, bufN); break;
  default:          throw jio_error("Error: unknown machine type %d.", machine);
  }
  return 0;
}

/*========================================================================================================================== */
/* int_memorychange(): Changes an array of ints into machine-independent format in memory. See short_memorychange().         */
/*========================================================================================================================== */
int int_memorychange(int *buf, int bufN)
{
  int i,temp;
  if (machine == RT_SUN) return 0;
  for (i=0; i < bufN; i++){
    switch(machine){
    case DEC:
    case VAX:       temp = REVERSE_4BYTE(*buf); break;
    default:        throw jio_error("Error: unknown machine type %d.", machine);
    }
    *buf = temp;
    buf++;
  }
  return 0;
}

/*========================================================================================================================== */
/* float_memorychange(): Changes an array of floats into machine-independent format in memory. See short_memorychange().     */
/*========================================================================================================================== */
int float_memorychange(float *buf, int bufN)
{
  int i;
  union unionT u,v;

  if (machine == RT_SUN) return 0;
  for (i=0; i < bufN; i++){
    u.fval = *buf;
    switch(machine){
    case DEC:    v.ival = REVERSE_4BYTE(u.ival);        /* DEC:  reverse the bytes */
                 break;
    case VAX:    v.ival = FLOAT_STD_TO_VAX(u.ival);  /* VAX: shuffle the bytes */
                 break;
    default:     throw jio_error("Error: unknown machine type %d.", machine);
    }
    *buf = v.fval;
    buf++;
  }
  return 0;
}


/*=============================================================================================================================
 | write_string:  Writes a string into a binary file, prefixed with its length and terminated with EOS.
 |
 | PARAMETERS:
 |   f   = binary file to write into.  The file must be already open.
 |   str = string to write.
 |
 | RETURNS:
 |    1 if successful, 0 if not
 | HISTORY:
 |   12.Oct 89   jmt       Created.
 |   19.Jan 90   jmt       Use write_shorts, not fwrite, to write the length.
 |   14.Jun 91   arthurem  added return value
 |   16 Jul 93   sloboda   changed return value for the case of an empty string.
 +===========================================================================================================================*/
int write_string(FILE* fp, const char* str)
{
     int    status;
     short  len;

  len = (short) strlen (str);      /* get length of string */
  status = write_shorts (fp, &len, 1);    /* write the length */
  if (status == 0) return(0);

  status = fwrite (str, len+1, 1, fp);    /* write the string terminated with EOS. */
  if (status == 0) return(0);
  return(1);
}


/*=============================================================================================================================
 | ROUTINES FOR SCALED VECTORS:
 |
 |   With these routines you can read/write a block of vectors from/to file, the information in the file is compacted.
 |
 |   In the file the coefficients are stored as so called scaled-floatbytes,
 |   where each "channel" (or you may say dimension) is scaled separately by the min,max value as follows:
 |
 |        scaled_value = (value - min) / (max-min);      this transforms the interval [min,max] to [0,1]
 |   and:
 |        value = ((max-min) * scaled_value) + min;      this transforms the interval [0,1] to [min,max]
 |
 |   These scaled values are then stored as regular floatbytes -- of course we need to keep some extra information
 |   in the file (the number of coefficients, number of vectors, and the <min,max> values per channel).
 |
 | NOTE:   Use ONLY these routines for accessing the stored vectors!
 |
 +===========================================================================================================================*/

typedef struct {
  float  min;
  float  max;
}                 MIN_MAX;

#define VEC_MAGIC ( ('V'<<24) | ('E'<<16) | ('C'<<8) | 'S' )   /*  = 0x56454353 = VECS */

/*=============================================================================================================================
 | write_scaled_vectors:  Writes vectors of vectors of floats to a file, the coefficients are stored "compacted".
 |
 | PARAMETERS:
 |   fp        = binary file pointer.  (Assumed to be already open.)
 |   wherefrom = pointer to beginning (float) location to start writing floats from
 |   coeffN    = number of coefficients per vector
 |   vectorN   = number of vectors to write
 |
 | RETURNS:
 |   Number of coefficients successfully written
 |
 | HISTORY:
 |  24 Nov 93  sloboda   Created.
 |  26.Nov 93  sloboda   added a "magic number" to the record which is written
 +===========================================================================================================================*/
int write_scaled_vectors(FILE *fp, float *wherefrom, int coeffN, int vectorN)
{
    MIN_MAX *channelA;                  /* malloc'ed array for the min/max values for each dimension */
    float   *deltaA;                    /* malloc'ed array for the interval widths for each dimension */
    float *tmpA;      /* temporary array to hold all data for conversion */
    float *tmpP;            /* temp pointer to step through temp array */
    float  *fromP = wherefrom;          /* temp pointer to step through source array */
    int     number_written;    /* number read from fread */
    int     coeffX, vectorX;            /* counters */
    int     total_coeffN = coeffN*vectorN;

  /* If bad input, don't bother */
  if (total_coeffN <= 0) return(0);

  /* Get a temp array to keep track of the min/max values in each channel */
  channelA = (MIN_MAX *)malloc(sizeof(MIN_MAX)*coeffN);
  deltaA   = (float *)  malloc(sizeof(float)*coeffN);

  /* Get a temp array to re-scale the floats */
  tmpA = (float *)malloc(sizeof(float)*total_coeffN);

  /* complain if it didn't work */
  if ((tmpA == NULL) || (channelA == NULL) || (deltaA == NULL))
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in write_scaled_vectors\n");
    return(0);
  }

  /* initialize the min/max values for each */
  for(coeffX=0; coeffX<coeffN; coeffX++)
    { channelA[coeffX].min =  1.0e30;
      channelA[coeffX].max = -1.0e30;
    }

  /* find out about the min/max values for each dimension */
  for(vectorX=0; vectorX<vectorN; vectorX++)
     for(coeffX=0; coeffX<coeffN; coeffX++, fromP++)
       { if (*fromP < channelA[coeffX].min) channelA[coeffX].min = *fromP;
         if (*fromP > channelA[coeffX].max) channelA[coeffX].max = *fromP;
       }

  /* initialize the delta array for each dimension */
  for(coeffX=0; coeffX<coeffN; coeffX++)
    {
      deltaA[coeffX] = channelA[coeffX].max - channelA[coeffX].min;
      if (deltaA[coeffX] == 0.0) deltaA[coeffX]=1.0;             /* just in case all coefficients are equal: */
    }                                                            /* then deltaA is never used anyway (but divided through!) */ 

  /* rescale the coefficients */
  fromP = wherefrom;
  tmpP = tmpA;
  for(vectorX=0; vectorX<vectorN; vectorX++)
     for(coeffX=0; coeffX<coeffN; coeffX++, fromP++, tmpP++)
        *tmpP = (*fromP - channelA[coeffX].min) / deltaA[coeffX];


  /* write the header information to the file */
  write_int (fp, VEC_MAGIC);
  write_int (fp, coeffN);
  write_int (fp, vectorN);
  write_floats (fp, (float*)channelA, 2*coeffN);

  /* now write the rescaled float coefficients as floatbytes */
  number_written = write_floatbytes(fp, tmpA, total_coeffN);

  free(tmpA);
  free(deltaA);
  free(channelA);
  return(number_written);
}

/*=============================================================================================================================
 | read_scaled_vectors:  read vectors of vectors from a file,  where the coefficients are stored "compacted", returns floats.
 |
 | PARAMETERS:
 |   fp         = binary file pointer.  (Assumed to be already open.)
 |   whereto    = pointer to beginning of an array of floats to store floats that will be read (will be allocated)
 |   coeffNP    = returns number of coefficients per vector
 |   vectorNP   = returns number of vectors read
 |
 | RETURNS:
 |   Number of coefficients successfully read, or -1 if the magic number isn't found
 |
 | NOTE:   the user has to free() the returned float array if necessary
 |
 | HISTORY:
 |  24 Nov 93  sloboda   Created.
 |  26.Nov 93  sloboda   added a "magic number" to the record which is written, and
 |                       now a malloc'ed array is returned
 |  27 Nov 93  kemp      changed
 |   3 Dec 93  sloboda   if the magic number isn't found, -1 is returned
 +===========================================================================================================================*/
int read_scaled_vectors( FILE *fp, float **whereto, int *coeffNP, int *vectorNP)
{
  int vectorN = read_scaled_vectors_range( fp, whereto,  coeffNP, vectorNP, 0, -1 );

  if (vectorN > 0) return (vectorN * *coeffNP);    /* because return value here is number of vectors */
  else return vectorN;
}

/*=============================================================================================================================
 | read_scaled_vectors_range:  read vectors of vectors (only from..to) from a file,
 |                             where the coefficients are stored "compacted", returns floats.
 |
 | PARAMETERS:
 |   fp         = binary file pointer.  (Assumed to be already open.)
 |   whereto    = pointer to beginning of an array of floats to store floats that will be read (will be allocated)
 |   coeffNP    = returns number of coefficients per vector
 |   vectorNP   = returns TOTAL number of vectors in file
 |   from       = first 'frame' to read from
 |   to         = last 'frame' to read to
 |
 |
 | RETURNS:
 |   Number of FRAMES!! successfully read, or -1 if the magic number isn't found, -2 if (from..to) are out of range
 |
 | NOTE:   the user has to free() the returned float array if necessary
 |
 | HISTORY:
 |  24 Nov 93  sloboda   Created.
 |  26.Nov 93  sloboda   added a "magic number" to the record which is written, and
 |                       now a malloc'ed array is returned
 |  27 Nov 93  kemp      changed
 |   3 Dec 93  sloboda   if the magic number isn't found, -1 is returned
 +===========================================================================================================================*/
int read_scaled_vectors_range( FILE *fp, float **whereto, int *coeffNP, int *vectorNP, int from, int to)
{
  MIN_MAX *channelA;          /* malloc'ed array for the min/max values for each dimension */
  float   *deltaA;            /* malloc'ed array for the interval widths for each dimension */
  float  *tmpA;          /* array which is returnded */
  float  *tmpP;          /* temp pointer to step through temp array */
  int     number_read;        /* number read from fread */
  int     coeffX, vectorX;
  int     coeffN, vectorN;
  int     vectorR;            /* how many vectors (or frames) to read */
  int     total_coeffN;
  int     magic;

  magic = read_int( fp );
  if (magic != VEC_MAGIC)
    return(-1);

  coeffN  = read_int( fp );
  vectorN = read_int( fp );

  /* set range */
  if (to < 0 || to >= vectorN) to = vectorN - 1;
  if (from < 0)  from = 0;
  if (from > to) {
    *coeffNP  = coeffN;
    *vectorNP = vectorN;
    return -2;
  }
  vectorR      = to - from + 1;
  total_coeffN = coeffN * vectorR;

  /* Get a temp array to keep track of the min/max values in each channel */
  channelA = (MIN_MAX *)malloc(sizeof(MIN_MAX)*coeffN);
  deltaA   = (float *)  malloc(sizeof(float)*coeffN);

  /* Get a temp array to store the intermediate results */
  tmpA = (float *)malloc(sizeof(float)*total_coeffN);
  if ((tmpA == NULL) || (channelA == NULL) || (deltaA == NULL))
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't malloc more space in read_scaled_vectors\n");
    return(0);
  }

  /* read in the min/max array */
  number_read = read_floats(fp, (float*)channelA, 2*coeffN);
  if (number_read != 2*coeffN)
  {
    fprintf(stderr, "\n\n >>> ERROR: Couldn't read the <min,max> array in read_scaled_vectors\n");
    return(0);
  }

  /* initialize the delta array for each dimension */
  for(coeffX=0; coeffX<coeffN; coeffX++)
     deltaA[coeffX] = channelA[coeffX].max - channelA[coeffX].min;

  /* read the floatbytes into the temp array */
  fseek(fp, (long)(from * coeffN * sizeof(UBYTE)), SEEK_CUR);
  number_read = read_floatbytes(fp, tmpA, total_coeffN);

  if (number_read != total_coeffN)
  {
    fprintf(stderr, "\n\n >>> ERROR: unexpected file size - coeffs read: %d, coeffs expected: %d\n",number_read,total_coeffN);
    return(0);
  }

  /* re-scale the coefficients */
  tmpP = tmpA;
  for(vectorX=0; vectorX<vectorR; vectorX++)
     for(coeffX=0; coeffX<coeffN; coeffX++, tmpP++)
        *tmpP = (deltaA[coeffX] * *tmpP) + channelA[coeffX].min;

  free(deltaA);
  free(channelA);

  *whereto  = tmpA;              /* return the pointer to the malloc'ed array */
  *coeffNP  = coeffN;
  *vectorNP = vectorN;
  return(vectorR);
}

/*=============================================================================================================================
 | set_machine:  Allows one to set the machine type manually.  Note: if called with 0, no change occurs, then useful
 |    to query current value of machine
 |
 | PARAMETERS:
 |   new_machine:  Integer specifying which machine type to set it to (see mach_ind_io.h)
 |
 | RETURNS:
 |  The previous value of the machine type
 |   
 | HISTORY:
 |   14.Jun 91  arthurem   created
 +===========================================================================================================================*/
int set_machine(int new_machine)
{
  int old_machine = machine;
  
  if (new_machine != 0) 
    machine = new_machine;
  
  return(old_machine);
}

/*=============================================================================================================================
 | check_byte_swap:  Does a little magic (hope it works!) to decide if adc data needs to be byte-swapped
 |        NOTE: this only works if your input adc data should only vary slightly between values, such as
 |              in adc files!
 |
 | PARAMETERS:
 |       buf:  Pointer to start of array of shorts
 |    bufN:  Number of shorts in buffer
 |
 | RETURNS:
 |    1 if bytes should be swapped, 0 if not
 |   
 | HISTORY:
 |   17.Jun 91  arthurem   created
 +===========================================================================================================================*/
int check_byte_swap( short *buf, int bufN )
{
    int  sum_dif_norm = 0;
    int  sum_dif_swap = 0;
    int  norm, swap;
    int  last_norm, last_swap;
    int  i;

  assert(bufN >= 139);

  last_norm = buf[128];
  last_swap = SWAP_SHORT(buf[128]);
/*  printf("    norm = %6d (%8x)\t    swap = %6d (%8x)\n", last_norm,last_norm,last_swap, last_swap); */

  for (i = 129; i < 139; i++) 
  {
    norm = buf[i];
    swap = SWAP_SHORT(norm);
    sum_dif_norm += abs(norm - last_norm);
    sum_dif_swap += abs(swap - last_swap);
/*    printf("\tdif_norm = %6d (%8x)\tdif_swap = %6d (%8x)\n", abs(norm - last_norm),abs(norm - last_norm), abs(swap - last_swap),abs(swap - last_swap)); */
/*    printf("    norm = %6d (%8x)\t    swap = %6d (%8x)\n", norm,norm,swap,swap); */
    last_norm = norm;
    last_swap = swap;
  }
  
/*  printf("sum_dif_norm = %6d\tsum_dif_swap = %6d\n", sum_dif_norm, sum_dif_swap); */
  return(sum_dif_swap < sum_dif_norm);
}

/*=============================================================================================================================
 | buf_byte_swap:  Swaps bytes of each short in an array of shorts
 |
 | PARAMETERS:
 |       buf:  Pointer to start of array of shorts
 |    bufN:  Number of shorts in buffer
 |
 | HISTORY:
 |   17.Jun 91  arthurem created
 +===========================================================================================================================*/
void buf_byte_swap( short *buf, int bufN )
{
  short *ptr;    /* used to step through array */
  int i;
  
  ptr = buf;
  
  for (i = 0; i<bufN; i++) 
  {
    *ptr = SWAP_SHORT(*ptr);
    ptr++;
  }
}

/*###########################################################################################################################*/


