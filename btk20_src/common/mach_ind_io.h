/**
 * @file mach_ind_io.h
 * @brief Machine-dependent operation for reading/writing files
 * @author Fabian Jakobs
 * @note have to obsolete this
 */

#ifndef MACH_IND_IO_H
#define MACH_IND_IO_H

#include <stdio.h>

typedef  unsigned char   UBYTE;         /* type for reading/writing floatbytes */


/*
 *   FUNCTION DECLARATIONS
 */

void  init_mach_ind_io( void );

UBYTE float_to_ubyte( float f );
float ubyte_to_float( UBYTE u );

float read_float( FILE *fp );
int   read_floats(FILE *fp, float *whereto, int count );

float read_floatbyte( FILE *fp );
int   read_floatbytes(FILE *fp, float *whereto, int count );

int   read_int( FILE *fp );
int   read_ints(FILE *fp, int *whereto, int count );

short read_short( FILE *fp );
int   read_shorts(FILE *fp, short *whereto, int count);

int   read_string( FILE *f, char *str);

int   read_scaled_vectors( FILE *fp, float **whereto,  int* coeffNP, int *vectorNP );
int   read_scaled_vectors_range( FILE *fp, float **whereto, int *coeffNP, int *vectorNP, int from, int to);
int   write_scaled_vectors(FILE *fp, float *wherefrom, int  coeffN,  int  vectorN );

void  write_float( FILE *fp, float f);
int   write_floats(FILE *fp, float *wherefrom, int count);

void  write_floatbyte( FILE *fp, float f);
int   write_floatbytes(FILE *fp, float *wherefrom, int count);

void  write_int( FILE *fp, int i);
int   write_ints(FILE *fp, int *wherefrom, int count);

void  write_short( FILE *fp, short s);
int   write_shorts(FILE *fp, short *wherefrom, int count);

int   write_string(FILE *f, const char* str);

int   set_machine( int new_machine );

int   check_byte_swap( short *buf, int bufN );       /* See if adc data should be swapped */

void  buf_byte_swap( short *buf, int bufN );         /* Swap bytes for each short in an adc data buffer */

int short_memorychange(short *buf, int bufN);
int float_memorychange(float *buf, int bufN);
int int_memorychange(int *buf, int bufN);
int change_short(short *x);
int change_int(int *x);
int change_float(float *x);


/*************************************************************************************************************************/

#endif   /* MACH_IND_IO_H */
