/**
 * @file common.i
 * @brief Common operation
 * @author Fabian Jakobs
 */

%module(package="btk20") common

%{
#include "mach_ind_io.h"
#include "jexception.h"
#include "common.h"
%}

%init %{
  init_mach_ind_io( );
%}

#ifdef AUTODOC
%section "Common"
#endif

%rename(__str__) *::puts();

%include typedefs.i
%include jexception.i
%include "mach_ind_io.h"

FILE* btk_fopen(const char* filename, const char* mode);
void  btk_fclose(const char* filename, FILE* fp);
