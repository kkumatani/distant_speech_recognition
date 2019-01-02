/*
File Name: PC_lattice.c
Last Modification Date:	3/9/94	10:15:31
Current Version: PC_lattice.c	1.3
File Creation Date: Fri Aug 27 19:37:04 1993
Author: Ramesh Gopinath  <ramesh@dsp.rice.edu>

Copyright: All software, documentation, and related files in this distribution
           are Copyright (c) 1993  Rice University

Permission is granted for use and non-profit distribution providing that this
notice be clearly maintained. The right to distribute any portion for profit
or as part of any commercial product is specifically reserved for the author.

Change History:

*/


/* This is a library of programs for power complementary lattice manipulations:
There are three functions viz., pclat, dpclat and ddpclat */
#include <math.h>
#include "pc_lattice.h"
/*
################################################################################
Pclat - h0, h1 : Outputs of lattice (upper and lower channels)
           k : Denormalized parameters that describe the lattice
	   k_ord : Order of k (i.e., length(k)-1)
#################################################################################
*/

void Pclat(double* h0, double* h1, const double* k, int k_ord)
{
  int i,j,stride;
  double tmp,gamma, *pr0, *pr1;

  *h0 = *k;
  *(h1+k_ord) = 1;
  gamma = 1 + *k * *k;

  for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
       i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {

    *pr1 = *h0;
    *pr0 = *++k * *h0;
    
    for (j=1; j<i; j++) {
      tmp = *k * *++pr0 + *++pr1;
      *pr1 = *pr0 - *k * *pr1;
      *pr0 = tmp;
    }
    
    *++pr0 = *++pr1;
    *pr1 = -*k * *pr1;
    gamma *= (1+*k * *k);
  }

  gamma = 1.0/sqrt(gamma);
  for (i=0, pr0=h0, pr1=h1; i<= k_ord; i++) {
    *pr0++ = *pr0 * gamma;
    *pr1++ = *pr1 * gamma;
  }
}

/*
################################################################################
Dpclat - h0, h1 : Derivatives of outputs of lattice (upper and lower channels)
           k : Denormalized parameters that describe the lattice
	   k_ord : Order of k (i.e., length(k)-1)
	   iii : Index of parameter with respect to which the derivatives are taken
#################################################################################
*/
void Dpclat(double* h0, double* h1, const double* k, int k_ord, int iii)
{
  int i, j, stride;
  double tmp, gamma, *pr0, *pr1;

  if (iii == 0) {
    *h0 = 1;
    *(h1+k_ord) = -*k;
    tmp = (1+ *k * *k);
    gamma = tmp*tmp*tmp;

    for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
	 i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {
      
      *pr1 = *h0;    
      *pr0 = *++k * *h0;
      
      for (j=1; j<i; j++) {
	tmp = *k * *++pr0 + *++pr1;
	*pr1 = *pr0 - *k * *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = *++pr1;
      *pr1 = -*k * *pr1;
      gamma *= (1+*k * *k);
    }
  }
  else {

  *h0 = *k;
  *(h1+k_ord) = 1;
  gamma = 1 + *k * *k;

  for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
       i<iii;i++,pr0=h0,pr1=h1 + --stride) {

    *pr1 = *h0;    
    *pr0 = *++k * *h0;
    
    for (j=1; j<i; j++) {
      tmp = *k * *++pr0 + *++pr1;
      *pr1 = *pr0 - *k * *pr1;
      *pr0 = tmp;
    }
    
    *++pr0 = *++pr1;
    *pr1 = -*k * *pr1;
    gamma *= (1+*k * *k);
    }

    /* case i = iii */
    *pr1 = -*++k * *pr0;

      for (j=1; j<i; j++) {
	tmp = *++pr0 - *k * *++pr1;
	*pr1 = -*k * *pr0 - *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = -*k * *++pr1;
      *pr1 = - *pr1;
    tmp = (1+ *k * *k);
    gamma *= tmp*tmp*tmp;

    /* case i > iii */
  for (i=iii+1,pr0=h0,pr1=h1 + --stride;
       i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {

    *pr1 = *h0;    
    *pr0 = *++k * *h0;
    
    for (j=1; j<i; j++) {
      tmp = *k * *++pr0 + *++pr1;
      *pr1 = *pr0 - *k * *pr1;
      *pr0 = tmp;
    }
    
    *++pr0 = *++pr1;
    *pr1 = -*k * *pr1;
    gamma *= (1+*k * *k);
    }
  }
  gamma = 1.0/sqrt(gamma);
  for (i=0, pr0=h0, pr1=h1; i<= k_ord; i++) {
    *pr0++ = *pr0 * gamma;
    *pr1++ = *pr1 * gamma;
  }
}

/*
################################################################################
Dpclat - h0, h1 : Second derivatives of outputs of lattice (upper and lower channels)
           k : Denormalized parameters that describe the lattice
	   k_ord : Order of k (i.e., length(k)-1)
	   iii,jjj : Indices of parameters with respect to which two derivatives are taken
#################################################################################
*/
void Ddpclat(double* h0, double* h1, const double* k, int k_ord,
	     int iii, int jjj)
{
  int i,j, stride;
  double tmp, tmp1, tmp2, gamma, *pr0, *pr1;
  
  if (iii != jjj) {
    if (iii > jjj) { /* Swap iii and jjj */
      i = iii;
      iii = jjj;
      jjj = i;
    }
    
    /* Case of first Rotate iii */
    if (iii == 0) {
      *h0 = 1;
      *(h1+k_ord) = -*k;
      tmp = (1 + *k * *k);
      gamma = tmp*tmp*tmp;
      
      /* Check for jjj */
      for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
	   i<jjj ;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 + *k * *k);
      }
      
      /* case i=jjj */
      *pr1 = -*++k * *pr0;
      
      for (j=1; j<i; j++) {
	tmp = *++pr0 - *k * *++pr1;
	*pr1 = -*k * *pr0 - *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = -*k * *++pr1;
      *pr1 = - *pr1;
      tmp = (1 + *k * *k);
      gamma *= tmp*tmp*tmp;
      
      /* case i > jjj */
      for (i=jjj+1,pr0=h0,pr1=h1 + --stride;
	   i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 +*k * *k);
      }
    }
    else { /* iii not equal to 0 */
      *h0 = *k;
      *(h1+k_ord) = 1;
      gamma = (1 + *k * *k);
      
      for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
	   i<iii;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 + *k * *k);
      }
      
      /* case i = iii */
      *pr1 = -*++k * *pr0;
      
      for (j=1; j<i; j++) {
	tmp = *++pr0 - *k * *++pr1;
	*pr1 = -*k * *pr0 - *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = -*k * *++pr1;
      *pr1 = - *pr1;
      tmp = (1 + *k * *k);
      gamma *= tmp*tmp*tmp;
      
      for (i=iii+1,pr0=h0,pr1=h1+ --stride;
	   i<jjj;i++,pr0=h0,pr1=h1 + --stride) { 
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 +*k * *k);
      }
      
      /* case i = jjj */
      *pr1 = -*++k * *pr0;
      
      for (j=1; j<i; j++) {
	tmp = *++pr0 - *k * *++pr1;
	*pr1 = -*k * *pr0 - *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = -*k * *++pr1;
      *pr1 = - *pr1;
      tmp = (1 + *k * *k);
      gamma *= tmp*tmp*tmp;
      
      /* case i > jjj */
      for (i=jjj+1,pr0=h0,pr1=h1 + --stride;
	   i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 + *k * *k);
      }
    }
    gamma = 1.0/sqrt(gamma);
    for (i=0, pr0=h0, pr1=h1; i<= k_ord; i++) {
      *pr0++ = *pr0 * gamma;
      *pr1++ = *pr1 * gamma;
    }
  }
  else  {
    if (iii == 0) {
      tmp = *k * *k;
      tmp1 = (2*tmp-1);
      tmp2 = (1+tmp);
      tmp2 = tmp2*tmp2*sqrt(tmp2);
      *h0 = -3.0*(*k)/tmp2;
      *(h1+k_ord) = tmp1/tmp2;;
      gamma = 1;
      
      for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
	   i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 +*k * *k);
      }   
    }
    else {
      *h0 = *k;
      *(h1+k_ord) = 1;
      gamma = 1 + (*k * *k);
      
      for (i=1,pr0=h0,stride=k_ord-1,pr1=h1+stride;
	   i<iii;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 + *k * *k);
      }
      
      /* case i = iii */
      tmp = *++k;
      tmp = tmp*tmp;
      tmp2 = (2*tmp-1);
      tmp1 = (1+tmp);
      tmp1 = tmp1*tmp1*sqrt(tmp1);
      tmp2 = tmp2/tmp1;
      tmp1 = -3*(*k)/tmp1;
      
      *pr1 = tmp2 * *pr0;
      *pr0 = tmp1 * *pr0;
      
      for (j=1; j<iii; j++) {
	tmp = tmp1 * *++pr0 + tmp2**++pr1;
	*pr1 = tmp2 * *pr0 - tmp1* *pr1;
	*pr0 = tmp;
      }
      
      *++pr0 = tmp2 * *++pr1;
      *pr1 =  -tmp1 * *pr1;
      
      /* case i > iii */
      for (i=iii+1,pr0=h0,pr1=h1 + --stride;
	   i<=k_ord;i++,pr0=h0,pr1=h1 + --stride) {
	
	*pr1 = *h0;    
	*pr0 = *++k * *h0;
	
	for (j=1; j<i; j++) {
	  tmp = *k * *++pr0 + *++pr1;
	  *pr1 = *pr0 - *k * *pr1;
	  *pr0 = tmp;
	}
	
	*++pr0 = *++pr1;
	*pr1 = -*k * *pr1;
	gamma *= (1 + *k * *k);
      }      
    }
    gamma = 1.0/sqrt(gamma);
    for (i=0, pr0=h0, pr1=h1; i<= k_ord; i++) {
      *pr0++ = *pr0 * gamma;
      *pr1++ = *pr1 * gamma;
    }
  }
}
