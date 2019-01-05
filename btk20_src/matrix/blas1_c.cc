# include "blas1_c.h"

//****************************************************************************80

float cabs1 ( complex <float> z )

//****************************************************************************80
//
//  Purpose:
//
//    CABS1 returns the L1 norm of a number.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    The L1 norm of a complex number is the sum of the absolute values
//    of the real and imaginary components.
//
//    CABS1 ( Z ) = abs ( real ( Z ) ) + abs ( imaginary ( Z ) )
//
//  Modified:
//
//    31 March 2007
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, complex <float> Z, the number whose norm is desired.
//
//    Output, float CABS1, the L1 norm of Z.
//
{
  float value;

  value = r4_abs ( real ( z ) ) + r4_abs ( imag ( z ) );

  return value;
}
//****************************************************************************80

float cabs2 ( complex <float> z )

//****************************************************************************80
//
//  Purpose:
//
//    CABS2 returns the L2 norm of a number.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    The L2 norm of a complex number is the square root of the sum 
//    of the squares of the real and imaginary components.
//
//    CABS2 ( Z ) = sqrt ( ( real ( Z ) )**2 + ( imaginary ( Z ) )**2 )
//
//  Modified:
//
//    10 April 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, complex <float> Z, the number whose norm is desired.
//
//    Output, float CABS2, the L2 norm of Z.
//
{
  float value;

  value = sqrt ( pow ( real ( z ), 2 ) 
               + pow ( imag ( z ), 2 ) );

  return value;
}
//****************************************************************************80

void caxpy ( int n, complex <float> ca, complex <float> cx[], 
  int incx, complex <float> cy[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    CAXPY computes a constant times a vector plus a vector.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    10 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of elements in CX and CY.
//
//    Input, complex <float> CA, the multiplier of CX.
//
//    Input, complex <float> CX[], the first vector.
//
//    Input, int INCX, the increment between successive entries of CX.
//
//    Input/output, complex <float> CY[], the second vector.
//    On output, CY(*) has been replaced by CY(*) + CA * CX(*).
//
//    Input, int INCY, the increment between successive entries of CY.
//
{
  int i;
  int ix;
  int iy;

  if ( n <= 0 )
  {
    return;
  }

  if ( cabs1 ( ca ) == 0.0 ) 
  {
    return;
  }

  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      cy[iy] = cy[iy] + ca * cx[ix];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      cy[i] = cy[i] + ca * cx[i];
    }

  }

  return;
}
//****************************************************************************80

void ccopy ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy )

//****************************************************************************80
//
//  Purpose:
//
//    CCOPY copies a vector X to a vector Y.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    10 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of elements in CX and CY.
//
//    Input, complex <float> CX[], the first vector.
//
//    Input, int INCX, the increment between successive entries of CX.
//
//    Output, complex <float> CY[], the second vector.
//
//    Input, int INCY, the increment between successive entries of CY.
//
{
  int i;
  int ix;
  int iy;

  if ( n <= 0 )
  {
    return;
  }

  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      cy[iy] = cx[ix];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      cy[i] = cx[i];
    }
  }
  return;
}
//****************************************************************************80

complex <float> cdotc ( int n, complex <float> cx[], int incx, 
  complex <float> cy[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    CDOTC forms the conjugated dot product of two vectors.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    10 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input, complex <float> CX[], the first vector.
//
//    Input, int INCX, the increment between successive entries in CX.
//
//    Input, complex <float> CY[], the second vector.
//
//    Input, int INCY, the increment between successive entries in CY.
//
//    Output, complex <float> CDOTC, the conjugated dot product of
//    the corresponding entries of CX and CY.
//
{
  int i;
  int ix;
  int iy;
  complex <float> value( 0.0, 0.0 );

  if ( n <= 0 )
  {
    return value;
  }

  if ( incx == 1 && incy == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      value = value + conj ( cx[i] ) * cy[i];
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      value = value + conj ( cx[ix] ) * cy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  return value;
}
//****************************************************************************80

complex <float> cdotu ( int n, complex <float> cx[], int incx, 
  complex <float> cy[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    CDOTU forms the unconjugated dot product of two vectors.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    10 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input, complex <float> CX[], the first vector.
//
//    Input, int INCX, the increment between successive entries in CX.
//
//    Input, complex <float> CY[], the second vector.
//
//    Input, int INCY, the increment between successive entries in CY.
//
//    Output, complex <float> CDOTU, the unconjugated dot product of
//    the corresponding entries of CX and CY.
//
{
  int i;
  int ix;
  int iy;
  complex <float> value( 0.0, 0.0 );

  if ( n <= 0 )
  {
    return value;
  }

  if ( incx == 1 && incy == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      value = value + cx[i] * cy[i];
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      value = value + cx[ix] * cy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  return value;
}
//****************************************************************************80

float cmach ( int job )

//****************************************************************************80
//
//  Purpose:
//
//    CMACH computes machine parameters for complex arithmetic.
//
//  Discussion:
//
//    Assume the computer has
//
//      B = base of arithmetic;
//      T = number of base B digits;
//      L = smallest possible exponent;
//      U = largest possible exponent;
//
//    then
//
//      EPS = B**(1-T)
//      TINY = 100.0 * B**(-L+T)
//      HUGE = 0.01 * B**(U-T)
//
//    If complex division is done by
//
//      1 / (X+i*Y) = (X-i*Y) / (X**2+Y**2)
//
//    then
//
//      TINY = sqrt ( TINY )
//      HUGE = sqrt ( HUGE )
//
//  Modified:
//
//    12 April 2006
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, int JOB:
//    1, EPS is desired;
//    2, TINY is desired;
//    3, HUGE is desired.
//
//    Output, float CMACH, the requested value.
//
{
  float eps;
  float huge;
  float s;
  complex <float> temp1;
  complex <float> temp2;
  complex <float> temp3;
  float tiny;
  float value;

  eps = 1.0;

  for ( ; ; )
  {
    eps = eps / 2.0;
    s = 1.0 + eps;
    if ( s <= 1.0 )
    {
      break;
    }
  }

  eps = 2.0 * eps;

  s = 1.0;

  for ( ; ; )
  {
    tiny = s;
    s = s / 16.0;

    if ( s * 1.0 == 0.0 )
    {
      break;
    }
  }

  tiny = ( tiny / eps ) * 100.0;
//
//  Had to insert this manually!
//
  tiny = sqrt ( tiny );

  if ( false )
  {
    temp1 = complex <float> ( 1.0,  0.0 ); 
    temp2 = complex <float> ( tiny, 0.0 );
    temp3 = temp1 / temp2;

    s = real ( temp3 );

    if ( s != 1.0 / tiny )
    {
      tiny = sqrt ( tiny );
    }
  }

  huge = 1.0 / tiny;

  if ( job == 1 )
  {
    value = eps;
  }
  else if ( job == 2 )
  {
    value = tiny;
  }
  else if ( job == 3 )
  {
    value = huge;
  }
  else
  {
    value = 0.0;
  }

  return value;
}
//****************************************************************************80

void crotg ( complex <float> *ca, complex <float> cb, float *c, 
  complex <float> *s )

//****************************************************************************80
//
//  Purpose:
//
//    CROTG determines a Givens rotation.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    Given values A and B, this routine computes:
//
//    If A = 0:
//
//      R = B
//      C = 0
//      S = (1,0).
//
//    If A /= 0:
//
//      ALPHA = A / abs ( A )
//      NORM  = sqrt ( ( abs ( A ) )**2 + ( abs ( B ) )**2 )
//      R     = ALPHA * NORM
//      C     = abs ( A ) / NORM
//      S     = ALPHA * conj ( B ) / NORM
//
//    In either case, the computed numbers satisfy the equation:
//
//    (         C    S ) * ( A ) = ( R )
//    ( -conj ( S )  C )   ( B ) = ( 0 )
//
//  Modified:
//
//    31 March 2007
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input/output, complex <float> *CA, on input, the value A.  On output,
//    the value R.
//
//    Input, complex <float> CB, the value B.
//
//    Output, float *C, the cosine of the Given rotation.
//
//    Output, complex <float> *S, the sine of the Givens rotation.
//
{
  complex <float> alpha;
  float norm;
  float scale;

  if ( cabs2 ( *ca ) == 0.0 )
  {
    *c = 0.0;
    *s = complex <float> ( 1.0, 0.0 );
    *ca = cb;
  }
  else
  {
    scale = cabs2 ( *ca ) + cabs2 ( cb );
    norm = scale * sqrt ( pow ( cabs2 ( *ca / scale ), 2 )
                        + pow ( cabs2 (  cb / scale ), 2 ) );
    alpha = *ca / cabs2 ( *ca );
    *c = cabs2 ( *ca ) / norm;
    *s = alpha * conj ( cb ) / norm;
    *ca = alpha * norm;
  }

  return;
}
//****************************************************************************80

void cscal ( int n, complex <float> ca, complex <float> cx[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    CSCAL scales a vector by a constant.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, complex <float> CA, the multiplier.
//
//    Input/output, complex <float> CX[], the vector to be scaled.
//
//    Input, int INCX, the increment between successive entries of CX.
//
{
  int i;

  if ( n <= 0 || incx <= 0 )
  {
    return;
  }

  if ( incx == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      cx[i] = ca * cx[i];
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      cx[i*incx] = ca * cx[i*incx];
    }
  }
  return;
}
//****************************************************************************80

complex <float> csign1 ( complex <float> z1, complex <float> z2 )

//****************************************************************************80
//
//  Purpose:
//
//    CSIGN1 is a transfer-of-sign function.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    The L1 norm is used.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, complex <float> Z1, Z2, the arguments.
//
//    Output, complex <float> CSIGN1,  a complex value, with the magnitude of
//    Z1, and the argument of Z2.
//
{
  complex <float> value;

  if ( cabs1 ( z2 ) == 0.0 )
  {
    value = complex <float> ( 0.0, 0.0 );
  }
  else
  {
    value = cabs1 ( z1 ) * ( z2 / cabs1 ( z2 ) );
  }

  return value;
}
//****************************************************************************80

complex <float> csign2 ( complex <float> z1, complex <float> z2 )

//****************************************************************************80
//
//  Purpose:
//
//    CSIGN2 is a transfer-of-sign function.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    The L2 norm is used.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, complex <float> Z1, Z2, the arguments.
//
//    Output, complex <float> CSIGN2,  a complex value, with the magnitude of
//    Z1, and the argument of Z2.
//
{
  complex <float> value;

  if ( cabs2 ( z2 ) == 0.0 )
  {
    value = complex <float> ( 0.0, 0.0 );
  }
  else
  {
    value = cabs2 ( z1 ) * ( z2 / cabs2 ( z2 ) );
  }

  return value;
}
//****************************************************************************80

void csrot ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy, float c, float s )

//****************************************************************************80
//
//  Purpose:
//
//    CSROT applies a plane rotation.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    The cosine C and sine S are real and the vectors CX and CY are complex.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input/output, complex <float> CX[], one of the vectors to be rotated.
//
//    Input, int INCX, the increment between successive entries of CX.
//
//    Input/output, complex <float> CY[], one of the vectors to be rotated.
//
//    Input, int INCY, the increment between successive elements of CY.
//
//    Input, float C, S, parameters (presumably the cosine and sine of
//    some angle) that define a plane rotation.
//
{
  complex <float> ctemp;
  int i;
  int ix;
  int iy;

  if ( n <= 0 )
  {
    return;
  }

  if ( incx == 1 && incy == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      ctemp = c * cx[i] + s * cy[i];
      cy[i] = c * cy[i] - s * cx[i];
      cx[i] = ctemp;
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      ctemp  = c * cx[ix] + s * cy[iy];
      cy[iy] = c * cy[iy] - s * cx[ix];
      cx[ix] = ctemp;
      ix = ix + incx;
      iy = iy + incy;
    }
  }
  return;
}
//****************************************************************************80

void csscal ( int n, float sa, complex <float> cx[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    CSSCAL scales a vector by a real constant.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, float SA, the multiplier.
//
//    Input/output, complex <float> CX[], the vector to be scaled.
//
//    Input, int INCX, the increment between successive entries of
//    the vector CX.
//
{
  int i;

  if ( n <= 0 || incx <= 0 )
  {
    return;
  }

  if ( incx == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      cx[i] = sa * cx[i];
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      cx[i*incx] = sa * cx[i*incx];
    }
  }
  return;
}
//****************************************************************************80

void cswap ( int n, complex <float> cx[], int incx, complex <float> cy[], 
  int incy )

//****************************************************************************80
//
//  Purpose:
//
//    CSWAP interchanges two vectors.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    Jack Dongarra
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input/output, complex <float> CX[], one of the vectors to swap.
//
//    Input, int INCX, the increment between successive entries of CX.
//
//    Input/output, complex <float> CY[], one of the vectors to swap.
//
//    Input, int INCY, the increment between successive elements of CY.
//
{
  complex <float> ctemp;
  int i;
  int ix;
  int iy;

  if ( n <= 0 )
  {
    return;
  }

  if ( incx == 1 && incy == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      ctemp = cx[i];
      cx[i] = cy[i];
      cy[i] = ctemp;
    }
  }
  else
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( -n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( -n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      ctemp = cx[ix];
      cx[ix] = cy[iy];
      cy[iy] = ctemp;
      ix = ix + incx;
      iy = iy + incy;
    }
  }

  return;
}
//****************************************************************************80

int i4_max ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MAX returns the maximum of two I4's.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, are two integers to be compared.
//
//    Output, int I4_MAX, the larger of I1 and I2.
//
//
{
  if ( i2 < i1 )
  {
    return i1;
  }
  else
  {
    return i2;
  }

}
//****************************************************************************80

int i4_min ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MIN returns the smaller of two I4's.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, two integers to be compared.
//
//    Output, int I4_MIN, the smaller of I1 and I2.
//
//
{
  if ( i1 < i2 )
  {
    return i1;
  }
  else
  {
    return i2;
  }

}
//****************************************************************************80

int icamax ( int n, complex <float> x[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    ICAMAX indexes the vector element of maximum absolute value.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    WARNING: This index is a 1-based index, not a 0-based index!
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, complex <float> X[], the vector.
//
//    Input, int INCX, the increment between successive entries of X.
//
//    Output, int ICAMAX, the index of the element of maximum
//    absolute value.
//
{
  int i;
  int ix;
  float smax;
  int value;

  value = 0;

  if ( n < 1 || incx  <=  0 )
  {
    return value;
  }

  value = 1;

  if ( n == 1 )
  {
    return value;
  }

  if ( incx != 1 )
  {
    ix = 0;
    smax = cabs1 ( x[0] );
    ix = ix + incx;

    for ( i = 1; i < n; i++ )
    {
      if ( smax < cabs1 ( x[ix] ) )
      {
        value = i + 1;
        smax = cabs1 ( x[ix] );
      }
      ix = ix + incx;
    }
  }
  else
  {
    smax = cabs1 ( x[0] );
    for ( i = 1; i < n; i++ )
    {
      if ( smax < cabs1 ( x[i] ) )
      {
        value = i + 1;
        smax = cabs1 ( x[i] );
      }
    }
  }

  return value;
}
//****************************************************************************80

bool lsame ( char ca, char cb )

//****************************************************************************80
//
//  Purpose:
//
//    LSAME returns TRUE if CA is the same letter as CB regardless of case.
//
//  Modified:
//
//    02 May 2005
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539, 
//    ACM Transactions on Mathematical Software, 
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, char CA, CB, the characters to compare.
//
//    Output, bool LSAME, is TRUE if the characters are equal,
//    disregarding case.
//
{
  if ( ca == cb )
  {
    return true;
  }

  if ( 'A' <= ca && ca <= 'Z' )
  {
    if ( ca - 'A' == cb - 'a' )
    {
      return true;
    }    
  }
  else if ( 'a' <= ca && ca <= 'z' )
  {
    if ( ca - 'a' == cb - 'A' )
    {
      return true;
    }
  }

  return false;
}
//****************************************************************************80

float r4_abs ( float x )

//****************************************************************************80
//
//  Purpose:
//
//    R4_ABS returns the absolute value of an R4.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, float X, the quantity whose absolute value is desired.
//
//    Output, float R4_ABS, the absolute value of X.
//
{
  float value;

  if ( 0.0 <= x )
  {
    value = x;
  } 
  else
  {
    value = -x;
  }
  return value;
}
//****************************************************************************80

float r4_sign ( float x )

//****************************************************************************80
//
//  Purpose:
//
//    R4_SIGN returns the sign of an R4.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, float X, the number whose sign is desired.
//
//    Output, float R4_SIGN, the sign of X.
//
{
  float value;

  if ( x < 0.0 )
  {
    value = -1.0;
  } 
  else
  {
    value = 1.0;
  }
  return value;
}
//****************************************************************************80

float scasum ( int n, complex <float> x[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    SCASUM takes the sum of the absolute values of a vector.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, complex <float> X[], the vector.
//
//    Input, int INCX, the increment between successive entries of X.
//
//    Output, float SCASUM, the sum of the absolute values.
//
{
  int i;
  int ix;
  float value;

  value = 0.0;

  if ( n <= 0 || incx <= 0 )
  {
    return value;
  }

  if ( incx == 1 )
  {
    for ( i = 0; i < n; i++ )
    {
      value = value + r4_abs ( real ( x[i] ) ) 
                    + r4_abs ( imag ( x[i] ) );
    }
  }
  else
  {
    ix = 0;
    for ( i = 0; i < n; i++ )
    {
      value = value + r4_abs ( real ( x[ix] ) ) 
                    + r4_abs ( imag ( x[ix] ) );
      ix = ix + incx;
    }
  }
  return value;
}
//****************************************************************************80

float scnrm2 ( int n, complex <float> x[], int incx )

//****************************************************************************80
//
//  Purpose:
//
//    SCNRM2 returns the euclidean norm of a vector.
//
//  Discussion:
//
//    This routine uses single precision complex arithmetic.
//
//    SCNRM2 := sqrt ( sum ( conjg ( x(1:n) ) * x(1:n) ) )
//            = sqrt ( dot_product ( x(1:n), x(1:n) ) )
//
//  Modified:
//
//    11 April 2006
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for FORTRAN usage,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, pages 308-323, 1979.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, complex <float> X[], the vector.
//
//    Input, int INCX, the increment between successive entries of X.
//
//    Output, float SCNRM2, the norm of the vector.
//
{
  int i;
  int ix;
  float scale;
  float ssq;
  float temp;
  float value;

  if ( n < 1 || incx < 1 )
  {
    value  = 0.0;
  }
  else
  {
    scale = 0.0;
    ssq = 1.0;
    ix = 0;

    for ( i = 0; i < n; i++ )
    {
      if ( real ( x[ix] ) != 0.0 )
      {
        temp = r4_abs ( real ( x[ix] ) );
        if ( scale < temp )
        {
          ssq = 1.0 + ssq * pow ( scale / temp, 2 );
          scale = temp;
        }
        else
        {
          ssq = ssq + pow ( temp / scale, 2 );
        }
      }

      if ( imag ( x[ix] ) != 0.0 )
      {
        temp = r4_abs ( imag ( x[ix] ) );
        if ( scale < temp )
        {
          ssq = 1.0 + ssq * pow ( scale / temp, 2 );
          scale = temp;
        }
        else
        {
          ssq = ssq + pow ( temp / scale, 2 );
        }
      }
      ix = ix + incx;
    }
    value  = scale * sqrt ( ssq );
  }
  return value;
}
//****************************************************************************80

void xerbla ( char *srname, int info )

//****************************************************************************80
//
//  Purpose:
//
//    XERBLA is an error handler for the LAPACK routines.
//
//  Modified:
//
//    02 May 2005
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539, 
//    ACM Transactions on Mathematical Software, 
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, char *SRNAME, the name of the routine
//    which called XERBLA.
//
//    Input, int INFO, the position of the invalid parameter in
//    the parameter list of the calling routine.
//
{
  cout << "\n";
  cout << "XERBLA - Fatal error!\n";
  cout << "  On entry to routine " << srname << "\n";
  cout << "  input parameter number " << info << " had an illegal value.\n";
  exit ( 1 );
}
