/*
 * @file beamformer.cc
 * @brief Beamforming in the subband domain.
 * @author John McDonough and Kenichi Kumatani
 */

#include "beamformer/beamformer.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_trig.h>
#include <matrix/blas1_c.h>
#include <matrix/linpack_c.h>
#include "postfilter/postfilter.h"

//float  sspeed = 343740.0;

// ----- members for class `SnapShotArray' -----
//
SnapShotArray::SnapShotArray(unsigned fftLn, unsigned nChn)
  : fftLen_(fftLn), nChan_(nChn)
{
  samples_  = new gsl_vector_complex*[nChan_];
  for (unsigned i = 0; i < nChan_; i++)
    samples_[i] = gsl_vector_complex_alloc(fftLen_);

  snapshots_ = new gsl_vector_complex*[fftLen_];
  for (unsigned i = 0; i < fftLen_; i++)
    snapshots_[i] = gsl_vector_complex_alloc(nChan_);
}

SnapShotArray::~SnapShotArray()
{
  for (unsigned i = 0; i < nChan_; i++)
    gsl_vector_complex_free(samples_[i]);
  delete[] samples_;

  for (unsigned i = 0; i < fftLen_; i++)
    gsl_vector_complex_free(snapshots_[i]);
  delete[] snapshots_;
}

void SnapShotArray::zero()
{
  for (unsigned i = 0; i < nChan_; i++)
    gsl_vector_complex_set_zero(samples_[i]);

  for (unsigned i = 0; i < fftLen_; i++)
    gsl_vector_complex_set_zero(snapshots_[i]);
}

/**
   @brief set the snapshot at each chanell bin to the inner member
   @param const gsl_vector_complex* samp[in]
   @param unsigned chanX[in]
   @note after you set all the channel data, call update() method.
*/
void SnapShotArray::set_samples(const gsl_vector_complex* samp, unsigned chanX)
{
  assert(chanX < nChan_);
  gsl_vector_complex_memcpy(samples_[chanX], samp);
}

void SnapShotArray::update()
{
  for (unsigned ifft = 0; ifft < fftLen_; ifft++) {
    for (unsigned irow = 0; irow < nChan_; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(samples_[irow], ifft);
      gsl_vector_complex_set(snapshots_[ifft], irow, rowVal);
    }
  }
}

/**
   @brief set the snapshot at each frequnecy bin to the internal member.
   @param const gsl_vector_complex* snapshots[in]
   @param unsigned fbinX[in]
   @note if this method is used for setting all the frequency components, 
         don NOT call update() method.
*/
void SnapShotArray::set_snapshots(const gsl_vector_complex* snapshots, unsigned fbinX)
{
  unsigned fftLen2 = fftLen_/2;
  assert( fbinX <= fftLen2 );

  gsl_vector_complex_memcpy( snapshots_[fbinX], snapshots );
  if( fbinX == 0 || fbinX == fftLen2 )
    return;

  for(unsigned chanX=0;chanX<nChan_;chanX++){
    gsl_vector_complex_set( snapshots_[fftLen2-fbinX], chanX,
                            gsl_complex_conjugate( gsl_vector_complex_get( snapshots, chanX ) ) );
  }
  return;
}

// ----- members for class `SpectralMatrixArray' -----
//
SpectralMatrixArray::SpectralMatrixArray(unsigned fftLn, unsigned nChn,
					 float forgetFact)
  : SnapShotArray(fftLn, nChn),
    mu_(gsl_complex_rect(forgetFact, 0))
{
  matrices_ = new gsl_matrix_complex*[fftLen_];
  for (unsigned i = 0; i < fftLen_; i++)
    matrices_[i] = gsl_matrix_complex_alloc(nChan_, nChan_);
}

SpectralMatrixArray::~SpectralMatrixArray()
{
  for (unsigned i = 0; i < fftLen_; i++)
    gsl_matrix_complex_free(matrices_[i]);
  delete[] matrices_;
}

void SpectralMatrixArray::zero()
{
  SnapShotArray::zero();

  for (unsigned i = 0; i < fftLen_; i++)
    gsl_matrix_complex_set_zero(matrices_[i]);
}

void SpectralMatrixArray::update()
{
  SnapShotArray::update();

  for (unsigned ifft = 0; ifft < fftLen_; ifft++) {
    gsl_matrix_complex* smat = matrices_[ifft];
    gsl_matrix_complex_scale(smat, mu_);

    for (unsigned irow = 0; irow < nChan_; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(snapshots_[ifft], irow);
      for (unsigned icol = 0; icol < nChan_; icol++) {
	gsl_complex colVal = gsl_vector_complex_get(snapshots_[ifft], icol);
	gsl_complex newVal = gsl_complex_mul(rowVal, colVal);
	gsl_complex oldVal = gsl_matrix_complex_get(smat, irow, icol);
	gsl_complex alpha  = gsl_complex_sub( gsl_complex_rect( 1.0, 0 ), mu_ );
	newVal =  gsl_complex_mul( alpha, newVal );
	gsl_matrix_complex_set(smat, irow, icol,
			       gsl_complex_add(oldVal, newVal));
      }
    }
  }
}

// ----- members for class `FBSpectralMatrixArray' -----
//
FBSpectralMatrixArray::FBSpectralMatrixArray(unsigned fftLn, unsigned nChn, float forgetFact)
  : SpectralMatrixArray(fftLn, nChn, forgetFact) { }

FBSpectralMatrixArray::~FBSpectralMatrixArray()
{
}

void FBSpectralMatrixArray::update()
{
  SnapShotArray::update();

  for (unsigned ifft = 0; ifft < fftLen_; ifft++) {
    gsl_matrix_complex* smat = matrices_[ifft];
    gsl_matrix_complex_scale(smat, mu_);

    for (unsigned irow = 0; irow < nChan_; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(snapshots_[ifft], irow);
      for (unsigned icol = 0; icol < nChan_; icol++) {
	gsl_complex colVal = gsl_vector_complex_get(samples_[ifft], icol);
	gsl_complex newVal = gsl_complex_mul(rowVal, colVal);
	gsl_complex oldVal = gsl_matrix_complex_get(smat, irow, icol);
	gsl_matrix_complex_set(smat, irow, icol,
			       gsl_complex_add(oldVal, newVal));
      }
    }
  }
}

/**
   @brief calculate the inverse matrixof 2 x 2 matrix,
          inv( A + bI ) 
   @note It is fast because this function doesn't use a special mothod.
   @param mat[in/out]
 */
static void calc_inverse_22mat_( gsl_matrix_complex *mat, float beta = 0.01 )
{
  gsl_complex mat00, mat11, mat01, mat10, det, val;

  mat00 = gsl_matrix_complex_get( mat, 0, 0 );
  mat11 = gsl_matrix_complex_get( mat, 1, 1 );
  mat01 = gsl_matrix_complex_get( mat, 0, 1 );
  mat10 = gsl_matrix_complex_get( mat, 1, 0 );

#define USE_TH_INVERSION
#ifdef USE_TH_INVERSION
#define MINDET_THRESHOLD (1.0E-07)
  det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
  if ( gsl_complex_abs( det ) < MINDET_THRESHOLD ){

    fprintf(stderr,"calc_inverse_22mat_:compensate for non-invertibility\n");
    mat00 = gsl_complex_add_real( mat00, beta );
    mat11 = gsl_complex_add_real( mat11, beta );
    det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
  }
#else
  mat00 = gsl_complex_add_real( mat00, beta );
  mat11 = gsl_complex_add_real( mat11, beta );
  det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
#endif

  val = gsl_complex_div( mat11, det );
  gsl_matrix_complex_set( mat, 0, 0, val );

  val = gsl_complex_div( mat00, det );
  gsl_matrix_complex_set( mat, 1, 1, val );

  val = gsl_complex_div( mat01, det );
  val = gsl_complex_mul_real(val, -1.0 );
  gsl_matrix_complex_set( mat, 0, 1, val );

  val = gsl_complex_div( mat10, det );
  val = gsl_complex_mul_real(val, -1.0 );
  gsl_matrix_complex_set( mat, 1, 0, val );

}

/**
   @brief calculate the Moore-Penrose pseudoinverse matrix

   @param gsl_matrix_complex *A[in] an input matrix. A[M][N] where M > N
   @param gsl_matrix_complex *invA[out] an pseudoinverse matrix of A
   @param float dThreshold[in]

   @note if M>N, invA * A = E. Otherwise, A * invA = E.
 */
bool pseudoinverse( gsl_matrix_complex *A, gsl_matrix_complex *invA, float dThreshold )
{
   size_t M = A->size1;
   size_t N = A->size2;
   int info;
   complex <float> *a = new complex <float>[M*N];
   complex <float> *s = new complex <float>[M+N];
   complex <float> *e = new complex <float>[M+N];
   complex <float> *u = new complex <float>[M*M];
   complex <float> *v = new complex <float>[N*N];
   size_t lda = M;
   size_t ldu = M;
   size_t ldv = N;
   bool ret = true;

   for ( size_t i = 0; i < M; i++ )
     for ( size_t j = 0; j < N; j++ ){
       gsl_complex gx = gsl_matrix_complex_get(A, i, j);
       a[i+j*lda] = complex <float>(GSL_REAL(gx), GSL_IMAG(gx));
     }

   info = csvdc (a, lda, M, N, s, e, u, ldu, v, ldv, 11);
   if ( info != 0 ){
     cout << "\n";
     cout << "Warning:\n";
     cout << "  CSVDC returned nonzero INFO = " << info << "\n";
     //gsl_matrix_complex_set_identity( invA );
     ret = false;//return(true);
   }

   for (size_t k = 0; k <N; k++ ){
     if ( abs(s[k]) < dThreshold ){
       s[k] = complex<float>(0.0, 0.0);
       fprintf( stderr, "pseudoinverse: s[%lu] = 0 because of %e < %e\n", k, abs(s[k]), dThreshold);
       ret = false;
     }
     else
       s[k] = complex<float>(1.0, 0.0) / s[k];
   }

   for ( size_t i = 0; i < M; i++ ){
     for ( size_t j = 0; j < N; j++ ){
       complex <float> x( 0.0, 0.0 );
       for ( size_t k = 0; k < N/*minN*/; k++ ){
         x = x + v[j+k*ldv] * s[k] * conj ( u[i+k*ldu] );
       }
       gsl_matrix_complex_set( invA, j, i, gsl_complex_rect( x.real(), x.imag() ) );
     }
   }

   delete [] a;
   delete [] s;
   delete [] e;
   delete [] u;
   delete [] v;

   return(ret);
}

/**
   @brief calculate a quiescent weight vector wq = C * v = C * inv( C_H * C ) * g. 
   @param gsl_vector_complex* wt[in/out] array manifold for the signal of a interest. wt[chanN]
   @param gsl_vector_complex** pWj[in] array manifolds for the interferences. pWj[NC-1][chanN]
   @param int chanN[in] the number of sensors
   @param int NC[in] the number of constraints. 
   @return 
 */
static bool calc_null_beamformer_( gsl_vector_complex* wt, gsl_vector_complex** pWj, int chanN, int NC = 2 )
{
  gsl_matrix_complex* constraintMat;  // the constraint matrix
  gsl_matrix_complex* constraintMat1; // the copy of constraintMat
  gsl_vector_complex* g; // [NC] the gain vector which has constant elements
  gsl_complex alpha;
  gsl_matrix_complex* invMat;
  gsl_vector_complex* v;

  constraintMat  = gsl_matrix_complex_alloc( chanN, NC );
  constraintMat1 = gsl_matrix_complex_alloc( chanN, NC );
  invMat = gsl_matrix_complex_alloc( NC, NC );
  if( NULL == constraintMat ||  NULL == constraintMat1  || NULL == invMat ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return false;
  }

  for (int i = 0; i < chanN; i++){
    gsl_matrix_complex_set( constraintMat, i, 0, gsl_vector_complex_get( wt, i ) );
    for(int j = 1; j < NC; j++)
      gsl_matrix_complex_set( constraintMat, i, j, gsl_vector_complex_get( pWj[j-1], i ) );
  }
  gsl_matrix_complex_memcpy( constraintMat1, constraintMat );

  g = gsl_vector_complex_alloc( NC );
  v = gsl_vector_complex_alloc( NC );
  if( NULL == g || NULL == v ){
    gsl_matrix_complex_free( constraintMat );
    gsl_matrix_complex_free( constraintMat1 );
    gsl_matrix_complex_free( invMat );
    fprintf(stderr,"gsl_vector_complex_alloc failed\n");
    return false;
  }

  gsl_vector_complex_set( g, 0, gsl_complex_rect( 1.0, 0.0 ) );
  for(int j = 1; j < NC; j++)
    gsl_vector_complex_set( g, j, gsl_complex_rect( 0.0, 0.0 ) );

  GSL_SET_COMPLEX( &alpha, 1.0, 0.0 );
  // calculate C_H * C 
  gsl_matrix_complex_set_zero( invMat );
  gsl_blas_zgemm( CblasConjTrans, CblasNoTrans, alpha, constraintMat, constraintMat1, alpha, invMat );
  // calculate inv( C_H * C ) 
  if( 2!=NC ){
    // write a code which calculates a NxN inverse matrix.
    pseudoinverse( invMat, invMat );
  }
  else{
    calc_inverse_22mat_( invMat );
  }
  // calculate inv( C_H * C ) * g
  gsl_vector_complex_set_zero( v );
  gsl_blas_zgemv( CblasNoTrans, alpha, invMat, g, alpha, v );
  // calculate C * v = C * inv( C_H * C ) *g
  gsl_vector_complex_set_zero( wt );
  gsl_blas_zgemv( CblasNoTrans, alpha, constraintMat1, v, alpha, wt );

  gsl_matrix_complex_free( constraintMat );
  gsl_matrix_complex_free( constraintMat1 );
  gsl_matrix_complex_free( invMat );
  gsl_vector_complex_free( g );
  gsl_vector_complex_free( v );

  return true;
}

/**
   @brief Calculate the blocking matrix for a distortionless beamformer and return its Hermitian transpose.      
   @param  gsl_vector_complex* arrayManifold[in] the fixed weights in an upper branch
   @param  int NC[in] the number of constraints in an upper branch
   @param  gsl_matrix_complex* blockMatA[out]
   @return success -> true, error -> false
   @note This code was transported from subbandBeamforming.py. 
 */
static bool calc_blocking_matrix_( gsl_vector_complex* arrayManifold, int NC, gsl_matrix_complex* blockMat )
{
  gsl_matrix_complex* PcPerp;
  gsl_vector_complex *vec, *rvec, *conj1;
  int vsize    = arrayManifold->size;
  int bsize    = vsize - NC;

  if( bsize <= 0 ){
    fprintf(stderr,"The number of sensors %d > the number of constraints %d\n",vsize,NC);
    return false;
  }

  gsl_matrix_complex_set_zero( blockMat );

  PcPerp = gsl_matrix_complex_alloc( vsize, vsize );
  if( NULL == PcPerp ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return false;
  }

  vec = gsl_vector_complex_alloc( vsize );
  rvec = gsl_vector_complex_alloc( vsize );
  conj1 = gsl_vector_complex_alloc( vsize );
  if( NULL==vec || NULL == rvec || NULL==conj1 ){
    fprintf(stderr,"gsl_vector_complex_alloc\n");
    return false;
  }

  double norm_vs = gsl_blas_dznrm2( arrayManifold );
  norm_vs = norm_vs * norm_vs; // note ...
  gsl_complex alpha;
  GSL_SET_COMPLEX ( &alpha, -1.0/norm_vs, 0 );

  gsl_matrix_complex_set_identity( PcPerp );
  for(int i = 0; i < vsize; i++)
    gsl_vector_complex_set( conj1, i, gsl_complex_conjugate( gsl_vector_complex_get( arrayManifold, i ) ) );
  gsl_blas_zgeru( alpha, conj1, arrayManifold, PcPerp );

  for(int idim=0;idim<bsize;idim++){
    double norm_vec;

    gsl_matrix_complex_get_col( vec, PcPerp, idim );
    for(int jdim=0;jdim<idim;jdim++){
      gsl_complex ip;

      gsl_matrix_complex_get_col( rvec, blockMat, jdim );
      for (int j = 0; j < vsize; j++){
	gsl_complex out;
	out = gsl_vector_complex_get(rvec, j);
      }
      gsl_blas_zdotc( rvec, vec, &ip );
      ip = gsl_complex_mul_real( ip, -1.0 );
      gsl_blas_zaxpy( ip, rvec, vec );
    }
    // I cannot understand why the calculation of the normalization coefficient norm_vec is different from that of norm_vs.
    // But I imitated the original python code faithfully.
    norm_vec = gsl_blas_dznrm2( vec );
    gsl_blas_zdscal ( 1.0 / norm_vec, vec );
    gsl_matrix_complex_set_col(blockMat, idim, vec );
  }

  gsl_matrix_complex_free( PcPerp );
  gsl_vector_complex_free( vec );
  gsl_vector_complex_free( rvec );
  gsl_vector_complex_free( conj1 );

#ifdef _MYDEBUG_
  // conjugate the blocking matrix
  for (int i = 0; i < vsize; i++){
    for (int j = 0; j < bsize; j++){
      gsl_complex val;

      val = gsl_matrix_complex_get( blockMat, i, j );
      gsl_matrix_complex_set( blockMat, i, j,  gsl_complex_conjugate( val ) );
      //printf ("%e %e, ",  GSL_REAL(val), GSL_IMAG(val) );
    }
    //printf("\n");
  }
#endif

  return true;
}

 /**
   @brief a wrapper function for calc_blocking_matrix_( gsl_vector_complex* , int, gsl_matrix_complex* )
   @note  you have to free the returned pointer afterwards. 
   @param  gsl_vector_complex* arrayManifold[in] the fixed weights in an upper branch
   @param  int NC[in] the number of constraints in an upper branch
   @return the pointer to a blocking matrix
 */
gsl_matrix_complex* getBlockingMatrix( gsl_vector_complex* arrayManifold, int NC )
{
  int vsize    = arrayManifold->size;
  int bsize    = vsize - NC;
  gsl_matrix_complex* blockMat;

  blockMat = gsl_matrix_complex_alloc( vsize, bsize );
  if( NULL == blockMat ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return NULL;
  }

  if( false==calc_blocking_matrix_( arrayManifold, NC, blockMat ) ){
    fprintf(stderr,"getBlockingMatrix() failed\n");
    return NULL;
  }
  return blockMat;
}

// ----- members for class `BeamformerWeights' -----
//

BeamformerWeights::BeamformerWeights( unsigned fftLen, unsigned chanN, bool halfBandShift, unsigned NC )
  : fftLen_(fftLen), chanN_(chanN),  halfBandShift_(halfBandShift), NC_(NC)
{
  this->alloc_weights_();
}

BeamformerWeights::~BeamformerWeights()
{
  this->free_weights_();
}

/**
   @brief calculate array manifold vectors for the delay & sum beamformer.
   @param double samplerate[in]
   @param const gsl_vector* delays[in] a vector whose element indicates time delay. delaysT[chanN]
   @param bool isGSC[in] if it's 'true', blocking matrices will be calculated.
 */
void BeamformerWeights::calcMainlobe( float samplerate, const gsl_vector* delays, bool isGSC )
{
  if (delays->size != chanN_ )
    throw jdimension_error("Number of delays does not match number of channels (%d vs. %d).\n",
			   delays->size, chanN_ );
  if( true == isGSC && chanN_ <=1 ){
    throw jdimension_error("The number of channels must be > 1 but it is %d\n",
			   chanN_ );
  }
  if( wq_ == NULL ) this->alloc_weights_();

  unsigned fftLen2 = fftLen_ / 2;

  if ( halfBandShift_==true ) {
    float fshift = 0.5;

    for (unsigned fbinX = 0; fbinX < fftLen2; fbinX++) {
      gsl_vector_complex* vec     = wq_[fbinX];
      gsl_vector_complex* vecConj = wq_[fftLen_ - 1 - fbinX];
      for (unsigned chanX = 0; chanX < chanN_; chanX++) {
	double val = -2.0 * M_PI * (fshift+fbinX) * samplerate * gsl_vector_get(delays, chanX) / fftLen_;
	gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), chanN_ ) );
	gsl_vector_complex_set(vecConj, chanX, gsl_complex_div_real( gsl_complex_polar(1.0, -val), chanN_ ) );
      }
    }

  } else {
    gsl_vector_complex* vec;
    gsl_vector_complex* vecConj;

    // calculate weights of a direct component.
    vec     = wq_[0];
    for (unsigned chanX = 0; chanX < chanN_; chanX++) 
      gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  0.0), chanN_ ) );    
    // calculate weights from FFT bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX < fftLen2; fbinX++) {
      vec     = wq_[fbinX];
      vecConj = wq_[fftLen_ - fbinX];
      for (unsigned chanX = 0; chanX < chanN_; chanX++) {
	double val = -2.0 * M_PI * fbinX * gsl_vector_get(delays, chanX) * samplerate / fftLen_;
	gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), chanN_ ) );
	gsl_vector_complex_set(vecConj, chanX, gsl_complex_div_real( gsl_complex_polar(1.0, -val), chanN_ ) );
      }
    }
    // for exp(-j*pi)
    vec     = wq_[fftLen2];
    for (unsigned chanX = 0; chanX < chanN_; chanX++) {
      double val = -M_PI * samplerate * gsl_vector_get(delays, chanX);
      gsl_vector_complex_set(vec, chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), chanN_ ) );
    }
  }

  this->setTimeAlignment();

  // calculate a blocking matrix for each frequency bin.
  if( true == isGSC ){  
    for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++) {
      if( false==calc_blocking_matrix_(  wq_[fbinX], 1, B_[fbinX] ) ){
	throw j_error("calc_blocking_matrix_() failed\n");
      }
    }
  }

}

/**
   @brief you can constrain a beamformer to preserve a target signal and, at the same time, to suppress an interference signal.
   @param float samplerate[in]
   @param const gsl_vector*  delaysT[in] a time delay vector for a target signal. delaysT[chanN]
   @param const gsl_vector** delaysJ[in] a time delay vector for an interference. delaysI[chanN]
 */
void BeamformerWeights::calcMainlobe2( float samplerate, const gsl_vector* delaysT, const gsl_vector* delaysI, bool isGSC  )
{
  if( delaysI->size != chanN_ )
    throw jdimension_error("The number of delays for an interference signal does not match number of channels (%d vs. %d).\n",
			   delaysI->size, chanN_ );
  if( chanN_ < 2 ){
    throw jdimension_error("The number of channels must be > 2 but it is %d\n",
			   chanN_ );
  }

  gsl_matrix* delaysIs = gsl_matrix_alloc( 1 /* = 2 - 1 */, chanN_ ); // the number of interferences = 1

  if( NULL == delaysIs )
    throw jallocation_error("gsl_matrix_complex_alloc failed\n");
  gsl_matrix_set_row( delaysIs, 0, delaysI );
  this->calcMainlobeN( samplerate, delaysT, delaysIs, 2, isGSC );

  gsl_matrix_free( delaysIs );
}

/**
   @brief put multiple constraints to preserve a target signal and, at the same time, to suppress interference signals.
   @param float samplerate[in]
   @param const gsl_vector*  delaysT[in] delaysT[chanN]
   @param const gsl_vector** delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
 */
void BeamformerWeights::calcMainlobeN( float samplerate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC, bool isGSC  )
{
  if( NC < 2  || NC > chanN_ )
    throw jdimension_error("1 < the number of constraints %d <= the number of sensors %d.\n",
			   NC, chanN_ );

  if( delaysT->size != chanN_ )
    throw jdimension_error("The number of delays does not match number of channels (%d vs. %d).\n",
			   delaysT->size, chanN_ );

  unsigned fftLen2 = fftLen_ / 2;
  gsl_vector_complex** pWj      = new gsl_vector_complex*[NC-1];
  gsl_vector_complex** pWjConj  = new gsl_vector_complex*[NC-1];
  for(int n=0;n<NC-1;n++){
     pWj[n]     = gsl_vector_complex_alloc( chanN_ );
     pWjConj[n] = gsl_vector_complex_alloc( chanN_ );
  }

#ifdef _MYDEBUG_
  for (int n = 0; n < NC-1; n++){
    printf("delay %d\n",n);
    for (unsigned chanX = 0; chanX < chanN(); chanX++ ){
      float val = gsl_matrix_get(delaysJ, n, chanX);
      printf ("%e, ",  (val) );
    }
    printf("\n");
    fflush(stdout);
  }
#endif

  // set values to wq_[].
  this->calcMainlobe( samplerate, delaysT, false );

  if ( halfBandShift_==true ) {
    float fshift = 0.5;

    for (unsigned fbinX = 0; fbinX < fftLen2; fbinX++) {
      gsl_vector_complex* vec     = wq_[fbinX];
      gsl_vector_complex* vecConj = wq_[fftLen_ - 1 - fbinX];
      for (unsigned chanX = 0; chanX < chanN_; chanX++) {
	gsl_complex wq_f  = gsl_vector_complex_get( vec,     chanX );
	gsl_complex wqc_f = gsl_vector_complex_get( vecConj, chanX );
	gsl_vector_complex_set( vec,     chanX, gsl_complex_mul_real( wq_f,  chanN_ ) );
	gsl_vector_complex_set( vecConj, chanX, gsl_complex_mul_real( wqc_f, chanN_ ) );
	for(int n=0;n<NC-1;n++){
	  double valJ = -2.0 * M_PI * (fshift+fbinX) * samplerate * gsl_matrix_get(delaysIs, n, chanX) / fftLen_;
	  gsl_vector_complex_set(pWj[n],     chanX, gsl_complex_polar(1.0,  valJ) );
	  gsl_vector_complex_set(pWjConj[n], chanX, gsl_complex_polar(1.0, -valJ) );
	}
      }
      if( false==calc_null_beamformer_( vec, pWj, chanN_, NC ) ){
	throw j_error("calc_null_beamformer_() failed\n");
      }
      if( false==calc_null_beamformer_( vecConj, pWjConj, chanN_, NC ) ){
	throw j_error("calc_null_beamformer_() failed\n");
      }
    }
  } else {
    gsl_vector_complex* vec;
    gsl_vector_complex* vecConj;

    // calculate weights of a direct component
    vec = wq_[0];
    for (unsigned chanX = 0; chanX < chanN_; chanX++)
      gsl_vector_complex_set( vec, chanX, gsl_complex_rect( 1.0/chanN_, 0.0 ) );

    // use the property of the symmetry : wq[1] = wq[fftLen-1]*, wq[2] = wq[fftLen-2]*,...
    for (unsigned fbinX = 1; fbinX < fftLen2; fbinX++) {
      vec     = wq_[fbinX];
      vecConj = wq_[fftLen_ - fbinX];
      for (unsigned chanX = 0; chanX < chanN_; chanX++) {
	gsl_complex wq_f  = gsl_vector_complex_get( vec,     chanX );
	//gsl_complex wqc_f = gsl_vector_complex_get( vecConj, chanX );
	gsl_vector_complex_set( vec,     chanX, gsl_complex_mul_real( wq_f,  chanN_ ) );
	//gsl_vector_complex_set( vecConj, chanX, gsl_complex_mul_real( wqc_f, chanN_ ) );
 	for(int n=0;n<NC-1;n++){
	  double valJ = -2.0 * M_PI * fbinX * samplerate * gsl_matrix_get(delaysIs, n, chanX) / fftLen_;
	  gsl_vector_complex_set(pWj[n],     chanX, gsl_complex_polar(1.0,  valJ) );
	  //gsl_vector_complex_set(pWjConj[n], chanX, gsl_complex_polar(1.0, -valJ) );
	}
      }
      if( false==calc_null_beamformer_( vec, pWj, chanN_, NC ) ){
	throw j_error("calc_null_beamformer_() failed\n");
      }
      //if( false==calc_null_beamformer_( vecConj, pWjConj, chanN_, NC ) ){
      //throw j_error("calc_null_beamformer_() failed\n");
      //}
    }

    // for exp(-j*pi)
    vec     = wq_[fftLen2];
    for (unsigned chanX = 0; chanX < chanN_; chanX++) {
      gsl_complex wq_f  = gsl_vector_complex_get( vec, chanX );
      gsl_vector_complex_set( vec, chanX, gsl_complex_mul_real( wq_f, chanN_ ) );
      for(int n=0;n<NC-1;n++){
	double val = -M_PI * samplerate *  gsl_matrix_get(delaysIs, n, chanX);
	gsl_vector_complex_set(vec, chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), chanN_ ) );
      }
      if( false==calc_null_beamformer_( vec, pWj, chanN_, NC ) ){
	throw j_error("calc_null_beamformer_() failed\n");
      }
    }

  }

  // calculate a blocking matrix for each frequency bin.
  if( true == isGSC ){
    for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++) {
      if( false==calc_blocking_matrix_( wq_[fbinX], NC, B_[fbinX] ) ){
	throw j_error("calc_blocking_matrix_() failed\n");
      }
    }
  }

  for(int n=0;n<NC-1;n++){
    gsl_vector_complex_free( pWj[n] );
    gsl_vector_complex_free( pWjConj[n] );
  }
  delete [] pWj;
  delete [] pWjConj;

}

/**
   @brief set an active weight vector for frequency bin 'fbinX' and calculate B[fbinX] * wa[fbinX].

   @param int fbinX[in] a frequency bin.
   @param const gsl_vector* packedWeight[in] a packed weight vector.
 */
void BeamformerWeights::calcSidelobeCancellerP_f( unsigned fbinX, const gsl_vector* packedWeight )
{

  if( packedWeight->size != ( 2 * ( chanN_ - NC_ ) ) )
    throw jdimension_error("the size of an active weight vector must be %d but it is %d\n",
			   ( 2 * ( chanN_ - NC_ ) ), packedWeight->size );

  if( fbinX >= fftLen_ )
    throw jdimension_error("Must be a frequency bin %d < the length of FFT %d\n",
			   fbinX, fftLen_ );

  // copy an active weight vector to a member.
  gsl_complex val;

  for (unsigned chanX = 0; chanX < chanN_ - NC_ ; chanX++) {
    GSL_SET_COMPLEX( &val, gsl_vector_get( packedWeight, 2*chanX ), gsl_vector_get( packedWeight, 2*chanX+1 ) );
    gsl_vector_complex_set( wa_[fbinX], chanX, val );
  }

  // calculate B[f] * wa[f]
  // gsl_vector_complex_set_zero( wl_[fbinX] );
  gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), B_[fbinX], wa_[fbinX], gsl_complex_rect(0.0,0.0), wl_[fbinX] );
}

void BeamformerWeights::calcSidelobeCancellerU_f( unsigned fbinX, const gsl_vector_complex* wa )
{
  if( fbinX >= fftLen_ )
    throw jdimension_error("Must be a frequency bin %d < the length of FFT %d\n",
			   fbinX, fftLen_ );

  // copy an active weight vector to a member.
  for (unsigned chanX = 0; chanX < chanN_ - NC_ ; chanX++) {
    gsl_vector_complex_set( wa_[fbinX], chanX, gsl_vector_complex_get( wa, chanX  ) );
  }
  
  // calculate B[f] * wa[f]
  // gsl_vector_complex_set_zero( wl_[fbinX] );
  gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), B_[fbinX], wa_[fbinX], gsl_complex_rect(0.0,0.0), wl_[fbinX] );
}

/**
   @brief multiply each weight of a beamformer with exp(j*k*pi), transform them back into time domain and write the coefficients multiplied with a window function.

   @param const String& fn[in]
   @param unsigned winType[in]
 */
bool BeamformerWeights::write_fir_coeff( const String& fn, unsigned winType )
{
  if( wq_ == NULL || NULL == wl_ ){
    printf("BeamformerWeights::write_fir_coeff :\n");
    return false;
  }

  unsigned fftLen2 = fftLen_ / 2;
  FILE* fp = btk_fopen(fn.c_str(), "w");

  if( NULL==fp ){
    printf("could not open %s\n",fn.c_str());
    return false;
  }

  fprintf(fp,"%d %d\n", chanN_, fftLen_ );
  gsl_vector *window = get_window( winType, fftLen_ );
  double *weights_n = new double[fftLen_*2];

  if( false== halfBandShift_ ){

    for( unsigned chanX = 0 ; chanX < chanN_ ; chanX++ ){
      for( unsigned fbinX = 0 ; fbinX <= fftLen2 ; fbinX++ ){
	// calculate wq(f) - B(f) wa(f) at each channel
	gsl_complex wq_fn = gsl_vector_complex_get( wq_[fbinX], chanX );
	gsl_complex wl_fn = gsl_vector_complex_get( wl_[fbinX], chanX );
	gsl_complex wH_fn = gsl_complex_conjugate( gsl_complex_sub( wq_fn, wl_fn ) ) ;
	gsl_complex H_f   = gsl_complex_polar(1.0,  M_PI * (fbinX+1) );
	gsl_complex val   = gsl_complex_mul( H_f, wH_fn ); // shift fftLen/2

	weights_n[2*fbinX]   = GSL_REAL( val );
	weights_n[2*fbinX+1] = GSL_IMAG( val );
	if( fbinX > 0 && fbinX < fftLen2 ){
	  weights_n[2*(fftLen_-fbinX)]   = GSL_REAL( gsl_complex_conjugate(val) );
	  weights_n[2*(fftLen_-fbinX)+1] = GSL_IMAG( gsl_complex_conjugate(val) );
	}
      }
      gsl_fft_complex_radix2_inverse( weights_n, /* stride= */ 1, fftLen_ );
      for( unsigned fbinX = 0 ; fbinX < fftLen_ ; fbinX++ ){// multiply a window
	double coeff = gsl_vector_get(window,fbinX)* weights_n[2*fbinX];
	fprintf(fp,"%e ",coeff);
      }
      fprintf(fp,"\n");

    }//for( unsigned chanX = 0 ; chanX < chanN_ ; chanX++ ){

  }

  gsl_vector_free( window );
  delete [] weights_n;
  fclose(fp);

  return(true);
}

void BeamformerWeights::setQuiescentVector( unsigned fbinX, gsl_vector_complex *wq_f, bool isGSC )
{
  for(unsigned chanX=0;chanX<chanN_;chanX++)
    gsl_vector_complex_set( wq_[fbinX], chanX, gsl_vector_complex_get( wq_f, chanX ) );
  
  if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      if( false==calc_blocking_matrix_(  wq_[fbinX], NC_, B_[fbinX] ) ){
	throw j_error("calc_blocking_matrix_() failed\n");
      }
  }
}

void BeamformerWeights::setQuiescentVectorAll( gsl_complex z, bool isGSC )
{
  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++){
    gsl_vector_complex_set_all( wq_[fbinX], z );
      if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
	if( false==calc_blocking_matrix_(  wq_[fbinX], NC_, B_[fbinX] ) ){
	  throw j_error("calc_blocking_matrix_() failed\n");
	}
      }
  }
}

/**
   @brief calculate the blocking matrix at the frequency bin and 
          set results to the internal member B_[].
          The quiescent vector has to be provided
   @param unsigned fbinX[in]
 */
void BeamformerWeights::calcBlockingMatrix( unsigned fbinX )
{
  if( false==calc_blocking_matrix_( wq_[fbinX], NC_, B_[fbinX] ) ){
    throw j_error("calc_blocking_matrix_() failed\n");
  }
}

void BeamformerWeights::alloc_weights_()
{
  wq_ = new gsl_vector_complex*[fftLen_];
  wa_ = new gsl_vector_complex*[fftLen_];
  wl_ = new gsl_vector_complex*[fftLen_];
  ta_ = new gsl_vector_complex*[fftLen_];
  B_  = new gsl_matrix_complex*[fftLen_];
  CSDs_  = new gsl_vector_complex*[fftLen_];
  wp1_   = gsl_vector_complex_alloc( fftLen_ );

  for (unsigned i = 0; i < fftLen_; i++){
    wq_[i] = gsl_vector_complex_alloc( chanN_ );
    wl_[i] = gsl_vector_complex_alloc( chanN_ );
    ta_[i] = gsl_vector_complex_alloc( chanN_ );
    CSDs_[i] = gsl_vector_complex_alloc( chanN_ * chanN_ );
    if( NULL == wq_[i] || NULL == wl_[i] || NULL == ta_[i] || NULL == CSDs_[i] )
      throw jallocation_error("gsl_matrix_complex_alloc failed\n");
    gsl_vector_complex_set_zero( wq_[i] );
    gsl_vector_complex_set_zero( wl_[i] );
    gsl_vector_complex_set_zero( ta_[i] );
    gsl_vector_complex_set_zero( CSDs_[i] );

    if( chanN_ == 1 || chanN_ == NC_ ){
      B_[i]  = NULL;
      wa_[i] = NULL;
    }
    else{
      B_[i]  = gsl_matrix_complex_alloc( chanN_, chanN_ - NC_ );
      wa_[i] = gsl_vector_complex_alloc( chanN_ - NC_ );
      if( NULL == wa_[i] || NULL == B_[i] )
        throw jallocation_error("gsl_matrix_complex_alloc failed\n");
      gsl_matrix_complex_set_zero( B_[i]  );
      gsl_vector_complex_set_zero( wa_[i] );
    }

  }
  gsl_vector_complex_set_zero( wp1_ );
}

void BeamformerWeights::free_weights_()
{
  if( NULL != wq_ ){
    for(unsigned fbinX=0;fbinX<fftLen_;fbinX++)
      gsl_vector_complex_free( wq_[fbinX] );
    delete [] wq_;
    wq_ = NULL;
  }

  if( NULL != wa_ ){
    for(unsigned fbinX=0;fbinX<fftLen_;fbinX++){
      if( NULL != wa_[fbinX] )
	gsl_vector_complex_free( wa_[fbinX] );
    }
    delete [] wa_;
    wa_ = NULL;
  }

  if( NULL != B_ ){
    for(unsigned fbinX=0;fbinX<fftLen_;fbinX++){
      if( NULL != B_[fbinX] )
        gsl_matrix_complex_free( B_[fbinX] );
    }
    delete [] B_;
     B_ = NULL;
  }

  if( NULL != wl_ ){
    for(unsigned fbinX=0;fbinX<fftLen_;fbinX++)
      gsl_vector_complex_free( wl_[fbinX] );
    delete [] wl_;
    wl_ = NULL;
  }

  if( NULL != ta_ ){
    for(unsigned fbinX=0;fbinX<fftLen_;fbinX++)
      gsl_vector_complex_free( ta_[fbinX] );
    delete [] ta_;
    ta_ = NULL;
  }

  if( NULL!=CSDs_ ){
    for (unsigned i = 0; i < fftLen_; i++)
      gsl_vector_complex_free(CSDs_[i]);
    delete [] CSDs_;
    CSDs_ = NULL;
  }

  if( NULL!=wp1_ ){
    gsl_vector_complex_free(wp1_);
    wp1_ = NULL;
  }
}

void BeamformerWeights::setTimeAlignment()
{
  for (unsigned i = 0; i < fftLen_; i++){
    gsl_vector_complex_memcpy( ta_[i], wq_[i] );
  }
}


// ----- members for class `SubbandBeamformer' -----
//
SubbandBeamformer::SubbandBeamformer(unsigned fftLen, bool halfBandShift, const String& nm)
  : VectorComplexFeatureStream(fftLen, nm),
    snapshot_array_(NULL),
    fftLen_(fftLen),
    fftLen2_(fftLen/2),
    halfBandShift_(halfBandShift)
{}

SubbandBeamformer::~SubbandBeamformer()
{
  if(  (int)channelList_.size() > 0 )
    channelList_.erase( channelList_.begin(), channelList_.end() );
}

void SubbandBeamformer::set_channel(VectorComplexFeatureStreamPtr& chan)
{
  channelList_.push_back(chan);
}

void SubbandBeamformer::clear_channel()
{
  if(  (int)channelList_.size() > 0 )
    channelList_.clear();
  snapshot_array_ = NULL;
}

const gsl_vector_complex* SubbandBeamformer::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  increment_();
  return vector_;
}

void SubbandBeamformer::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- members for class `SubbandDS' -----
//
SubbandDS::SubbandDS(unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandBeamformer( fftLen, halfBandShift, nm )
{
  bfweight_vec_.clear();
}


SubbandDS::~SubbandDS()
{
  for(unsigned i=0;i<bfweight_vec_.size();i++)
    delete bfweight_vec_[i];
}

void SubbandDS::clear_channel()
{
  if(  (int)channelList_.size() > 0 )
    channelList_.clear();
  for(unsigned i=0;i<bfweight_vec_.size();i++)
    delete bfweight_vec_[i];
  bfweight_vec_.clear();
  snapshot_array_ = NULL;
}

/**
   @brief calculate an array manifold vectors for the delay & sum beamformer.
   @param float samplerate[in]
   @param const gsl_vector* delays[in] delaysT[chanN]
 */
void SubbandDS::calc_array_manifold_vectors(float samplerate, const gsl_vector* delays)
{
  this->alloc_bfweight_( 1, 1 );
  bfweight_vec_[0]->calcMainlobe( samplerate, delays, false );
}

/**
   @brief you can put 2 constraints. You can constrain a beamformer to preserve a target signal and, at the same time, to suppress an interference signal.
   @param float samplerate[in]
   @param const gsl_vector*  delaysT[in] delaysT[chanN]
   @param const gsl_vector** delaysJ[in] delaysJ[chanN]
 */
void SubbandDS::calc_array_manifold_vectors_2(float samplerate, const gsl_vector* delaysT, const gsl_vector* delaysJ )
{
  this->alloc_bfweight_( 1, 2 );
  bfweight_vec_[0]->calcMainlobe2( samplerate, delaysT, delaysJ, false );
}

/**
   @brief you can put multiple constraints. For example, you can constrain a beamformer to preserve a target signal and, at the same time, to suppress interference signals.
   @param float samplerate[in]
   @param const gsl_vector* delaysT[in] delaysT[chanN]
   @param const gsl_matrix* delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
 */
void SubbandDS::calc_array_manifold_vectors_n(float samplerate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC )
{
  this->alloc_bfweight_( 1, NC );
  bfweight_vec_[0]->calcMainlobeN( samplerate, delaysT, delaysJ, NC, false );
}

void SubbandDS::alloc_image_()
{
  if( NULL == snapshot_array_ )
    snapshot_array_ = new SnapShotArray( fftLen_, chanN() );
}

void SubbandDS::alloc_bfweight_( int nSource, int NC )
{
  for(unsigned i=0;i<bfweight_vec_.size();i++){
    delete bfweight_vec_[i];
  }
  bfweight_vec_.resize(nSource);
  for(unsigned i=0;i<bfweight_vec_.size();i++){
    bfweight_vec_[i] = new BeamformerWeights( fftLen_, chanN(), halfBandShift_, NC );
  }
  this->alloc_image_();
}

#define MINFRAMES 0 // the number of frames for estimating CSDs.
const gsl_vector_complex* SubbandDS::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  if( 0 == bfweight_vec_.size() ){
    throw j_error("call calc_array_manifold_vectorsX() once\n");
  }

  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  const gsl_vector_complex* arrayManifold_f;
  gsl_complex val;
  unsigned fftLen = fftLen_;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f      = snapshot_array_->snapshot(fbinX);
      arrayManifold_f = bfweight_vec_[0]->wq_f(fbinX);
      gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
      gsl_vector_complex_set(vector_, fbinX, val);
#ifdef  _MYDEBUG_
      if ( fbinX % 100 == 0 ){
	fprintf(stderr,"fbinX %d\n",fbinX );
	for (unsigned chX = 0; chX < chanN(); chX++)
	  fprintf(stderr,"%f %f\n",GSL_REAL(  gsl_vector_complex_get( arrayManifold_f, chX ) ), GSL_IMAG(  gsl_vector_complex_get( arrayManifold_f, chX ) ) );
	fprintf(stderr,"VAL %f %f\n",GSL_REAL( val ), GSL_IMAG( val ) );
      }
#endif //_MYDEBUG_
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f      = snapshot_array_->snapshot(0);
    arrayManifold_f = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
    gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f      = snapshot_array_->snapshot(fbinX);
      arrayManifold_f = bfweight_vec_[0]->wq_f(fbinX);
      gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(vector_, fbinX, val);
	gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(vector_, fftLen2, val);
    }
  }

  increment_();
  return vector_;
}

void SubbandDS::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

void calc_all_delays(float x, float y, float z, const gsl_matrix* mpos, gsl_vector* delays)
{
  unsigned chanN = mpos->size1;

  for (unsigned chX = 0; chX < chanN; chX++) {
    double x_m = gsl_matrix_get(mpos, chX, 0);
    double y_m = gsl_matrix_get(mpos, chX, 1);
    double z_m = gsl_matrix_get(mpos, chX, 2);

    double delta = sqrt(x_m*x_m + y_m*y_m + z_m*z_m) / SSPEED;
    gsl_vector_set(delays, chX, delta);
  }

  // Normalize by delay of the middle element
  double mid = gsl_vector_get(delays, chanN/2);
  for (unsigned chX = 0; chX < chanN; chX++)
    gsl_vector_set(delays, chX, gsl_vector_get(delays, chX) - mid);
}


void calc_product(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product)
{
  gsl_complex a = gsl_complex_rect(1,0);
  gsl_complex b = gsl_complex_rect(0,0);
  gsl_blas_zgemv(CblasTrans, a, gs_W, synthesisSamples, b, product);
}

/**
   @brief calculate the output of the GSC beamformer for a frequency bin, that is,
          Y(f) = ( wq(f) - B(f) wa(f) )_H * X(f)

   @param gsl_vector_complex* snapShot[in] an input subband snapshot
   @param gsl_vector_gsl_vector_complex* wq[in/out] a quiescent weight vector
   @param gsl_vector_gsl_vector_complex* wa[in]     B * wa (sidelobe canceller)
   @param gsl_complex *pYf[out] (gsl_complex *)the output of the GSC beamformer
   @param bool normalizeWeight[in] Normalize the entire weight vector if true.
*/
void calc_gsc_output( const gsl_vector_complex* snapShot,
                      gsl_vector_complex* wl_f, gsl_vector_complex* wq_f,
                      gsl_complex *pYf, bool normalizeWeight )
{
  unsigned chanN = wq_f->size;
  gsl_complex wq_fn, wl_fn;
  gsl_vector_complex *myWq_f = gsl_vector_complex_alloc( chanN );

  if( wq_f->size != wl_f->size ){
    throw  j_error("calc_gsc_output:The lengths of weight vectors must be the same.\n");
  }

  // calculate wq(f) - B(f) wa(f)
  //gsl_vector_complex_sub( wq_f, wl_f );
  for(unsigned i=0;i<chanN;i++){
    wq_fn = gsl_vector_complex_get( wq_f ,i );
    wl_fn = gsl_vector_complex_get( wl_f ,i );
    gsl_vector_complex_set( myWq_f, i, gsl_complex_sub( wq_fn, wl_fn ) );
  }

  /* normalize the entire weight */
  /* w <- w / ( ||w|| * chanN )  */
  if ( true==normalizeWeight ){
    double norm = gsl_blas_dznrm2( myWq_f );
    for(unsigned i=0;i<chanN;i++){
      gsl_complex val;
      val = gsl_complex_div_real( gsl_vector_complex_get( myWq_f, i ), norm * chanN);
      gsl_vector_complex_set( myWq_f, i, val );
    }
    //double norm1 = gsl_blas_dznrm2( myWq_f ); fprintf(stderr,"norm %e %e\n",norm,norm1 );
  }

  // calculate  ( wq(f) - B(f) wa(f) )^H * X(f)
  gsl_blas_zdotc( myWq_f, snapShot, pYf );
  gsl_vector_complex_free( myWq_f );
}

SubbandGSC::~SubbandGSC()
{}

/**
   @brief  calculate the outputs of the GSC beamformer at each frame
 */
const gsl_vector_complex* SubbandGSC::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = fftLen_;

  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_gsc_weights_X() once\n");
  }

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);
      
      calc_gsc_output( snapShot_f, wl_f,  wq_f, &val, normalize_weight_ );
      gsl_vector_complex_set(vector_, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = snapshot_array_->snapshot(0);
    wq_f       = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(vector_, 0, val);
    //wq_f = _bfWeights->wq_f(0);
    //wl_f = _bfWeights->wl_f(0);
    //calc_gsc_output( snapShot_f, chanN(), wl_f, wq_f, &val );
    //gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);

      calc_gsc_output( snapShot_f, wl_f, wq_f, &val, normalize_weight_ );
      if( fbinX < fftLen2 ){
        gsl_vector_complex_set(vector_, fbinX,           val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2, val);
    }
  }

  increment_();

  return vector_;
}

void SubbandGSC::set_quiescent_weights_f(unsigned fbinX, const gsl_vector_complex * srcWq)
{
  this->alloc_bfweight_( 1, 1 );
  gsl_vector_complex* destWq = bfweight_vec_[0]->wq_f(fbinX);
  gsl_vector_complex_memcpy( destWq, srcWq );
  bfweight_vec_[0]->calcBlockingMatrix( fbinX );
}

void SubbandGSC::calc_gsc_weights(float samplerate, const gsl_vector* delaysT)
{
  this->alloc_bfweight_( 1, 1 );
  bfweight_vec_[0]->calcMainlobe( samplerate, delaysT, true );
}

void SubbandGSC::calc_gsc_weights_2( float samplerate, const gsl_vector* delaysT, const gsl_vector* delaysI )
{
  this->alloc_bfweight_( 1, 2 );
  bfweight_vec_[0]->calcMainlobe2( samplerate, delaysT, delaysI, true );
}

/**
   @brief calculate the quescent vectors with N linear constraints
   @param float samplerate[in]
   @param const gsl_vector* delaysT[in] delaysT[chanN]
   @param const gsl_matrix* delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
   @note you can put multiple constraints. For example, you can constrain a beamformer to preserve a target signal and, at the same time, to suppress interference signals.
 */
void SubbandGSC::calc_gsc_weights_n( float samplerate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC )
{
  if( NC == 2 ){
    gsl_vector* delaysI = gsl_vector_alloc( chanN() );

    for( unsigned i=0;i<chanN();i++)
      gsl_vector_set( delaysI, i, gsl_matrix_get( delaysIs, 0, i ) );
    this->alloc_bfweight_( 1, 2 );
    bfweight_vec_[0]->calcMainlobe2( samplerate, delaysT, delaysI, true );

    gsl_vector_free(delaysI);
  }
  else{
    this->alloc_bfweight_( 1, NC );
    bfweight_vec_[0]->calcMainlobeN( samplerate, delaysT, delaysIs, NC, true );
  }
}

bool SubbandGSC::write_fir_coeff( const String& fn, unsigned winType )
{
  if( 0 == bfweight_vec_.size() ){
    fprintf(stderr,"call calc_array_manifold_vectorsX() once\n");
    return(false);
  }
  return( bfweight_vec_[0]->write_fir_coeff(fn,winType) );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SubbandGSC::set_active_weights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_gsc_weights_x() once\n");
  }

  bfweight_vec_[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

void SubbandGSC::zero_active_weights()
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_gsc_weights_x() once\n");
  }

  gsl_vector_complex *wa = gsl_vector_complex_calloc( chanN() - bfweight_vec_[0]->NC() );
  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++){
   bfweight_vec_[0]->calcSidelobeCancellerU_f( fbinX, wa );
  }
  gsl_vector_complex_free( wa );
}

/**
   @brief solve the scaling ambiguity
   
   @param gsl_matrix_complex *W[in/out] MxN unmixing matrix. M and N are the number of sources and sensors, respectively.
   @param float dThreshold[in]
 */
static bool scaling_( gsl_matrix_complex *W, float dThreshold = 1.0E-8 )
{
  size_t M = W->size1; // the number of sources
  size_t N = W->size2; // the number of sensors
  gsl_matrix_complex *Wp = gsl_matrix_complex_alloc( N, M );

  pseudoinverse( W, Wp, dThreshold );

#if 0
  // W <= diag[W+] * W, where W+ is a pseudoinverse matrix.
  for ( size_t i = 0; i < M; i++ ){// src
    for ( size_t j = 0; j < N; j++ ){// mic
      //gsl_complex dii   = gsl_vector_complex_get( diagWp, i );
      gsl_complex dii   = gsl_matrix_complex_get(Wp, i, i );
      gsl_complex W_ij  = gsl_matrix_complex_get( W , i ,j );
      gsl_matrix_complex_set( W, i, j, gsl_complex_mul( dii, W_ij ) );
    }
  }
#else
  int stdsnsr = (int) N / 2;
  for ( size_t i = 0; i < M; i++ ){// src
    for ( size_t j = 0; j < N; j++ ){// mic
      gsl_complex Wp_mi = gsl_matrix_complex_get( Wp, stdsnsr, i );
      gsl_complex W_ij  = gsl_matrix_complex_get( W , i ,j );

      gsl_matrix_complex_set( W, i, j, gsl_complex_mul( Wp_mi, W_ij ) );
    }
  }
#endif

  gsl_matrix_complex_free( Wp );
  return(true);
}

/**
   @brief 
   @param unsigned fftLen[in] The point of the FFT
   @param bool halfBandShift[in]
   @param float mu[in] the forgetting factor for the covariance matrix
   @param float sigma2[in] the amount of diagonal loading
 */
SubbandGSCRLS::SubbandGSCRLS(unsigned fftLen, bool halfBandShift, float mu, float sigma2, const String& nm ):
  SubbandGSC( fftLen, halfBandShift, nm ),
  Zf_(NULL),
  wa_(NULL),
  mu_(mu),
  alpha_(-1.0),
  qctype_(NO_QUADRATIC_CONSTRAINT),
  is_wa_updated_(true)
{
  gz_ = new gsl_vector_complex*[fftLen];
  Pz_ = new gsl_matrix_complex*[fftLen];
  diagonal_weights_ = new float[fftLen];
  for (unsigned fbinX = 0; fbinX < fftLen; fbinX++){
    gz_[fbinX] = NULL;
    Pz_[fbinX] = NULL;
    diagonal_weights_[fbinX] = sigma2;
  }
  PzH_Z_ = NULL;
  _I     = NULL;
  mat1_  = NULL;
}

SubbandGSCRLS::~SubbandGSCRLS()
{
  free_subbandGSCRLS_image_();
  delete [] gz_;
  delete [] Pz_;
  delete [] diagonal_weights_;
}

void SubbandGSCRLS::init_precision_matrix( float sigma2 )
{
  free_subbandGSCRLS_image_();
  if( false == alloc_subbandGSCRLS_image_() )
    throw j_error("call calc_gsc_weights_x() once\n");

  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++){
    gsl_matrix_complex_set_zero( Pz_[fbinX] );
    for (unsigned chanX = 0; chanX < Pz_[fbinX]->size1; chanX++){
      gsl_matrix_complex_set( Pz_[fbinX], chanX, chanX, gsl_complex_rect( 1/sigma2, 0 ) );
    }
  }
}

void SubbandGSCRLS::set_precision_matrix( unsigned fbinX, gsl_matrix_complex *Pz )
{
  free_subbandGSCRLS_image_();
  if( false == alloc_subbandGSCRLS_image_() )
    throw j_error("call calc_gsc_weights_x() once\n");

  unsigned nChan = chanN();
  for (unsigned chanX = 0; chanX < nChan; chanX++) {
    for (unsigned chanY = 0; chanY < nChan; chanY++) {
      gsl_matrix_complex_set( Pz_[fbinX], chanX, chanY, gsl_matrix_complex_get(Pz, chanX, chanY) );
    }
  }
}

const gsl_vector_complex* SubbandGSCRLS::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = fftLen_;

  if( 0 == bfweight_vec_.size() )
    throw  j_error("call calc_gsc_weights_x() once\n");
  if( NULL == Zf_ )
    throw  j_error("set the precision matrix with init_precision_matrix() or set_precision_matrix()\n");

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX); chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == true ){
    throw  j_error("not yet implemented\n");
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = snapshot_array_->snapshot(0);
    wq_f       = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);

      calc_gsc_output( snapShot_f, wl_f, wq_f, &val, normalize_weight_ );
      if( fbinX < fftLen2 ){
        gsl_vector_complex_set(vector_, fbinX,           val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2, val);
    }

    this->update_active_weight_vector2_( frame_no );
  }

  increment_();

  return vector_;
}

void SubbandGSCRLS::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;

}

void SubbandGSCRLS::update_active_weight_vector2_( int frame_no )
{
  if( false == is_wa_updated_ )
    return;

  unsigned NC    = bfweight_vec_[0]->NC();
  unsigned nChan = chanN();
  gsl_matrix_complex** B  = bfweight_vec_[0]->B();
  gsl_vector_complex** old_wa = bfweight_vec_[0]->wa();

  for (unsigned fbinX = 1; fbinX <= fftLen_/2; fbinX++){
    const gsl_vector_complex *Xf = snapshot_array_->snapshot(fbinX);
    gsl_complex nu, de;

    // calc. output of the blocking matrix.    
    gsl_blas_zgemv( CblasConjTrans, gsl_complex_rect(1.0,0.0), B[fbinX], Xf, gsl_complex_rect(0.0,0.0), Zf_ );

    // calc. the gain vector 
    gsl_blas_zgemv( CblasConjTrans, gsl_complex_rect(1.0,0.0), Pz_[fbinX], Zf_, gsl_complex_rect(0.0,0.0), PzH_Z_ );
    gsl_blas_zgemv( CblasNoTrans,   gsl_complex_rect(1.0/mu_,0.0), Pz_[fbinX], Zf_, gsl_complex_rect(0.0,0.0), gz_[fbinX] );
    gsl_blas_zdotc( PzH_Z_, Zf_, &de );
    de = gsl_complex_add_real( gsl_complex_mul_real( de, 1.0/mu_ ), 1.0 );
    for(unsigned chanX =0;chanX<nChan-NC;chanX++){
      gsl_complex val;
      nu = gsl_vector_complex_get( gz_[fbinX], chanX );      
      val = gsl_complex_div( nu, de );
      gsl_vector_complex_set( gz_[fbinX], chanX, val );
    }

    // calc. the precision matrix
    for(unsigned chanX=0;chanX<nChan-NC;chanX++){
      for(unsigned chanY=0;chanY<nChan-NC;chanY++){
	gsl_complex oldPz, val1, val2;
	oldPz = gsl_matrix_complex_get( Pz_[fbinX], chanX, chanY );
	val1  = gsl_complex_mul( gsl_vector_complex_get( gz_[fbinX], chanX ), gsl_complex_conjugate( gsl_vector_complex_get( PzH_Z_, chanY ) ) );
	val2  = gsl_complex_mul_real( gsl_complex_sub( oldPz, val1 ),  1.0/mu_ );
	gsl_matrix_complex_set( Pz_[fbinX], chanX, chanY, val2 );
      }
    }

    { // update the active weight vecotr
      gsl_complex epA  = gsl_complex_conjugate( gsl_vector_complex_get(vector_, fbinX ) );

      gsl_matrix_complex_memcpy( mat1_, Pz_[fbinX] );
      gsl_matrix_complex_scale(  mat1_, gsl_complex_rect( - diagonal_weights_[fbinX], 0.0 ) );
      gsl_matrix_complex_add( mat1_, _I );
      gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), mat1_, old_wa[fbinX], gsl_complex_rect(0.0,0.0), wa_ );
      for(unsigned chanX=0;chanX<nChan-NC;chanX++){
	gsl_complex val1 = gsl_vector_complex_get( wa_, chanX );
	gsl_complex val2 = gsl_complex_mul( gsl_vector_complex_get( gz_[fbinX], chanX ), epA );
	gsl_vector_complex_set( wa_, chanX, gsl_complex_add( val1, val2 ) );
      }
      if( qctype_ == CONSTANT_NORM ){
	double nrmwa = gsl_blas_dznrm2( wa_ );
	for(unsigned chanX=0;chanX<nChan-NC;chanX++)
	  gsl_vector_complex_set( wa_, chanX, gsl_complex_mul_real( gsl_vector_complex_get( wa_, chanX), alpha_/nrmwa ) );
      }
      else if( qctype_ == THRESHOLD_LIMITATION ){
	double nrmwa = gsl_blas_dznrm2( wa_ );
	if( ( nrmwa * nrmwa ) >= alpha_ ){
	  for(unsigned chanX=0;chanX<nChan-NC;chanX++)
	    gsl_vector_complex_set( wa_, chanX, gsl_complex_mul_real( gsl_vector_complex_get( wa_, chanX), alpha_/nrmwa ) );
	}
      }
      //fprintf( stderr, "%d: %e\n", frame_no, gsl_blas_dznrm2 ( wa_ ) );
      bfweight_vec_[0]->calcSidelobeCancellerU_f( fbinX, wa_ );
    }
  }

}

/*
  @brief allocate image blocks for gz_[] and Pz_[].
 */
bool SubbandGSCRLS::alloc_subbandGSCRLS_image_()
{
  if( 0 == bfweight_vec_.size() ){
    fprintf(stderr,"call calc_weights_x() once\n");
    return false;
  }
  unsigned NC    = bfweight_vec_[0]->NC();
  unsigned nChan = chanN();

  Zf_ = gsl_vector_complex_calloc( nChan - NC );
  wa_ = gsl_vector_complex_calloc( nChan - NC );

  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++){
    Pz_[fbinX] = gsl_matrix_complex_calloc( nChan - NC , nChan - NC );
    gz_[fbinX] = gsl_vector_complex_calloc( nChan - NC );
    bfweight_vec_[0]->calcSidelobeCancellerU_f( fbinX, wa_ );
  }

  PzH_Z_ = gsl_vector_complex_calloc( nChan - NC );
  _I     = gsl_matrix_complex_alloc( nChan - NC, nChan - NC );
  gsl_matrix_complex_set_identity( _I );
  mat1_  = gsl_matrix_complex_alloc( nChan - NC, nChan - NC );

  return true;
}

void SubbandGSCRLS::free_subbandGSCRLS_image_()
{
  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++) {
    if( NULL != Pz_[fbinX] ){
      gsl_vector_complex_free( gz_[fbinX] );
      gsl_matrix_complex_free( Pz_[fbinX] );
      gz_[fbinX] = NULL;
      Pz_[fbinX] = NULL;
    }
  }

  if( NULL != Zf_ ){
    gsl_vector_complex_free( Zf_ );
    gsl_vector_complex_free( wa_ );
    Zf_ = NULL;
    wa_ = NULL;
    gsl_vector_complex_free( PzH_Z_ );
    gsl_matrix_complex_free( _I );
    gsl_matrix_complex_free( mat1_ );
    PzH_Z_ = NULL;
    _I     = NULL;
    mat1_  = NULL;
  }
}

// ----- definition for class `SubbandMMI' -----
//

SubbandMMI::~SubbandMMI()
{
  if( NULL != interference_outputs_ ){
    for (unsigned i = 0; i < nSource_; i++)
      gsl_vector_complex_free(interference_outputs_[i]);
    delete [] interference_outputs_;
  }
  if( NULL != avg_output_ )
    gsl_vector_complex_free( avg_output_ );
}

void SubbandMMI::use_binary_mask(float avgFactor, unsigned fwidth, unsigned type)
{
  avg_factor_ = avgFactor;
  fwidth_ = fwidth;
  use_binary_mask_ = true;
  binary_mask_type_ = type;
  interference_outputs_ = new gsl_vector_complex*[nSource_];
  for (unsigned i = 0; i < nSource_; i++)
    interference_outputs_[i] = gsl_vector_complex_alloc(fftLen_);

  avg_output_ = gsl_vector_complex_alloc(fftLen_);
  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++)
    gsl_vector_complex_set(avg_output_, fbinX, gsl_complex_rect( 0.0, 0.0 ));
}

/**
   @brief calculate a quiescent weight vector and blocking matrix for each frequency bin.
   @param double samplerate[in]
   @param const gsl_matrix* delayMat[in] delayMat[nSource][nChan]
 */
void SubbandMMI::calc_weights(float samplerate, const gsl_matrix* delayMat)
{
  this->alloc_bfweight_(nSource_, 1);
  gsl_vector *delaysT = gsl_vector_alloc(chanN());

  for( unsigned srcX=0;srcX<nSource_;srcX++){
    gsl_matrix_get_row(delaysT, delayMat, srcX);
    bfweight_vec_[srcX]->calcMainlobe(samplerate, delaysT, true);
  }

  gsl_vector_free(delaysT);
}

/**
   @brief calculate a quiescent weight vector with N constraints and blocking matrix for each frequency bin.
   @param float samplerate[in]
   @param const gsl_matrix* delayMat[in] delayMat[nSource][nChan]
   @param unsigned NC[in] the number of linear constraints
 */
void SubbandMMI::calc_weights_n( float samplerate, const gsl_matrix* delayMat, unsigned NC )
{
  if( 0 == bfweight_vec_.size() )
    this->alloc_bfweight_( nSource_, NC );

  gsl_vector* delaysT  = gsl_vector_alloc( chanN() );
  gsl_vector* tmpV     = gsl_vector_alloc( chanN() );
  gsl_matrix* delaysIs = gsl_matrix_alloc( NC - 1, chanN() ); // the number of interferences = 1

  for(unsigned srcX=0;srcX<nSource_;srcX++){
    gsl_matrix_get_row( delaysT, delayMat, srcX );
    for(unsigned srcY=0,i=0;i<NC-1;srcY++){
      if( srcY == srcX ) continue;
      gsl_matrix_get_row( tmpV, delayMat, srcY );
      gsl_matrix_set_row( delaysIs, i, tmpV );
      i++;
    }
    bfweight_vec_[srcX]->calcMainlobeN( samplerate, delaysT, delaysIs, NC, true );
  }

  gsl_vector_free( delaysT );
  gsl_vector_free( tmpV );
  gsl_matrix_free( delaysIs );
}

/**
   @brief set an active weight vector for each frequency bin, 
          calculate the entire weights of a lower branch, and make a NxM demixing matrix. 
          N is the number of sound sources and M is the number of channels.
   @param unsigned fbinX[in]
   @param const gsl_matrix* packedWeights[in] [nSource][nChan]
   @param int option[in] 0:do nothing, 1:solve scaling, 
 */
void SubbandMMI::set_active_weights_f(unsigned fbinX, const gsl_matrix* packedWeights, int option)
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_weights_x() once\n");
  }
  if( packedWeights->size1 != nSource_ ){
    throw  j_error("The number of columns must be the number of sources %d\n", nSource_);
  }

  {// calculate the entire weight of a lower branch.
    gsl_vector* packedWeight = gsl_vector_alloc( packedWeights->size2 );

    for(unsigned srcX=0;srcX<nSource_;srcX++){
      gsl_matrix_get_row( packedWeight, packedWeights, srcX );
      bfweight_vec_[srcX]->calcSidelobeCancellerP_f( fbinX, packedWeight );
    }
    gsl_vector_free( packedWeight );
  }


  {// make a demixing matrix and solve scaling ambiguity
    gsl_vector_complex* tmpV;
    gsl_matrix_complex* Wl_f = gsl_matrix_complex_alloc( nSource_, chanN() ); // a demixing matrix
    gsl_vector_complex* new_wl_f = gsl_vector_complex_alloc( chanN() );
    gsl_complex val;

    for(unsigned srcX=0;srcX<nSource_;srcX++){
      tmpV = bfweight_vec_[srcX]->wl_f( fbinX );
      gsl_matrix_complex_set_row( Wl_f, srcX, tmpV );
    }
    // conjugate a matrix
    for(unsigned i=0;i<Wl_f->size1;i++){// Wl_f[nSource_][nChan_]
       for(unsigned j=0;j<Wl_f->size2;j++){
	 val = gsl_matrix_complex_get( Wl_f, i, j );
	 gsl_matrix_complex_set( Wl_f, i, j, gsl_complex_conjugate( val ) );
       }
    }

    if( option == 1 ){
      if( false==scaling_( Wl_f, 1.0E-7 ) )
	fprintf(stderr,"%d : scaling is not performed\n", fbinX);
    }

    // put an updated vector back
    for(unsigned srcX=0;srcX<nSource_;srcX++){
      gsl_matrix_complex_get_row( new_wl_f, Wl_f, srcX );
      for(unsigned i=0;i<new_wl_f->size;i++){// new_wl_f[nChan_]
	val = gsl_vector_complex_get( new_wl_f, i );
	gsl_vector_complex_set( new_wl_f, i, gsl_complex_conjugate( val ) );
      }
      bfweight_vec_[srcX]->setSidelobeCanceller_f( fbinX, new_wl_f );
    }

    gsl_vector_complex_free( new_wl_f );
    gsl_matrix_complex_free( Wl_f );
  }

}

/**
   @brief set active weight matrix & vector for each frequency bin, 
          calculate the entire weights of a lower branch, and make a Ns x Ns demixing matrix, 
          where Ns is the number of sound sources.
   @param unsigned fbinX[in]
   @param const gsl_vector* pkdWa[in] active weight matrices [2 * nSrc * (nChan-NC) * nSrc]
   @param const gsl_vector* pkdwb[in] active weight vectors  [2 * nSrc * nSrc]
   @param int option[in] 0:do nothing, 1:solve scaling, 
 */
void SubbandMMI::set_hi_active_weights_f( unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option )
{

  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_weights_x() once\n");
  }
  unsigned NC = bfweight_vec_[0]->NC();
  if( pkdWa->size != (2*nSource_*(chanN()-NC)*nSource_) ){
    throw  j_error("The size of the 2nd arg must be 2 * %d * %d * %d\n", nSource_, (chanN()-NC), nSource_ );
  }
  if( pkdwb->size != (2*nSource_*nSource_) ){
    throw  j_error("The size of the 3rd arg must be 2 * %d * %d\n", nSource_, nSource_ );
  }

  {// make a demixing matrix and solve scaling ambiguity
    gsl_matrix_complex** Wa_f = new gsl_matrix_complex*[nSource_];
    gsl_vector_complex** wb_f = new gsl_vector_complex*[nSource_];
    gsl_matrix_complex*  Wc_f = gsl_matrix_complex_alloc( nSource_, nSource_ );// concatenate wb_f[] and conjugate it
    gsl_vector_complex*  we_f = gsl_vector_complex_alloc( chanN()-NC ); // the entire weight ( Wa_f[] * wb_f[] )
    const gsl_complex alpha = gsl_complex_rect(1.0,0.0);
    const gsl_complex beta  = gsl_complex_rect(0.0,0.0);

    for(unsigned srcX=0;srcX<nSource_;srcX++){
      Wa_f[srcX] = gsl_matrix_complex_alloc( chanN()-NC, nSource_ ); // active matrices
      wb_f[srcX] = gsl_vector_complex_alloc( nSource_ ); // active vectors
    }

    // unpack vector data for active matrices
    for(unsigned srcX=0,i=0;srcX<nSource_;srcX++){
      for(unsigned chanX=0;chanX<chanN()-NC;chanX++){
	for(unsigned srcY=0;srcY<nSource_;srcY++){
	  gsl_complex val = gsl_complex_rect( gsl_vector_get( pkdWa, 2*i ), gsl_vector_get( pkdWa, 2*i+1 ) );
	  gsl_matrix_complex_set( Wa_f[srcX], chanX, srcY, val );
	  i++;
	}
      }
    }
    // unpack vector data for active vectors and make a demixing matrix
    for(unsigned srcX=0,i=0;srcX<nSource_;srcX++){
      for(unsigned srcY=0;srcY<nSource_;srcY++){
	gsl_complex val = gsl_complex_rect( gsl_vector_get( pkdwb, 2*i ), gsl_vector_get( pkdwb, 2*i+1 ) );
	gsl_matrix_complex_set( Wc_f, srcX, srcY, gsl_complex_conjugate( val ) );
	//gsl_matrix_complex_set( Wc_f, srcX, srcY, val );
	i++;
      }
    }

    if( option == 1 ){
      if( false==scaling_( Wc_f, 1.0E-7 ) )
        fprintf(stderr,"%d : scaling is not performed\n", fbinX);
    }

    // put an updated vector back and calculate the entire active weights
    for(unsigned srcX=0;srcX<nSource_;srcX++){
      for(unsigned srcY=0;srcY<nSource_;srcY++){
	gsl_complex val = gsl_matrix_complex_get( Wc_f, srcX, srcY );
	gsl_vector_complex_set( wb_f[srcX], srcY, gsl_complex_conjugate( val ) );
	//gsl_vector_complex_set( wb_f[srcX], srcY, val );
      }
      gsl_blas_zgemv( CblasNoTrans, alpha, Wa_f[srcX], wb_f[srcX], beta, we_f );
      bfweight_vec_[srcX]->calcSidelobeCancellerU_f( fbinX, we_f );
    }

    for(unsigned srcX=0;srcX<nSource_;srcX++){
      gsl_matrix_complex_free( Wa_f[srcX] );
      gsl_vector_complex_free( wb_f[srcX] );
    }
    
    delete [] Wa_f;
    delete [] wb_f;
    gsl_matrix_complex_free( Wc_f );
    gsl_vector_complex_free( we_f );
  }

}

/**
   @brief  calculate the outputs of the GSC beamformer at each frame
 */
const gsl_vector_complex* SubbandMMI::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = fftLen_;

  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_weights X() once\n");
  }

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[targetSourceX_]->wq_f(fbinX);
      wl_f = bfweight_vec_[targetSourceX_]->wl_f(fbinX);

      calc_gsc_output( snapShot_f, wl_f,  wq_f, &val );
      gsl_vector_complex_set(vector_, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = snapshot_array_->snapshot(0);
    wq_f       = bfweight_vec_[targetSourceX_]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(vector_, 0, val);
    //wq_f = bfweight_vec_[targetSourceX_]->wq_f(0);
    //wl_f = bfweight_vec_[targetSourceX_]->wl_f(0);      
    //calc_gsc_output( snapShot_f, chanN(), wl_f,  wq_f, &val );
    //gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[targetSourceX_]->wq_f(fbinX);
      wl_f = bfweight_vec_[targetSourceX_]->wl_f(fbinX);

      calc_gsc_output( snapShot_f, wl_f,  wq_f, &val );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(vector_, fbinX,           val);
	gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(vector_, fftLen2, val);
    }
  }

  {// post-filtering
    float alpha;
    gsl_vector_complex** wq  = bfweight_vec_[targetSourceX_]->arrayManifold(); // use a D&S beamformer output as a clean signal.
    gsl_vector_complex*  wp1 = bfweight_vec_[0]->wp1();
    gsl_vector_complex** prevCSDs = bfweight_vec_[targetSourceX_]->CSDs();

    if(  frame_no_ > 0 )
      alpha =  alpha_;
    else
      alpha = 0.0;

    if( (int)TYPE_APAB & pftype_ ){
      ApabFilter( wq, snapshot_array_, fftLen, chanN(), halfBandShift_, vector_, chanN()/2 );
    }
    else if( (int)TYPE_ZELINSKI1_REAL & pftype_ || (int)TYPE_ZELINSKI1_ABS & pftype_ ){

      if( (int)TYPE_ZELINSKI2 & pftype_  )
	wq =  bfweight_vec_[targetSourceX_]->wq(); // just use a beamformer output as a clean signal.

      if( frame_no_ < MINFRAMES )// just update cross spectral densities
	ZelinskiFilter( wq, snapshot_array_, halfBandShift_, vector_, prevCSDs, wp1, alpha, (int)NO_USE_POST_FILTER);
      else
	ZelinskiFilter( wq, snapshot_array_, halfBandShift_, vector_, prevCSDs, wp1, alpha, pftype_ );
    }
  }

  if( use_binary_mask_==true ){// binary mask
    this->calc_interference_outputs_();

    if( binary_mask_type_ == 0 )
      gsl_vector_complex_memcpy( interference_outputs_[targetSourceX_], vector_);
    this->binary_masking_( interference_outputs_, targetSourceX_, vector_);
  }

  increment_();
  return vector_;
}


/**
   @brief construct beamformers for all the sources and get the outputs.
   @return Interference signals are stored in interference_outputs_[].
 */
void SubbandMMI::calc_interference_outputs_()
{
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned fftLen = fftLen_;

  if( halfBandShift_ == true ){
    for (unsigned srcX = 0; srcX < nSource_; srcX++) {
      if( binary_mask_type_ == 0 ){// store GSC's outputs in interference_outputs_[].
	if( srcX == targetSourceX_ )
	  continue;
	for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
	  snapShot_f = snapshot_array_->snapshot(fbinX);
	  wq_f = bfweight_vec_[srcX]->wq_f(fbinX);
	  wl_f = bfweight_vec_[srcX]->wl_f(fbinX);
	  calc_gsc_output( snapShot_f, wl_f,  wq_f, &val );
	  gsl_vector_complex_set( interference_outputs_[srcX], fbinX, val);
	}
      }
      else{ // store outputs of an upper branch interference_outputs_[].
	for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
	  snapShot_f = snapshot_array_->snapshot(fbinX);
	  wq_f = bfweight_vec_[srcX]->wq_f(fbinX);
	  gsl_blas_zdotc(wq_f, snapShot_f, &val);
	  gsl_vector_complex_set( interference_outputs_[srcX], fbinX, val);
	}
      }
    }// for (unsigned srcX = 0; srcX < nSource_; srcX++)
  }
  else{
    unsigned fftLen2 = fftLen/2;

    for (unsigned srcX = 0; srcX < nSource_; srcX++) {
      if( binary_mask_type_ == 0 ){// store GSC's outputs.
	if( srcX == targetSourceX_ )
	  continue;

	// calculate a direct component.
	snapShot_f = snapshot_array_->snapshot(0);
	wq_f       = bfweight_vec_[srcX]->wq_f(0);
	gsl_blas_zdotc( wq_f, snapShot_f, &val);
	gsl_vector_complex_set(interference_outputs_[srcX], 0, val);

	// calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
	for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
	  snapShot_f = snapshot_array_->snapshot(fbinX);
	  wq_f = bfweight_vec_[srcX]->wq_f(fbinX);
	  wl_f = bfweight_vec_[srcX]->wl_f(fbinX);
	  calc_gsc_output( snapShot_f, wl_f,  wq_f, &val );
	  if( fbinX < fftLen2 ){
	    gsl_vector_complex_set(interference_outputs_[srcX], fbinX, val);
	    gsl_vector_complex_set(interference_outputs_[srcX], fftLen_ - fbinX, gsl_complex_conjugate(val) );
	  }
	  else
	    gsl_vector_complex_set(interference_outputs_[srcX], fftLen2, val);
	}// for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
      }
      else{ // store outputs of an upper branch interference_outputs_[].
	// calculate a direct component.
	snapShot_f = snapshot_array_->snapshot(0);
	wq_f       = bfweight_vec_[srcX]->wq_f(0);
	gsl_blas_zdotc( wq_f, snapShot_f, &val);
	gsl_vector_complex_set(interference_outputs_[srcX], 0, val);

	// calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
	for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
	  snapShot_f = snapshot_array_->snapshot(fbinX);
	  wq_f = bfweight_vec_[srcX]->wq_f(fbinX);
	  gsl_blas_zdotc( wq_f, snapShot_f, &val);
	  if( fbinX < fftLen2 ){
	    gsl_vector_complex_set(interference_outputs_[srcX], fbinX, val);
	    gsl_vector_complex_set(interference_outputs_[srcX], fftLen_ - fbinX, gsl_complex_conjugate(val) );
	  }
	  else
	    gsl_vector_complex_set(interference_outputs_[srcX], fftLen2, val);
	}// for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
      }
    }// for (unsigned srcX = 0; srcX < nSource_; srcX++)
  }

  {// post-filtering
    for (unsigned srcX = 0; srcX < nSource_; srcX++) {
      if( binary_mask_type_ == 0 && srcX == targetSourceX_ )
	continue;

      float alpha;
      gsl_vector_complex** wq  = bfweight_vec_[srcX]->arrayManifold(); // use a D&S beamformer output as a clean signal.
      gsl_vector_complex*  wp1 = bfweight_vec_[0]->wp1();
      gsl_vector_complex** prevCSDs = bfweight_vec_[srcX]->CSDs();

      if(  frame_no_ > 0 )
	alpha =  alpha_;
      else
	alpha = 0.0;

      if( (int)TYPE_APAB & pftype_ ){
	ApabFilter( wq, snapshot_array_, fftLen, chanN(), halfBandShift_, interference_outputs_[srcX], chanN()/2 );
      }
      else if( (int)TYPE_ZELINSKI1_REAL & pftype_ || (int)TYPE_ZELINSKI1_ABS & pftype_ ){

	if( (int)TYPE_ZELINSKI2 & pftype_  )
	  wq =  bfweight_vec_[srcX]->wq(); // just use a beamformer output as a clean signal.

	if( frame_no_ < MINFRAMES )// just update cross spectral densities
	  ZelinskiFilter( wq, snapshot_array_, halfBandShift_, interference_outputs_[srcX], prevCSDs, wp1, alpha, (int)NO_USE_POST_FILTER);
	else
	  ZelinskiFilter( wq, snapshot_array_, halfBandShift_, interference_outputs_[srcX], prevCSDs, wp1, alpha, pftype_ );
      }
    }// for (unsigned srcX = 0; srcX < nSource_; srcX++)
  }

  return;
}

/*
  @brief averaging the output of a beamformer recursively.
          Y'(f,t) = a * Y(f,t) + ( 1 -a ) *  Y'(f,t-1)

*/
static void set_averaged_output_( gsl_vector_complex* avgOutput, unsigned fbinX, gsl_vector_complex* curOutput, float avgFactor )
{
  gsl_complex prev = gsl_complex_mul_real( gsl_vector_complex_get( avgOutput, fbinX ), avgFactor );
  gsl_complex curr = gsl_complex_mul_real( gsl_vector_complex_get( curOutput, fbinX ), ( 1.0 - avgFactor ) );
  gsl_vector_complex_set( avgOutput, fbinX, gsl_complex_add(prev,curr) );
}

/**
   @brief calculate a mean of subband components over frequency bins.

 */
static gsl_complex getMeanOfSubbandC( int fbinX, gsl_vector_complex *output, unsigned fftLen, unsigned fwidth )
{
  if( fwidth <= 1 )
    return( gsl_vector_complex_get( output, fbinX ) );

  int fbinStart, fbinEnd;
  unsigned count = 0;
  gsl_complex sum = gsl_complex_rect( 0.0, 0.0 );

  fbinStart = fbinX - fwidth/2;
  if( fbinStart < 1 ) fbinStart = 1; // a direct component is not used
  fbinEnd = fbinX + fwidth/2;
  if( fbinEnd >= fftLen ) fbinEnd = fftLen - 1;

  for(int i=fbinStart;i<=fbinEnd;i++,count++){
    sum = gsl_complex_add( sum, gsl_vector_complex_get( output, i ) );
  }

  return( gsl_complex_div_real( sum, (double) count ) );
}

/**
   @brief Do binary masking. If an output power of a target source > outputs of interferences, the target signal is set to 0. 

   @note if avgFactor >= 0, a recursively averaged subband compoent is set instead of 0.
   @param gsl_vector_complex** interferenceOutputs[in]
   @param unsinged targetSourceX[in]
   @param gsl_vector_complex* output[in/out]
*/
void SubbandMMI::binary_masking_( gsl_vector_complex** interferenceOutputs, unsigned targetSourceX, gsl_vector_complex* output )
{
  unsigned fftLen = fftLen_;
  gsl_complex tgtY, valY, itfY;
  gsl_complex newVal;
  gsl_vector_complex* targetOutput = interference_outputs_[targetSourceX_];

  if( halfBandShift_ == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      double tgtPow, valPow, maxPow = 0.0;

      tgtY = gsl_vector_complex_get( targetOutput, fbinX );
      tgtPow = gsl_complex_abs2( tgtY );
      for (unsigned srcX = 0; srcX < nSource_; srcX++) {
	if( srcX == targetSourceX_ )
	  continue;
	valY = gsl_vector_complex_get( interferenceOutputs[srcX], fbinX );
	valPow = gsl_complex_abs2( valY );
	if( valPow > maxPow ){
	  maxPow = valPow;
	  itfY = valY;
	}
      }// for (unsigned srcX = 0; srcX < nSource_; srcX++)
      if( avg_factor_ >= 0.0 )
	newVal = gsl_complex_mul_real( getMeanOfSubbandC( (int)fbinX, avg_output_, fftLen_, fwidth_ ), avg_factor_ );
      else
	newVal = gsl_complex_rect( 0.0, 0.0 );
      if( tgtPow < maxPow ){
	gsl_vector_complex_set( output, fbinX, newVal );
	if( avg_factor_ >= 0.0 )
	  gsl_vector_complex_set( avg_output_, fbinX, newVal );
      }
      else{
	if( avg_factor_ >= 0.0 )
	  set_averaged_output_( avg_output_, fbinX, output, avg_factor_ );
      }
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      double tgtPow, valPow, maxPow = 0.0;

      tgtY = gsl_vector_complex_get( targetOutput, fbinX );
      tgtPow = gsl_complex_abs2( tgtY );
      for (unsigned srcX = 0; srcX < nSource_; srcX++) {// seek a subband component with maximum power
	if( srcX == targetSourceX_ )
	  continue;
	valY = gsl_vector_complex_get( interferenceOutputs[srcX], fbinX );
	valPow = gsl_complex_abs2( valY );
	if( valPow > maxPow ){
	  maxPow = valPow;
	  itfY = valY;
	}
      }// for (unsigned srcX = 0; srcX < nSource_; srcX++)
      if( avg_factor_ >= 0.0 ) // set the estimated value (avg_factor_ * avg_output_[t-1])
	newVal = gsl_complex_mul_real( getMeanOfSubbandC( (int)fbinX, avg_output_, fftLen_/2, fwidth_ ), avg_factor_ );
      else // set 0 to the output
	newVal = gsl_complex_rect( 0.0, 0.0 );
      if( tgtPow < maxPow ){
	if( fbinX < fftLen2 ){
	  gsl_vector_complex_set( output, fbinX, newVal );
	  gsl_vector_complex_set( output, fftLen_ - fbinX, newVal );
	}
	else
	  gsl_vector_complex_set( output, fftLen2, newVal );
	if( avg_factor_ >= 0.0 )
	  gsl_vector_complex_set( avg_output_, fbinX, newVal );
      }
      else{
	if( avg_factor_ >= 0.0 )
	  set_averaged_output_( avg_output_, fbinX, output, avg_factor_ );
      }
    }//for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
  }

  return;
}

SubbandMVDR::SubbandMVDR( unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandDS( fftLen, halfBandShift, nm )
{
  if( halfBandShift == true ){
    throw jallocation_error("halfBandShift==true is not yet supported\n");
  }

  R_    = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  invR_ = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  if( R_ == NULL || invR_ == NULL ){
    throw jallocation_error("SubbandMVDR: gsl_matrix_complex_alloc failed\n");
  }

  wmvdr_ = (gsl_vector_complex** )malloc( (fftLen/2+1) * sizeof(gsl_vector_complex*) );
  if( wmvdr_ == NULL ){
    throw jallocation_error("SubbandMVDR: gsl_vector_complex_alloc failed\n");
  }

  diagonal_weights_ = (float *)calloc( (fftLen/2+1), sizeof(float) );
  if( diagonal_weights_ == NULL ){
    throw jallocation_error("SubbandMVDR: cannot allocate RAM\n");
  }

  for( unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++){
    R_[fbinX] = NULL;
    invR_[fbinX]  = NULL;
    wmvdr_[fbinX] = NULL;
  }

}

// ----- definition for class `SubbandMVDR' -----
//

SubbandMVDR::~SubbandMVDR()
{
  unsigned fftLen2 = fftLen_ / 2;

  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=R_[fbinX] )
      gsl_matrix_complex_free( R_[fbinX] );
    if( NULL!=invR_[fbinX] )
      gsl_matrix_complex_free( invR_[fbinX] );
    if( NULL!= wmvdr_[fbinX] )
      gsl_vector_complex_free( wmvdr_[fbinX] );
  }
  free(R_);
  free(invR_);
  free(wmvdr_);
  free(diagonal_weights_);
}

void SubbandMVDR::clear_channel()
{
  unsigned fftLen2 = fftLen_ / 2;

  SubbandDS::clear_channel();

  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=R_[fbinX] ){
      gsl_matrix_complex_free( R_[fbinX] );
      R_[fbinX] = NULL;
    }
    if( NULL!= wmvdr_[fbinX] ){
      gsl_vector_complex_free( wmvdr_[fbinX] );
      wmvdr_[fbinX] = NULL;
    }
  }
}

bool SubbandMVDR::calc_mvdr_weights( float samplerate, float dThreshold, bool calcInverseMatrix )
{
  if( NULL == R_[0] ){
    throw jallocation_error("Set a spatial spectral matrix before calling calc_mvdr_weights()\n");
  }
  if( 0 == bfweight_vec_.size() ){
    throw j_error("call calc_array_manifold_vectorsX() once\n");
  }

  unsigned nChan = chanN();
  gsl_vector_complex *tmpH = gsl_vector_complex_alloc( nChan );
  gsl_complex val1 = gsl_complex_rect( 1.0, 0.0 );
  gsl_complex val0 = gsl_complex_rect( 0.0, 0.0 );
  gsl_complex Lambda;
  bool ret;

  if( NULL == wmvdr_[0] ){
    wmvdr_[0] = gsl_vector_complex_alloc( nChan );
  }
  for( unsigned chanX=0 ; chanX < nChan ;chanX++ ){
    gsl_vector_complex_set( wmvdr_[0], chanX, val1 );
  }
  for(unsigned fbinX=1;fbinX<=fftLen_/2;fbinX++){
    gsl_complex norm;
    const gsl_vector_complex* arrayManifold_f = bfweight_vec_[0]->wq_f(fbinX);

    if( NULL == invR_[fbinX] )
      invR_[fbinX] = gsl_matrix_complex_alloc( nChan, nChan );

    // calculate the inverse matrix of the coherence matrix
    if( true == calcInverseMatrix ){
      ret = pseudoinverse( R_[fbinX], invR_[fbinX], dThreshold );
      if( false==ret )
	gsl_matrix_complex_set_identity( invR_[fbinX] );
    }

    gsl_blas_zgemv( CblasConjTrans, val1, invR_[fbinX], arrayManifold_f, val0, tmpH ); // tmpH = invR^H * d
    gsl_blas_zdotc( tmpH, arrayManifold_f, &Lambda ); // Lambda = d^H * invR * d
    norm = gsl_complex_mul_real( Lambda, nChan );

    if( NULL == wmvdr_[fbinX] ){
      wmvdr_[fbinX] = gsl_vector_complex_alloc( nChan );
    }
    for( unsigned chanX=0 ; chanX < nChan ;chanX++ ){
      gsl_complex val = gsl_vector_complex_get( tmpH, chanX );// val = invR^H * d
      gsl_vector_complex_set( wmvdr_[fbinX], chanX, gsl_complex_div( val, norm /*Lambda*/ ) );
    }
  }

  gsl_vector_complex_free( tmpH );

  return true;
}

/**
   @brief set the spatial spectral matrix for the MVDR beamformer

   @param unsigned fbinX[in]
   @param gsl_matrix_complex* Rnn[in]
 */
bool SubbandMVDR::set_noise_spatial_spectral_matrix(unsigned fbinX, gsl_matrix_complex* Rnn)
{

  if( Rnn->size1 != chanN() ){
    fprintf(stderr,"The number of the rows of the matrix must be %d but it is %lu\n", chanN(), Rnn->size1 );
    return false;
  }
  if( Rnn->size2 != chanN() ){
    fprintf(stderr,"The number of the columns of the matrix must be %d but it is %lu\n", chanN(), Rnn->size2 );
    return false;
  }

  if( R_[fbinX] == NULL ){
    R_[fbinX] = gsl_matrix_complex_alloc( chanN(), chanN() );
  }

  for(unsigned m=0;m<chanN();m++){
    for(unsigned n=0;n<chanN();n++){
      gsl_matrix_complex_set( R_[fbinX], m, n, gsl_matrix_complex_get( Rnn, m, n ) );
    }
  }

  return true;
}

/**
   @brief calculate the coherence matrix in the case of the diffuse noise field.

   @param const gsl_matrix* micPositions[in] geometry of the microphone array. micPositions[no. channels][x,y,z]
   @param float samplerate[in]
   @param float sspeed[in]
 */
bool SubbandMVDR::set_diffuse_noise_model( const gsl_matrix* micPositions, float samplerate, float sspeed )
{
  size_t micN  = micPositions->size1;

  if( micN != chanN() ){
    fprintf(stderr,"The number of microphones must be %d but it is %lu\n", chanN(), micN);
    return false;
  }
  if( micPositions->size2 < 3 ){
    fprintf(stderr,"The microphone positions should be described in the three dimensions\n");
    return false;
  }

  gsl_matrix *dm = gsl_matrix_alloc( micN, micN );

  for(unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++){
    if( R_[fbinX] == NULL ){
      R_[fbinX] = gsl_matrix_complex_alloc( micN, micN );
    }
  }

  {// calculate the distance matrix.
     for(unsigned m=0;m<micN;m++){
       for(unsigned n=0;n<m;n++){ //for(unsigned n=0;n<micN;n++){
	 double Xm = gsl_matrix_get( micPositions, m, 0 );
	 double Xn = gsl_matrix_get( micPositions, n, 0 );
	 double Ym = gsl_matrix_get( micPositions, m, 1 );
	 double Yn = gsl_matrix_get( micPositions, n, 1 );
	 double Zm = gsl_matrix_get( micPositions, m, 2 );
	 double Zn = gsl_matrix_get( micPositions, n, 2 );

	 double dx = Xm - Xn;
	 double dy = Ym - Yn;
	 double dz = Zm - Zn;
	 gsl_matrix_set( dm, m, n, sqrt( dx * dx + dy * dy + dz * dz ) );
       }
     }
     //for(unsigned m=0;m<micN;m++){ gsl_matrix_set( dm, m, m, 0.0 );}
  }

  {
    for(unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++){
      //double omega_d_c = 2.0 * M_PI * samplerate * fbinX / ( fftLen_ * sspeed );
      double omega_d_c = 2.0 * samplerate * fbinX / ( fftLen_ * sspeed );

      for(unsigned m=0;m<micN;m++){
        for(unsigned n=0;n<m;n++){
          double Gamma_mn = gsl_sf_sinc( omega_d_c * gsl_matrix_get( dm, m, n ) );
          gsl_matrix_complex_set( R_[fbinX], m, n, gsl_complex_rect( Gamma_mn, 0.0 ) );
        }// for(unsigned n=0;n<micN;n++){
      }// for(unsigned m=0;m<micN;m++){
      for(unsigned m=0;m<micN;m++){
        gsl_matrix_complex_set( R_[fbinX], m, m, gsl_complex_rect( 1.0, 0.0 ) );
      }
      for(unsigned m=0;m<micN;m++){
        for(unsigned n=m+1;n<micN;n++){
          gsl_complex val = gsl_matrix_complex_get( R_[fbinX], n, m );
          gsl_matrix_complex_set( R_[fbinX], m, n, val );
        }
      }
    }// for(unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++){
  }
  //gsl_sf_sinc (double x);

  gsl_matrix_free(dm);

  return true;
}

void SubbandMVDR::set_all_diagonal_loading(float diagonalWeight)
{
  if( R_ == NULL ){
    throw j_error("Construct first a noise covariance matrix\n");
  }
  for(unsigned fbinX=0;fbinX<=fftLen_/2;fbinX++){
    diagonal_weights_[fbinX] = diagonalWeight;
    for( unsigned chanX=0 ; chanX < chanN() ;chanX++ ){// diagonal loading
      gsl_complex val = gsl_matrix_complex_get( R_[fbinX], chanX, chanX );
      gsl_matrix_complex_set( R_[fbinX], chanX, chanX, gsl_complex_add_real( val, diagonal_weights_[fbinX] ) );
    }
  }
}

void SubbandMVDR::set_diagonal_looading(unsigned fbinX, float diagonalWeight)
{
  if( R_ == NULL ){
    throw j_error("Construct first a noise covariance matrix\n");
  }
  diagonal_weights_[fbinX] = diagonalWeight;
  for( unsigned chanX=0 ; chanX < chanN() ;chanX++ ){// diagonal loading
    gsl_complex val = gsl_matrix_complex_get( R_[fbinX], chanX, chanX );
    gsl_matrix_complex_set( R_[fbinX], chanX, chanX, gsl_complex_add_real( val, diagonal_weights_[fbinX] ) );
  }
}

const gsl_vector_complex* SubbandMVDR::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if( 0 == bfweight_vec_.size() ){
    throw j_error("call calc_array_manifold_vectorsX() once\n");
  }
  if( NULL == wmvdr_[0] ){
    throw j_error("call calc_mvdr_weights() once\n");
  }

  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  gsl_complex val;
  unsigned fftLen = fftLen_;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

   if( halfBandShift_ == true ){
     // TODO : implement
   }
   else{
     unsigned fftLen2 = fftLen/2;

     // calculate a direct component.
     snapShot_f = snapshot_array_->snapshot(0);
     gsl_blas_zdotc( wmvdr_[0], snapShot_f, &val );
     gsl_vector_complex_set(vector_, 0, val);

     // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
     for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
       snapShot_f = snapshot_array_->snapshot(fbinX);
       gsl_blas_zdotc( wmvdr_[fbinX], snapShot_f, &val );
       if( fbinX < fftLen2 ){
	 gsl_vector_complex_set(vector_, fbinX, val);
	 gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
       }
       else
         gsl_vector_complex_set(vector_, fftLen2, val);
     }
   }

   increment_();
   return vector_;
}

void SubbandMVDR::divide_nondiagonal_elements(unsigned fbinX, float mu)
{
  for ( size_t chanX=0; chanX<chanN(); chanX++ ){
    for ( size_t chanY=0; chanY<chanN(); chanY++ ){
      if( chanX != chanY ){
        gsl_complex Rxy = gsl_matrix_complex_get( R_[fbinX], chanX, chanY );
        gsl_matrix_complex_set( R_[fbinX], chanX, chanY, gsl_complex_div( Rxy, gsl_complex_rect( (1.0+mu), 0.0 ) ) );
      }
    }
  }
}

// ----- definition for class `SubbandMVDRGSC' -----
//

SubbandMVDRGSC::SubbandMVDRGSC( unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandMVDR( fftLen, halfBandShift, nm ), normalize_weight_(false)
{
}

SubbandMVDRGSC::~SubbandMVDRGSC()
{
}

void SubbandMVDRGSC::set_active_weights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("set the quiescent vector once\n");
  }
  bfweight_vec_[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

void SubbandMVDRGSC::zero_active_weights()
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_gsc_weights_x() once\n");
  }

  gsl_vector_complex *wa = gsl_vector_complex_calloc( chanN() - bfweight_vec_[0]->NC() );
  for (unsigned fbinX = 0; fbinX < fftLen_; fbinX++){
   bfweight_vec_[0]->calcSidelobeCancellerU_f( fbinX, wa );
  }
  gsl_vector_complex_free( wa );
}

/**
   @brief compute the blocking matrix so as to satisfy the orthogonal condition 
          with the delay-and-sum beamformer's weight.
 */
bool SubbandMVDRGSC::calc_blocking_matrix1( float samplerate, const gsl_vector* delaysT )
{
  this->alloc_bfweight_( 1, 1 );
  bfweight_vec_[0]->calcMainlobe( samplerate, delaysT, true );
  return true;
}

/**
   @brief compute the blocking matrix so as to satisfy the orthogonal condition
          with the MVDR beamformer's weight.
 */
bool SubbandMVDRGSC::calc_blocking_matrix2()
{
  if( NULL == wmvdr_[0] ){
    return false;
  }

  this->alloc_bfweight_( 1, 1 );

  if( halfBandShift_ == true ){
     // TODO : implement
    return false;
  }
  else{
    unsigned fftLen2 = fftLen_/2;

    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      gsl_vector_complex* destWq = bfweight_vec_[0]->wq_f(fbinX);
      gsl_vector_complex_memcpy( destWq, wmvdr_[fbinX] );
      bfweight_vec_[0]->calcBlockingMatrix( fbinX );
    }
  }

  return true;
}

void SubbandMVDRGSC::upgrade_blocking_matrix()
{
  gsl_matrix_complex** B = bfweight_vec_[0]->B();
  gsl_vector_complex *weight = gsl_vector_complex_alloc( this->chanN() );

  /* printf("SubbandMVDRGSC: set the orthogonal matrix of the entire vector to the blocking matrix\n");*/

  for (unsigned fbinX = 1; fbinX < fftLen_; fbinX++) {
    gsl_vector_complex_memcpy( weight, bfweight_vec_[0]->wq_f(fbinX) );
    gsl_vector_complex_sub(    weight, bfweight_vec_[0]->wl_f(fbinX) );

    if( false==calc_blocking_matrix_( weight, bfweight_vec_[0]->NC(), B[fbinX] ) ){
      throw j_error("calc_blocking_matrix_() failed\n");
    }
  }

  gsl_vector_complex_free( weight );
}

const gsl_vector_complex* SubbandMVDRGSC::blocking_matrix_output( int outChanX )
{
  const gsl_vector_complex* snapShot_f;
   
  if( halfBandShift_ == true ){
    // TODO : implement
  }
  else{
    const unsigned fftLen2 = fftLen_/2;
    gsl_matrix_complex** B = bfweight_vec_[0]->B();
    gsl_vector_complex* bi = gsl_vector_complex_alloc( this->chanN() );
    gsl_complex val;
       
    for (unsigned fbinX = 0; fbinX <= fftLen2; fbinX++) {
      gsl_matrix_complex_get_col( bi, B[fbinX], outChanX );
      snapShot_f = snapshot_array_->snapshot(fbinX);
      gsl_blas_zdotc( bi, snapShot_f, &val);
      gsl_vector_complex_set(vector_, fbinX, val );
    }

    gsl_vector_complex_free( bi );
  }

  return vector_;
}

const gsl_vector_complex* SubbandMVDRGSC::next(int frame_no)
{

  if (frame_no == frame_no_) return vector_;

  if( 0 == bfweight_vec_.size() ){
    throw j_error("call calc_array_manifold_vectorsX() once\n");
  }
  if( NULL == wmvdr_[0] ){
    throw j_error("call calc_mvdr_weights() once\n");
  }

  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_complex val;
  unsigned fftLen = fftLen_;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

   if( halfBandShift_ == true ){
     // TODO : implement
   }
   else{
     unsigned fftLen2 = fftLen/2;

     // calculate a direct component.
     snapShot_f = snapshot_array_->snapshot(0);
     gsl_blas_zdotc( wmvdr_[0], snapShot_f, &val );
     gsl_vector_complex_set(vector_, 0, val);

     // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
     for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
       snapShot_f = snapshot_array_->snapshot(fbinX);
       wl_f = bfweight_vec_[0]->wl_f(fbinX);
      
       calc_gsc_output( snapShot_f, wl_f, wmvdr_[fbinX], &val, normalize_weight_ );
       if( fbinX < fftLen2 ){
	 gsl_vector_complex_set(vector_, fbinX, val);
	 gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
       }
       else
	 gsl_vector_complex_set(vector_, fftLen2, val);
     }
   }

   increment_();
   return vector_;
}

// ----- members for class `SubbandOrthogonalizer' -----
//
SubbandOrthogonalizer::SubbandOrthogonalizer(SubbandMVDRGSCPtr &beamformer, int outChanX,  const String& nm)
  : VectorComplexFeatureStream(beamformer->fftLen(), nm),
    beamformer_(beamformer),
    outChanX_(outChanX)
{
}

SubbandOrthogonalizer::~SubbandOrthogonalizer()
{
}

const gsl_vector_complex* SubbandOrthogonalizer::next(int frame_no)
{
  const gsl_vector_complex* vector;
  if (frame_no == frame_no_) return vector_;

  if( outChanX_ <= 0 ){
    vector = beamformer_->next(frame_no);
  }
  else{
    vector = beamformer_->blocking_matrix_output( outChanX_ - 1 );
  }

  gsl_vector_complex_memcpy(vector_, vector );

  increment_();

  return vector_;
}


const gsl_vector_complex* SubbandBlockingMatrix::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = fftLen_;

  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call calc_gsc_weights_x() once\n");
  }

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);
      
      calc_gsc_output( snapShot_f, wl_f,  wq_f, &val, normalize_weight_ );
      gsl_vector_complex_set(vector_, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = snapshot_array_->snapshot(0);
    wq_f       = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(vector_, 0, val);
    //wq_f = _bfWeights->wq_f(0);
    //wl_f = _bfWeights->wl_f(0);
    //calc_gsc_output( snapShot_f, chanN(), wl_f, wq_f, &val );
    //gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);
      
      calc_gsc_output( snapShot_f, wl_f, wq_f, &val, normalize_weight_ );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(vector_, fbinX,           val);
	gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(vector_, fftLen2, val);
    }
  }

  increment_();
  
  return vector_;
}


// ----- definition for class DOAEstimatorSRPBase' -----
// 
DOAEstimatorSRPBase::DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax ):
  widthTheta_(0.25),
  widthPhi_(0.25),
  minTheta_(-M_PI),
  maxTheta_(M_PI),
  minPhi_(-M_PI),
  maxPhi_(M_PI),
  fbinMin_(1),
  fbinMax_(fbinMax),
  nBest_(nBest),
  table_initialized_(false),
  accRPs_(NULL),
  rpMat_(NULL),
  engery_threshold_(0.0)
{
  nBestRPs_   = gsl_vector_calloc( nBest_ );
  argMaxDOAs_ = gsl_matrix_calloc( nBest_, 2 );
}

DOAEstimatorSRPBase::~DOAEstimatorSRPBase()
{
  if( NULL != nBestRPs_ )
    gsl_vector_free( nBestRPs_ );
  if( NULL != argMaxDOAs_ )
    gsl_matrix_free( argMaxDOAs_ );

  clear_table_();
}

#ifdef __MBDEBUG__
void DOAEstimatorSRPBase::allocDebugWorkSapce()
{
  float nTheta = ( maxTheta_ - minTheta_ ) / widthTheta_ + 0.5 + 1;
  float nPhi   = ( maxPhi_ - minPhi_ ) / widthPhi_  + 0.5 + 1;
  rpMat_ = gsl_matrix_calloc( (int)nTheta, (int)nPhi );
}
#endif /* #ifdef __MBDEBUG__ */

void DOAEstimatorSRPBase::clear_table_()
{
  //fprintf(stderr,"DOAEstimatorSRPBase::clear_table_()\n");
  if( true == table_initialized_ ){
    for(unsigned i=0;i<svTbl_.size();i++){
      for(unsigned fbinX=0;fbinX<=fbinMax_;fbinX++)
	gsl_vector_complex_free( svTbl_[i][fbinX] );
      free( svTbl_[i] );
    }
    svTbl_.clear();
#ifdef __MBDEBUG__
    if( NULL != rpMat_ ){
      gsl_matrix_free( rpMat_ );
      rpMat_ = NULL;
    }
#endif /* #ifdef __MBDEBUG__ */
    if( NULL != accRPs_ ){
      gsl_vector_free( accRPs_ );
      accRPs_ = NULL;
    }
  }
  table_initialized_ = false;
  //fprintf(stderr,"DOAEstimatorSRPBase::clear_table_()2\n");
}


void DOAEstimatorSRPBase::get_nbest_hypotheses_from_accrp_()
{
  for(unsigned n=0;n<nBest_;n++){
    gsl_vector_set( nBestRPs_, n, -10e10 );
    gsl_matrix_set( argMaxDOAs_, n, 0, -M_PI);
    gsl_matrix_set( argMaxDOAs_, n, 1, -M_PI);
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=minPhi_;phiIdx<nPhi_;phi+=widthPhi_,phiIdx++){
      double rp = gsl_vector_get( accRPs_, unitX++ );
#ifdef __MBDEBUG__
      gsl_matrix_set( rpMat_, thetaIdx, phiIdx, rp );
#endif /* #ifdef __MBDEBUG__ */
      if( rp > gsl_vector_get( nBestRPs_, nBest_-1 ) ){
	//  decide the order of the candidates
	for(unsigned n1=0;n1<nBest_;n1++){
	  if( rp > gsl_vector_get( nBestRPs_, n1 ) ){
	    // shift the other candidates
	    for(unsigned n2=nBest_-1;n2>n1;n2--){
	      gsl_vector_set( nBestRPs_,   n2, gsl_vector_get( nBestRPs_, n2-1 ) );
	      gsl_matrix_set( argMaxDOAs_, n2, 0, gsl_matrix_get( argMaxDOAs_, n2-1, 0 ) );
	      gsl_matrix_set( argMaxDOAs_, n2, 1, gsl_matrix_get( argMaxDOAs_, n2-1, 1 ) );
	    }
	    // keep this as the n1-th best candidate
	    gsl_vector_set( nBestRPs_, n1, rp );
	    gsl_matrix_set( argMaxDOAs_, n1, 0, theta);
	    gsl_matrix_set( argMaxDOAs_, n1, 1, phi);
	    break;
	  }
	}
	// for(unsinged n1=0;n1<nBest_-1;n1++)
      }
    }
  }

}

void DOAEstimatorSRPBase::init_accs_()
{
  if( NULL != accRPs_ )
    gsl_vector_set_zero(accRPs_);
#ifdef __MBDEBUG__
  if( NULL != rpMat_ )
    gsl_matrix_set_zero( rpMat_ );
#endif /* #ifdef __MBDEBUG__ */

  for(unsigned n=0;n<nBest_;n++){
    gsl_vector_set(nBestRPs_, n, -10e10 );
    gsl_matrix_set(argMaxDOAs_, n, 0, -M_PI);
    gsl_matrix_set(argMaxDOAs_, n, 1, -M_PI);
  }
}

void DOAEstimatorSRPBase::set_search_param(float minTheta, float maxTheta, float minPhi, float maxPhi,
                                           float widthTheta, float widthPhi)
{
  if( minTheta > maxTheta )
    throw jparameter_error("Invalid argument: minTheta %f > maxTheta %f\n", minTheta, maxTheta);

  if( minPhi > maxPhi )
    throw jparameter_error("Invalid argument: minPhi %f > maxPhi %f\n", minPhi, maxPhi);

  minTheta_ = minTheta;
  maxTheta_ = maxTheta;
  minPhi_   = minPhi;
  maxPhi_   = maxPhi;
  widthTheta_ = widthTheta;
  widthPhi_   = widthPhi;
  clear_table_();
}

// ----- definition for class DOAEstimatorSRPDSBLA' -----
//
DOAEstimatorSRPDSBLA::DOAEstimatorSRPDSBLA( unsigned nBest, unsigned samplerate, unsigned fftLen, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  SubbandDS(fftLen, false, nm ),
  samplerate_(samplerate)
{
  //fprintf(stderr,"DOAEstimatorSRPDSBLA\n");
  arraygeometry_ = NULL;
  set_search_param();
}

DOAEstimatorSRPDSBLA::~DOAEstimatorSRPDSBLA()
{
  if( NULL != arraygeometry_ )
    gsl_matrix_free( arraygeometry_ );
}

void DOAEstimatorSRPDSBLA::set_array_geometry( gsl_vector *positions )
{
  if( NULL != arraygeometry_ )
    gsl_matrix_free( arraygeometry_ );

  arraygeometry_ = gsl_matrix_alloc( positions->size, 3 );
  for(unsigned i=0;i<positions->size;i++){
    gsl_matrix_set( arraygeometry_, i, 0, gsl_vector_get( positions, i ) );
  }
}

void DOAEstimatorSRPDSBLA::calc_steering_unit_table_()
{
  int nChan = (int)chanN();
  if( nChan == 0 )
    throw jparameter_error("DOAEstimatorSRPDSBLA:calc_steering_unit_table_():: Set the channel\n");

  nTheta_ = (unsigned)( ( maxTheta_ - minTheta_ ) / widthTheta_ + 0.5 );
  nPhi_   = 1;
  int maxUnit  = nTheta_ * nPhi_;
  svTbl_.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    svTbl_[i] = (gsl_vector_complex **)malloc((fbinMax_+1)*sizeof(gsl_vector_complex *));
    if( NULL == svTbl_[i] )
      throw jallocation_error("DOAEstimatorSRPDSBLA:calc_steering_unit_table_():: could not allocate image : %d\n", maxUnit );
    for(unsigned fbinX=0;fbinX<=fbinMax_;fbinX++)
      svTbl_[i][fbinX] = gsl_vector_complex_calloc(nChan);
  }

  if( NULL != accRPs_ )
    gsl_vector_free(accRPs_);
  accRPs_ = gsl_vector_calloc(maxUnit);

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    gsl_vector_complex *weights;

    set_look_direction_( nChan, theta );
    weights = svTbl_[unitX][0];
    for(unsigned chanX=0;chanX<nChan;chanX++)
      gsl_vector_complex_set( weights, chanX, gsl_complex_rect(1,0) );
    for(unsigned fbinX=fbinMin_;fbinX<=fbinMax_;fbinX++){
      weights = svTbl_[unitX][fbinX];
      gsl_vector_complex_memcpy( weights, bfweight_vec_[0]->wq_f(fbinX)) ;
    }
    unitX++;
  }
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  table_initialized_ = true;
}

float DOAEstimatorSRPDSBLA::calc_response_power_(unsigned unitX)
{
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  const gsl_vector_complex* snapShot_f;
  gsl_complex val;
  double rp  = 0.0;

  if( halfBandShift_ == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = fbinMin_; fbinX <= fbinMax_; fbinX++) {
      snapShot_f = snapshot_array_->snapshot(fbinX);
      weights    = svTbl_[unitX][fbinX];
      gsl_blas_zdotc( weights, snapShot_f, &val);

      if( fbinX < fftLen2_ ){
        gsl_vector_complex_set(vector_, fbinX, val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
        rp += 2 * gsl_complex_abs2( val );
      }
      else{
        gsl_vector_complex_set(vector_, fftLen2_, val);
        rp += gsl_complex_abs2( val );
      }
    }
  }
  else{
    throw  j_error("halfBandShift_ == true is not implemented yet\n");
  }

  return rp / ( fbinMax_ - fbinMin_ + 1.0 ); // ( X0^2 + X1^2 + ... + XN^2 )
}

const gsl_vector_complex* DOAEstimatorSRPDSBLA::next( int frame_no )
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<nBest_;n++){
    gsl_vector_set( nBestRPs_, n, -10e10 );
    gsl_matrix_set( argMaxDOAs_, n, 0, -M_PI);
    gsl_matrix_set( argMaxDOAs_, n, 1, -M_PI);
  }

  this->alloc_image_();
#define __MBDEBUG__
  if( false == table_initialized_ )
    calc_steering_unit_table_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  energy_ = calc_energy( snapshot_array_, fbinMin_, fbinMax_, fftLen2_, halfBandShift_ );
  if( energy_ < engery_threshold_ ){
#ifdef __MBDEBUG__
    fprintf(stderr,"Energy %e is less than threshold\n", energy_);
#endif /* #ifdef __MBDEBUG__ */
    increment_();
    return vector_;
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
      //set_look_direction_( theta, phi );
    rp = calc_response_power_( unitX );
    gsl_vector_set( accRPs_, unitX, gsl_vector_get( accRPs_, unitX ) + rp );
    unitX++;
#ifdef __MBDEBUG__
    gsl_matrix_set( rpMat_, thetaIdx, 0, rp);
#endif /* #ifdef __MBDEBUG__ */
    //fprintf( stderr, "t=%0.8f rp=%e\n" , theta, rp );
    if( rp > gsl_vector_get( nBestRPs_, nBest_-1 ) ){
      //  decide the order of the candidates
      for(unsigned n1=0;n1<nBest_;n1++){
	if( rp > gsl_vector_get( nBestRPs_, n1 ) ){
	  // shift the other candidates
	  for(unsigned n2=nBest_-1;n2>n1;n2--){
	    gsl_vector_set( nBestRPs_,   n2, gsl_vector_get( nBestRPs_, n2-1 ) );
	    gsl_matrix_set( argMaxDOAs_, n2, 0, gsl_matrix_get( argMaxDOAs_, n2-1, 0 ) );
	    gsl_matrix_set( argMaxDOAs_, n2, 1, gsl_matrix_get( argMaxDOAs_, n2-1, 1 ) );
	  }
	  // keep this as the n1-th best candidate
	  gsl_vector_set( nBestRPs_, n1, rp );
	  gsl_matrix_set( argMaxDOAs_, n1, 0, theta);
	  gsl_matrix_set( argMaxDOAs_, n1, 1, 0);
	  break;
	}
      }
      // for(unsinged n1=0;n1<nBest_-1;n1++)
    }
  }

  increment_();
  return vector_;
}

void DOAEstimatorSRPDSBLA::set_look_direction_( int nChan, float theta )
{
  gsl_vector* delays = gsl_vector_alloc(nChan);
  double refPosition = gsl_matrix_get(arraygeometry_, 0, 0);

  gsl_vector_set(delays, 0, 0);
  for(int chanX=1;chanX<nChan;chanX++){
    double dist = gsl_matrix_get(arraygeometry_, chanX, 0) - refPosition;
    if( dist < 0 ){ dist = -dist; }

    gsl_vector_set(delays, chanX, dist * cos(theta));
  }
  calc_array_manifold_vectors(samplerate_, delays);
  gsl_vector_free(delays);
}

void DOAEstimatorSRPDSBLA::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

float calc_energy(SnapShotArrayPtr snapShotArray, unsigned fbinMin, unsigned fbinMax, unsigned fftLen2, bool  halfBandShift)
{
  float rp = 0.0;
  unsigned chanN;
  gsl_complex val;

  if( halfBandShift == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = fbinMin; fbinX <= fbinMax; fbinX++) {
      const gsl_vector_complex* F = snapShotArray->snapshot(fbinX);
      chanN = F->size;

      gsl_blas_zdotc( F, F, &val ); // x^H y

      if( fbinX < fftLen2 ){
	rp += 2 * gsl_complex_abs2( val );
      }
      else{
	rp += gsl_complex_abs2( val );
      }
    }
  }
  else{
    throw  j_error("halfBandShift_ == true is not implemented yet\n");
  }

  //fprintf(stderr,"Engery %e\n", rp / ( 2* fftLen2 * chanN ) );

  return rp / ( 2* fftLen2 * chanN );
}
