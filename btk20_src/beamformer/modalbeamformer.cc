/*
 * @file modalbeamformer.cc
 * @brief Beamforming in the spherical harmonics domain.
 * @author Kenichi Kumatani
 */
#include "beamformer/modalbeamformer.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_blas.h>

static void calc_time_delays_of_spherical_array_(float theta, float phi, unsigned chanN, float rad, gsl_vector *theta_s, gsl_vector *phi_s, gsl_vector* delays)
{
  double dist, theta_sn, phi_sn;

  for(unsigned chanX=0;chanX<chanN;chanX++){
    theta_sn = gsl_vector_get( theta_s, chanX );
    phi_sn   = gsl_vector_get( phi_s,   chanX );

    dist = rad * ( sin(theta_sn) * sin(theta) * cos(phi_sn-phi) + cos(theta_sn) * cos(theta) );
    gsl_vector_set( delays, chanX, - dist / SSPEED );
  }
}

static void normalize_weights_( gsl_vector_complex *weights, float wgain )
{
  double nrm = wgain / gsl_blas_dznrm2( weights );// / gsl_blas_dznrm2() returns the Euclidean norm

  for(unsigned i=0;i<weights->size;i++)
    gsl_vector_complex_set( weights, i, gsl_complex_mul_real( gsl_vector_complex_get( weights, i ), nrm ) );
}

/**
	@brief compute the mode amplitude.
	@param int order[in]
	@param double ka[in] k : wavenumber, a : radius of the rigid sphere
	@param double kr[in] k : wavenumber, r : distance of the observation point from the origin
 */
gsl_complex modeAmplitude(int order, double ka)
{
  if( ka == 0 )
    return gsl_complex_rect( 1, 0 );

  gsl_complex bn;
  switch (order){
  case 0:
    {
      double ka2 = ka  * ka;
      double      j0 = gsl_sf_sinc( ka/M_PI );
      double      y0 = - cos(ka)/ka;
      gsl_complex h0 = gsl_complex_rect( j0, y0 );
      double      val1 = cos(ka)/ka - sin(ka)/ka2;
      gsl_complex eika = gsl_complex_polar(1,ka);
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_mul( gsl_complex_rect(ka,1), eika ), ka2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );
      bn = gsl_complex_sub( gsl_complex_rect(j0, 0), gsl_complex_mul( grad, h0 ) );
    }
    break;
  case 1:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka;
      double j1 =  ( sin(ka)/ka2 ) - ( cos(ka)/ka ) ;
      double y1 = -( cos(ka)/ka2 ) - ( sin(ka)/ka ) ;
      gsl_complex h1 = gsl_complex_rect( j1, y1 );
      double val1 = ( - 0.5/ka ) * ( -cos(ka)/ka + sin(ka)/ka2 ) + 0.5 * ( 3*cos(ka)/ka2 + sin(ka)/ka - (3-ka2)*sin(ka)/ka3 );
      double j0 = gsl_sf_sinc( ka/M_PI );
      double y0 = -cos(ka)/ka;
      gsl_complex h0 = gsl_complex_rect( j0, y0 );
      double j2 =  ( 3/ka3 - 1/ka ) * sin(ka) - ( 3/ka2 ) * cos(ka);
      double y2 = -( 3/ka3 - 1/ka ) * cos(ka) - ( 3/ka2 ) * sin(ka);
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      gsl_complex hdiff = gsl_complex_sub( h0, h2 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h1, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );

      bn = gsl_complex_sub( gsl_complex_rect(j1, 0), gsl_complex_mul( grad, h1 ) );
    }
    break;
  case 2:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka;
      double ka4 = ka3 * ka;
      double j2 =  ( 3/ka3 - 1/ka )*sin(ka) - ( 3*cos(ka)/ka2 );
      double y2 = -( 3/ka3 - 1/ka )*cos(ka) - ( 3*sin(ka)/ka2 );
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      double val1 = 0.5 * ( -cos(ka)/ka + sin(ka)/ka2 + (18-ka2)*cos(ka)/ka3 + (-18+7*ka2)*sin(ka)/ka4 );
      double j1 =  ( sin(ka)/ka2 ) - ( cos(ka)/ka ) ;
      double y1 = -( cos(ka)/ka2 ) - ( sin(ka)/ka ) ;
      gsl_complex h1 = gsl_complex_rect( j1, y1 );
      double j3 = ( -15+ka2 )*cos(ka)/ka3 - ( -15+6*ka2 )*sin(ka)/ka4;
      double y3 = ( -15+ka2 )*sin(ka)/ka3 + ( -15+6*ka2 )*cos(ka)/ka4;
      gsl_complex h3 = gsl_complex_rect( j3, y3 );
      gsl_complex hdiff = gsl_complex_sub( h1, h3 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h2, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );

      bn = gsl_complex_sub( gsl_complex_rect(j2, 0), gsl_complex_mul( grad, h2 ) );
    }
    break;
  case 3:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka;
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double j3 = ( -15+ka2 )*cos(ka)/ka3 - ( -15+6*ka2 )*sin(ka)/ka4;
      double y3 = ( -15+ka2 )*sin(ka)/ka3 + ( -15+6*ka2 )*cos(ka)/ka4;
      gsl_complex h3 = gsl_complex_rect( j3, y3 );
      double val1 = 0.5 * ( -3*cos(ka)/ka2 + (3-ka2)*sin(ka)/ka3 + (120-11*ka2)*cos(ka)/ka4 + (-120+51*ka2-ka4)*sin(ka)/ka5 );
      double j2 =  ( 3/ka3 - 1/ka )*sin(ka) - ( 3*cos(ka)/ka2 );
      double y2 = -( 3/ka3 - 1/ka )*cos(ka) - ( 3*sin(ka)/ka2 );
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      double j4 = ( -105+10*ka2 )*cos(ka)/ka4 + (105-45*ka2+ka4)*sin(ka)/ka5 ;
      double y4 = ( -105+10*ka2 )*sin(ka)/ka4 - (105-45*ka2+ka4)*cos(ka)/ka5;
      gsl_complex h4 = gsl_complex_rect( j4, y4 );
      gsl_complex hdiff = gsl_complex_sub( h2, h4 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h3, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );

      bn = gsl_complex_sub( gsl_complex_rect(j3, 0), gsl_complex_mul( grad, h3 ) );
    }
    break;
  default:
    {
      int status;
      gsl_sf_result jn, jn_p, jn_n;
      gsl_sf_result yn, yn_p, yn_n;
      gsl_complex   hn, hn_p, hn_n;
      double djn, dyn;
      gsl_complex dhn;
      gsl_complex grad;
      gsl_complex hdiff;

      status = gsl_sf_bessel_jl_e( order, ka, &jn);// the (regular) spherical Bessel function of the first kind
      status = gsl_sf_bessel_yl_e( order, ka, &yn);// the (irregular) spherical Bessel function of the second kind
      hn = gsl_complex_rect(jn.val, yn.val); // Spherical Hankel function of the first kind

      status = gsl_sf_bessel_jl_e( order-1, ka, &jn_p );
      status = gsl_sf_bessel_jl_e( order+1, ka, &jn_n );
      djn = ( jn_p.val - jn.val / ka - jn_n.val ) / 2;

      status = gsl_sf_bessel_yl_e( order-1, ka, &yn_p );
      status = gsl_sf_bessel_yl_e( order+1, ka, &yn_n );
      dyn = ( yn_p.val - yn.val / ka - yn_n.val ) / 2;

      hn_p = gsl_complex_rect( jn_p.val, yn_p.val );
      hn_n = gsl_complex_rect( jn_n.val, yn_n.val );

      hdiff = gsl_complex_sub( hn_p, hn_n );
      dhn = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( hn, ka) ), 2 );

      //printf ("status  = %s\n", gsl_strerror(status));
      //printf ("J0(5.0) = %.18f +/- % .18f\n", result.val, result.err);

      grad = gsl_complex_div( gsl_complex_rect( djn, 0 ), dhn );
      bn   = gsl_complex_add_real( gsl_complex_negative( gsl_complex_mul( grad, hn ) ), jn.val );
    }
    break;
  }

  return bn;
}

/**
   @brief compute the spherical harmonics transformation of the input shapshot
   @param int maxOrder[in] 
   @param const gsl_vector_complex *XVec[in] the input snapshot vector
   @param gsl_vector_complex **sh_s[in] the orthogonal bases (spherical harmonics) sh_s[sensors index][basis index]
   @param gsl_vector_complex *F[out] the spherical transformation coefficinets will be stored.
 */
static void spherical_harmonics_transformation_( int maxOrder, const gsl_vector_complex *XVec, gsl_vector_complex **sh_s,
					      gsl_vector_complex *F )
{
  gsl_complex Fmn;

  for(int n=0,idx=0;n<maxOrder;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      gsl_blas_zdotu( XVec, sh_s[idx], &Fmn );
      gsl_vector_complex_set( F, idx, Fmn );
      idx++;
    }
  }

  return;
}

/**
   @brief calculate the spherical harmonic for a steering vector
   @param int degree[in]
   @param int order[in]
   @param double theta [in] the look direction
   @param double phi[in] the look direction
   @return the spherical harmonic
 */
static gsl_complex sphericalHarmonic( int degree, int order, double theta, double phi )
{
  int status;
  gsl_sf_result sphPnm;
  gsl_complex Ymn;

  if( order < degree || order < -degree ){
    fprintf( stderr, "The oder must be less than the degree but %d > %dn", order, degree );
  }

  //fprintf(stderr,"%d %d %e ", order, degree, cos(theta));
  if( degree >= 0 ){
    // \sqrt{(2l+1)/(4\pi)} \sqrt{(l-m)!/(l+m)!} P_l^m(x), and derivatives, m >= 0, l >= m, |x| <= 1
    status = gsl_sf_legendre_sphPlm_e( order /* =l */, degree /* =m */,cos(theta), &sphPnm);
  }
  else{
    status = gsl_sf_legendre_sphPlm_e( order /* =l */, -  degree /* =m */,cos(theta), &sphPnm);
    if( ( (-degree) % 2 ) != 0 )
      sphPnm.val = - sphPnm.val;
  }

  Ymn = gsl_complex_mul_real( gsl_complex_polar( 1.0, degree*phi ), sphPnm.val );
  //fprintf(stderr,"%e %e \n", sphPnm.val, sphPnm.err);

  return Ymn;
}

static void calcDCWeights( unsigned maxOrder, gsl_vector_complex *weights )
{
  for(int n=0,idx=0;n<maxOrder;n++){
     for( int m=-n;m<=n;m++){/* degree */
       if( n == 0 ){
         gsl_vector_complex_set( weights, idx, gsl_complex_rect(1,0) );
       }
       else{
         gsl_vector_complex_set( weights, idx, gsl_complex_rect(0,0) );
       }
       idx++;
     }
  }
}

// ----- definition for class `ModeAmplitudeCalculator' -----
//
gsl_complex modeAmplitude(int order, double ka);

ModeAmplitudeCalculator::ModeAmplitudeCalculator(int order, float minKa, float maxKa, float wid)
  : mode_amplitude_(NULL), minKa_(minKa), maxKa_(maxKa), wid_(wid)
{
  float sizef = ( maxKa_ - minKa_ ) / wid_;
  int size = (int)(sizef + 0.5);
  unsigned idx = 0;
  mode_amplitude_ = gsl_vector_complex_alloc( size );

  for(float ka=minKa;idx<size;ka+=wid,idx++){
    gsl_complex val = modeAmplitude( order, ka );
    gsl_vector_complex_set( mode_amplitude_, idx, val );
  }
}


ModeAmplitudeCalculator:: ~ModeAmplitudeCalculator()
{
  if(mode_amplitude_!=NULL)
    gsl_vector_complex_free(mode_amplitude_);
}

// ----- definition for class `EigenBeamformer' -----
// 
EigenBeamformer::EigenBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :  SubbandDS( fftLen, halfBandShift, nm ),
     samplerate_(sampleRate),
     NC_(NC),
     maxOrder_(maxOrder),
     weights_normalized_(normalizeWeight),
     mode_mplitudes_(NULL),
     F_(NULL),
     sh_s_(NULL),
     st_snapshot_array_(NULL),
     theta_(0.0),
     phi_(0.0),
     a_(0.0),
     theta_s_(NULL),
     phi_s_(NULL),
     beampattern_(NULL),
     WNG_(NULL),
     wgain_(1.0),
     sigma2_(0.0)
{
  bfweight_vec_.resize(1); // the steering vector
  bfweight_vec_[0] = NULL;

  dim_ = maxOrder_ * maxOrder_;
  //dim_ = 0;
  //for(int n=0;n<maxOrder_;n++)
  //for( int m=-n;m<=n;m++)
  //dim_++;
}
  
EigenBeamformer::~EigenBeamformer()
{
  if( NULL != mode_mplitudes_ )
    gsl_matrix_complex_free( mode_mplitudes_ );

  if( NULL != F_ ){
    gsl_vector_complex_free( F_ );
    F_ = NULL;
  }

  if( NULL != sh_s_ ){
    for(unsigned dimX=0;dimX<dim_;dimX++)
      gsl_vector_complex_free( sh_s_[dimX] );
    free(sh_s_);
    sh_s_ = NULL;
  }

  if( NULL != theta_s_ )
    gsl_vector_free( theta_s_ );
  if( NULL != phi_s_ )
    gsl_vector_free( phi_s_ );
  if( NULL != beampattern_ )
    gsl_matrix_free( beampattern_ );
  if( NULL != WNG_ )
    gsl_vector_free(WNG_ );
}

/**
   @brief compute the output of the eigenbeamformer
   @param unsigned fbinX[in]
   @param gsl_matrix_complex *bMat[in] mode amplitudes 
   @param double theta[in] the steering direction
   @param double phi[in] the steering direction
   @param gsl_vector_complex *weights[out]
*/
void EigenBeamformer::calc_weights_( unsigned fbinX, gsl_vector_complex *weights )
{
  unsigned norm = dim_ * (unsigned)phi_s_->size;

  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn ) + sigma2_;
    double      de  = norm * bn2;
    gsl_complex in;
    gsl_complex inbn;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }

    inbn = gsl_complex_mul( in, bn );
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
      gsl_complex YmnA = gsl_complex_conjugate(sphericalHarmonic( m, n, theta_, phi_ ));

      //weight = gsl_complex_div( sphericalHarmonic( m, n, theta_, phi_ ), gsl_complex_mul_real( gsl_complex_mul( in, bn ), 4 * M_PI ) ); // follow Rafaely's paper
      //gsl_vector_complex_set( weights, idx, gsl_complex_conjugate(weight) ); 
      weight = gsl_complex_div_real( gsl_complex_mul( gsl_complex_mul_real( YmnA, 4 * M_PI ), inbn ), de ); // HMDI beamfomrer; see S Yan's paper
      gsl_vector_complex_set( weights, idx, weight ); 
      idx++;
    }
  }
  
  if( true==weights_normalized_ )
    normalize_weights_( weights, wgain_ );

  return;
}

const gsl_vector_complex* EigenBeamformer::next( int frame_no )
{
  //fprintf(stderr, "EigenBeamformer::next" );

  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == false ){
    // calculate a direct component.
    XVec    = snapshot_array_->snapshot(0);
    spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
    st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, 0 );
    weights = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( weights, F_, &val ); //gsl_blas_zdotc( weights, F_, &val );
    gsl_vector_complex_set(vector_, 0, val);
    //gsl_vector_complex_set(vector_, 0, gsl_vector_complex_get( XVec, XVec->size/2) );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++) {
      XVec  = snapshot_array_->snapshot(fbinX);
      spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
      st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, fbinX );
      weights =  bfweight_vec_[0]->wq_f(fbinX);
      gsl_blas_zdotc( weights, F_, &val ); // gsl_blas_zdotc( weights, F_, &val ); x^H y
      
      if( fbinX < fftLen2_ ){
	gsl_vector_complex_set(vector_, fbinX, val);
	gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2_, val);
    }
  }
  else{
    throw j_error("halfBandShift_ == true is not implemented yet\n");
  }

  increment_();
  return vector_;
}

void EigenBeamformer::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( NULL != snapshot_array_ )
    snapshot_array_->zero();

  if( NULL != st_snapshot_array_ )
    st_snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

/**
   @brief set the geometry of the EigenMikeR
*/
void EigenBeamformer::set_eigenmike_geometry( )
{
   gsl_vector *theta_s = gsl_vector_alloc( 32 );
   gsl_vector *phi_s   = gsl_vector_alloc( 32 );

   gsl_vector_set( theta_s, 0, 69 * M_PI / 180 );
   gsl_vector_set( phi_s,   0, 0.0 );

   gsl_vector_set( theta_s, 1, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   1, 32 * M_PI / 180 );

   gsl_vector_set( theta_s, 2, 111 * M_PI / 180 );
   gsl_vector_set( phi_s,   2, 0 );

   gsl_vector_set( theta_s, 3, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   3, 328 * M_PI / 180 );

   gsl_vector_set( theta_s, 4, 32 * M_PI / 180 );
   gsl_vector_set( phi_s,   4, 0);

   gsl_vector_set( theta_s, 5, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   5, 45 * M_PI / 180 );

   gsl_vector_set( theta_s, 6, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   6, 69 * M_PI / 180 );

   gsl_vector_set( theta_s, 7, 125 * M_PI / 180);
   gsl_vector_set( phi_s,   7, 45  * M_PI / 180);

   gsl_vector_set( theta_s, 8, 148 * M_PI / 180);
   gsl_vector_set( phi_s,   8, 0 );

   gsl_vector_set( theta_s, 9, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   9, 315 * M_PI / 180 );

   gsl_vector_set( theta_s, 10, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   10, 291 * M_PI / 180 );

   gsl_vector_set( theta_s, 11, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   11, 315 * M_PI / 180 );

   gsl_vector_set( theta_s, 12, 21 * M_PI / 180 );
   gsl_vector_set( phi_s,   12, 91 * M_PI / 180 );

   gsl_vector_set( theta_s, 13, 58 * M_PI / 180 );
   gsl_vector_set( phi_s,   13, 90 * M_PI / 180 );

   gsl_vector_set( theta_s, 14, 121 * M_PI / 180 );
   gsl_vector_set( phi_s,   14, 90 * M_PI / 180 );

   gsl_vector_set( theta_s, 15, 159 * M_PI / 180 );
   gsl_vector_set( phi_s,   15, 89 * M_PI / 180 );

   gsl_vector_set( theta_s, 16, 69 * M_PI / 180 );
   gsl_vector_set( phi_s,   16, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 17, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   17, 212 * M_PI / 180 );

   gsl_vector_set( theta_s, 18, 111 * M_PI / 180 );
   gsl_vector_set( phi_s,   18, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 19, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   19, 148 * M_PI / 180 );

   gsl_vector_set( theta_s, 20, 32 * M_PI / 180 );
   gsl_vector_set( phi_s,   20, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 21, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   21, 225 * M_PI / 180 );

   gsl_vector_set( theta_s, 22, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   22, 249 * M_PI / 180 );

   gsl_vector_set( theta_s, 23, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   23, 225 * M_PI / 180 );
   
   gsl_vector_set( theta_s, 24, 148 * M_PI / 180 );
   gsl_vector_set( phi_s,   24, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 25, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   25, 135 * M_PI / 180 );

   gsl_vector_set( theta_s, 26, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   26, 111 * M_PI / 180 );

   gsl_vector_set( theta_s, 27, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   27, 135 * M_PI / 180 );

   gsl_vector_set( theta_s, 28, 21 * M_PI / 180 );
   gsl_vector_set( phi_s,   28, 269 * M_PI / 180 );

   gsl_vector_set( theta_s, 29, 58 * M_PI / 180 );
   gsl_vector_set( phi_s,   29, 270 * M_PI / 180 );

   gsl_vector_set( theta_s, 30, 122 * M_PI / 180 );
   gsl_vector_set( phi_s,   30, 270 * M_PI / 180 );

   gsl_vector_set( theta_s, 31, 159 * M_PI / 180 );
   gsl_vector_set( phi_s,   31, 271 * M_PI / 180 );

   for ( unsigned i = 0; i < 32; i++ ) {
     fprintf(stderr, "%d : %e %e\n", i, gsl_vector_get( theta_s, i ),
	     gsl_vector_get( phi_s, i ) );
     fflush(stderr);
   }

   set_array_geometry( 42, theta_s, phi_s );

   gsl_vector_free( theta_s );
   gsl_vector_free( phi_s );
}

void EigenBeamformer::set_array_geometry( double a, gsl_vector *theta_s, gsl_vector *phi_s )
{
  if ( theta_s->size != phi_s->size ) {
    jparameter_error("The numbers of the sensor positions have to be the same but %lu != %lu\n", theta_s->size, phi_s->size);
  }

  a_ = a; // radius

  if ( NULL != theta_s_ )
    gsl_vector_free( theta_s_ );
  theta_s_ = gsl_vector_alloc( theta_s->size );

  if ( NULL != phi_s_ )
    gsl_vector_free( phi_s_ );
  phi_s_ = gsl_vector_alloc( phi_s->size );

  for ( unsigned i = 0; i < theta_s->size; i++) {
    gsl_vector_set( theta_s_, i, gsl_vector_get( theta_s, i ) );
    gsl_vector_set( phi_s_,   i, gsl_vector_get( phi_s,   i ) );
  }

  if ( false == calc_spherical_harmonics_at_each_position_( theta_s, phi_s ) ) {
    fprintf(stderr,"calc_spherical_harmonics_at_each_position_() failed\n");
  }
}

bool EigenBeamformer::calc_spherical_harmonics_at_each_position_( gsl_vector *theta_s, gsl_vector *phi_s )
{
  int nChan = (int)theta_s->size;

  if( NULL != F_ )
    gsl_vector_complex_free( F_ );
  F_ = gsl_vector_complex_alloc( dim_ );

  if( NULL != sh_s_ )
    free(sh_s_);
  sh_s_ = (gsl_vector_complex **)malloc(dim_*sizeof(gsl_vector_complex *));
  if( NULL == sh_s_ ){
    fprintf(stderr,"calc_spherical_harmonics_at_each_position_ : cannot allocate memory\n");
    return false;
  }
  for(int dimX=0;dimX<dim_;dimX++)
    sh_s_[dimX] = gsl_vector_complex_alloc( nChan );

  // compute spherical harmonics for each sensor
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
	gsl_complex Ymn_s = sphericalHarmonic( m, n, 
					       gsl_vector_get( theta_s, chanX ), 
					       gsl_vector_get( phi_s, chanX ) );
	//gsl_vector_complex_set( sh_s_[chanX], idx, gsl_complex_div_real( Ymn_s, 2 * sqrt( M_PI ) ) );// based on Meyer and Elko's descripitons. Do not gsl_complex_conjugate
	gsl_vector_complex_set( sh_s_[idx], chanX, gsl_complex_conjugate( Ymn_s ) );// based on Rafaely's definition
      }
      idx++;
    }
  }

  return true;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void EigenBeamformer::set_look_direction( double theta, double phi )
{
  //fprintf(stderr, "EigenBeamformer::set_look_direction\n" );
  fflush(stderr);

  if( theta < 0 || theta >  M_PI ){
    fprintf(stderr,"ERROR: Out of range of theta\n");
  }

  theta_ = theta;
  phi_   = phi;

  if( NULL == mode_mplitudes_ )
    calc_mode_amplitudes_();

  if( NULL == bfweight_vec_[0] )
    alloc_steering_unit_(1);

  calc_steering_unit_( 0, false /* isGSC */  );
}

const gsl_matrix_complex *EigenBeamformer::mode_amplitudes()
{
  if( a_ == 0.0 ){
    fprintf(stderr,"set the radius of the rigid sphere\n");
    return NULL;
  }

  if( NULL == mode_mplitudes_ )
    if( false == calc_mode_amplitudes_() ){
      fprintf(stderr,"Did you set the multi-channel data?\n");
      return NULL;
    }

  return mode_mplitudes_;
}

const gsl_vector *EigenBeamformer::array_geometry( int type )
{
  if( type == 0 ){
    return theta_s_;
  }
  return phi_s_;
}

void EigenBeamformer::alloc_image_( bool flag )
{
  if( a_ == 0.0 ){
    throw j_error("set the radius of the rigid sphere\n");
  }

  if( NULL == snapshot_array_ )
    snapshot_array_ = new SnapShotArray( fftLen_, channelList_.size() );

  if( NULL == st_snapshot_array_ ){
    st_snapshot_array_ = new SnapShotArray( fftLen_, dim_ );
    st_snapshot_array_->zero();
  }

  if( NULL == mode_mplitudes_ )
    calc_mode_amplitudes_();

  if( NULL == bfweight_vec_[0] && flag ){
    alloc_steering_unit_(1);
    calc_steering_unit_(0);
  }
}

bool EigenBeamformer::calc_mode_amplitudes_()
{
  mode_mplitudes_ = gsl_matrix_complex_alloc( fftLen2_+1, maxOrder_ );
  for (unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++) {
    double ka = 2.0 * M_PI * fbinX * a_ * samplerate_ / ( fftLen_ * SSPEED );

    for(int n=0;n<maxOrder_;n++){/* order */
      gsl_matrix_complex_set( mode_mplitudes_, fbinX, n, modeAmplitude( n, ka ) );
    }
  }

  return true;
}

bool EigenBeamformer::alloc_steering_unit_( int unitN )
{
  for(unsigned unitX=0;unitX<bfweight_vec_.size();unitX++){
    if( NULL != bfweight_vec_[unitX] ){
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;
  }

  if( bfweight_vec_.size() != unitN )
    bfweight_vec_.resize( unitN );

  for(unsigned unitX=0;unitX<unitN;unitX++){
    bfweight_vec_[unitX] = new BeamformerWeights( fftLen_, dim_, halfBandShift_, NC_ );
  }

  return true;
}

bool EigenBeamformer::calc_steering_unit_( int unitX, bool isGSC )
{
#if 0
  const String thisname = name();
  printf("calc_steering_unit_ %s %d\n", thisname.c_str(), (int)isGSC);
#endif

  gsl_vector_complex* weights;

  if( unitX >= bfweight_vec_.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  //fprintf(stderr, "calcDCWeights\n");
  weights = bfweight_vec_[unitX]->wq_f(0); 
  calcDCWeights( maxOrder_, weights );

  for(unsigned fbinX=1;fbinX<=fftLen2_;fbinX++){
    //fprintf(stderr, "calc_weights_(%d)\n", fbinX);
    weights = bfweight_vec_[unitX]->wq_f(fbinX); 
    calc_weights_( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      bfweight_vec_[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  bfweight_vec_[unitX]->setTimeAlignment();

  return true;
}


void planeWaveOnSphericalAperture( double ka, double theta, double phi, 
				   gsl_vector *theta_s, gsl_vector *phi_s, gsl_vector_complex *p )
{
  for(unsigned chanX=0;chanX<theta_s->size;chanX++){
    double theta_sn = gsl_vector_get( theta_s, chanX );
    double phi_sn   = gsl_vector_get( phi_s,   chanX );
    double ang = ka * ( sin(theta_sn) * sin(theta) * cos( phi_sn - phi ) + cos(theta_sn) * cos(theta) );
    gsl_vector_complex_set( p, chanX,  gsl_complex_polar( 1, ang ) );
  }
  return;
}

/**
   @brief compute the beam pattern at a frequnecy
   @param unsigned fbinX[in] frequency bin
   @param double theta[in] the look direction
   @param double phi[in]   the look direction
   @return the matrix of the beam patters where each column and row indicate the direction of the plane wave impinging on the sphere.
 */
gsl_matrix *EigenBeamformer::beampattern( unsigned fbinX, double theta, double phi,
					     double minTheta, double maxTheta, double minPhi, double maxPhi, double widthTheta, double widthPhi )
{
  float nTheta = ( maxTheta - minTheta ) / widthTheta + 0.5 + 1;
  float nPhi   = ( maxPhi - minPhi ) / widthPhi + 0.5 + 1;
  double ka = 2.0 * M_PI * fbinX * a_ * samplerate_ / ( fftLen_ * SSPEED );
  gsl_vector_complex *p = gsl_vector_complex_alloc( theta_s_->size );

  if( NULL != beampattern_ )
    gsl_matrix_free( beampattern_ );
  beampattern_ = gsl_matrix_alloc( (int)nTheta, (int)nPhi );
						
  set_look_direction( theta, phi );
  unsigned thetaIdx = 0;
  for(double theta=minTheta;thetaIdx<(int)nTheta;theta+=widthTheta,thetaIdx++){
    unsigned phiIdx = 0;;
    for(double phi=minPhi;phiIdx<(int)nPhi;phi+=widthPhi,phiIdx++){
      gsl_complex val;
      
      planeWaveOnSphericalAperture( ka, theta, phi, theta_s_, phi_s_, p );
      spherical_harmonics_transformation_( maxOrder_, p, sh_s_, F_ );
      gsl_vector_complex *weights = bfweight_vec_[0]->wq_f(fbinX);
      gsl_blas_zdotc( weights, F_, &val );
      //gsl_blas_zdotu( weights, F_, &val );
      gsl_matrix_set( beampattern_, thetaIdx, phiIdx, gsl_complex_abs( val ) );
    }
  }

  gsl_vector_complex_free( p );

  return beampattern_;
}


// ----- definition for class DOAEstimatorSRPEB' -----
// 

DOAEstimatorSRPEB::DOAEstimatorSRPEB( unsigned nBest, unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  //_beamformer = new EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, nm );
}

DOAEstimatorSRPEB::~DOAEstimatorSRPEB()
{
}

void DOAEstimatorSRPEB::calc_steering_unit_table_()
{
  int nChan = (int)chanN();
  if( nChan == 0 ){
    return;
  }

  this->alloc_image_( false );

  nTheta_ = (unsigned)( ( maxTheta_ - minTheta_ ) / widthTheta_ + 0.5 );
  nPhi_   = (unsigned)( ( maxPhi_ - minPhi_ ) / widthPhi_ + 0.5 );
  int maxUnit  = nTheta_ * nPhi_;

  svTbl_.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    svTbl_[i] = (gsl_vector_complex **)malloc((fbinMax_+1)*sizeof(gsl_vector_complex *));
    if( NULL == svTbl_[i] ){
      fprintf(stderr,"could not allocate image : %d\n", maxUnit );
    }
    for(unsigned fbinX=0;fbinX<=fbinMax_;fbinX++)
      svTbl_[i][fbinX] = gsl_vector_complex_calloc( dim_ );
  }

  if( NULL != accRPs_ )
    gsl_vector_free( accRPs_ );
  accRPs_ = gsl_vector_calloc( maxUnit );

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=minPhi_;phiIdx<nPhi_;phi+=widthPhi_,phiIdx++){
      gsl_vector_complex *weights;

      set_look_direction( theta, phi );
      weights = svTbl_[unitX][0];
      for(unsigned n=0;n<weights->size;n++)
	gsl_vector_complex_set( weights, n, gsl_complex_rect(1,0) );
      for(unsigned fbinX=fbinMin_;fbinX<=fbinMax_;fbinX++){
	weights = svTbl_[unitX][fbinX];
	calc_weights_( fbinX, weights ); // call the function through the pointer
	//for(unsigned n=0;n<weights->size;n++)
	//gsl_vector_complex_set( weights, n, gsl_complex_conjugate( gsl_vector_complex_get( weights, n ) ) );
      }
      unitX++;
    }
  }
  
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  table_initialized_ = true;
}

float DOAEstimatorSRPEB::calc_response_power_( unsigned unitX )
{  
  const gsl_vector_complex* F;       /* spherical transformation coefficient */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;
  double rp  = 0.0;
  
  if( halfBandShift_ == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = fbinMin_; fbinX <= fbinMax_; fbinX++) {
      F = st_snapshot_array_->snapshot(fbinX);
      weights = svTbl_[unitX][fbinX];
      gsl_blas_zdotc( weights, F, &val ); // x^H y

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

  return rp / ( fbinMax_ - fbinMin_ + 1 ); // ( X0^2 + X1^2 + ... + XN^2 )
}

const gsl_vector_complex* DOAEstimatorSRPEB::next( int frame_no )
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<nBest_;n++){
    gsl_vector_set( nBestRPs_, n, -10e10 );
    gsl_matrix_set( argMaxDOAs_, n, 0, -M_PI);
    gsl_matrix_set( argMaxDOAs_, n, 1, -M_PI);
  }

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

  // update the spherical harmonics transformation coefficients
  if( halfBandShift_ == false ){
    const gsl_vector_complex* XVec;    /* snapshot at each frequency */

    for (unsigned fbinX = fbinMin_; fbinX <=  fbinMax_; fbinX++) {
      XVec    = snapshot_array_->snapshot(fbinX);
      spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
      st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, fbinX );
    }
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=minPhi_;phiIdx<nPhi_;phi+=widthPhi_,phiIdx++){
      //set_look_direction( theta, phi );
      rp = calc_response_power_( unitX );
      gsl_vector_set( accRPs_, unitX, gsl_vector_get( accRPs_, unitX ) + rp );
      unitX++;
#ifdef __MBDEBUG__
      gsl_matrix_set( rpMat_, thetaIdx, phiIdx, rp);
#endif /* #ifdef __MBDEBUG__ */
      //   fprintf( stderr, "t=%0.8f p=%0.8f rp=%e\n" , theta, phi, rp );
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

  increment_();
  return vector_;
}

void DOAEstimatorSRPEB::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- definition for class `SphericalDSBeamformer' -----
//

SphericalDSBeamformer::SphericalDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalDSBeamformer::~SphericalDSBeamformer()
{}

gsl_vector *SphericalDSBeamformer::calc_wng()
{
  if( NULL == WNG_ )
    WNG_ = gsl_vector_alloc( fftLen2_ + 1 );
  double norm = theta_s_->size / ( M_PI * M_PI );

  for (unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++) {
    double val = 0;
    for(int n=0;n<maxOrder_;n++){/* order */
      gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); 
      double bn2 =  gsl_complex_abs2( bn );
      val += ( (2*n+1) * bn2 );
    }
    gsl_vector_set( WNG_, fbinX, val * val * norm );
  }

  return WNG_;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::calc_steering_unit_()
 */
void SphericalDSBeamformer::calc_weights_( unsigned fbinX,
					  gsl_vector_complex *weights )
{
  //fprintf( stderr, "SphericalDSBeamformer::calc_weights_\n" );

  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, theta_, phi_ ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx, gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==weights_normalized_ )
    normalize_weights_( weights, wgain_ );

  return;
}

bool SphericalDSBeamformer::calc_spherical_harmonics_at_each_position_( gsl_vector *theta_s, gsl_vector *phi_s )
{
  int nChan = (int)theta_s->size;

  if( NULL != F_ )
    gsl_vector_complex_free( F_ );
  F_ = gsl_vector_complex_alloc( dim_ );

  if( NULL != sh_s_ )
    free(sh_s_);
  sh_s_ = (gsl_vector_complex **)malloc(dim_*sizeof(gsl_vector_complex *));
  if( NULL == sh_s_ ){
    fprintf(stderr,"calc_spherical_harmonics_at_each_position_ : cannot allocate memory\n");
    return false;
  }
  for(int dimX=0;dimX<dim_;dimX++)
    sh_s_[dimX] = gsl_vector_complex_alloc( nChan );

  // compute spherical harmonics for each sensor
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
	gsl_complex Ymn_s = sphericalHarmonic( m, n, 
					       gsl_vector_get( theta_s, chanX ), 
					       gsl_vector_get( phi_s, chanX ) );
	gsl_vector_complex_set( sh_s_[idx], chanX, gsl_complex_conjugate( Ymn_s ) );// 
      }
      idx++;
    }
  }

  return true;
}

// ----- definition for class `DualSphericalDSBeamformer' -----
//

DualSphericalDSBeamformer::DualSphericalDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  bfweight_vec2_.resize(1); // the steering vector
  bfweight_vec2_[0] = NULL;

}

DualSphericalDSBeamformer::~DualSphericalDSBeamformer()
{}

bool DualSphericalDSBeamformer::alloc_steering_unit_( int unitN )
{
  for(unsigned unitX=0;unitX<bfweight_vec_.size();unitX++){
    if( NULL != bfweight_vec_[unitX] ){
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;

    if( NULL != bfweight_vec2_[unitX] ){
      delete bfweight_vec2_[unitX];
    }
    bfweight_vec2_[unitX] = NULL;
  }

  if( bfweight_vec_.size() != unitN ){
    bfweight_vec_.resize( unitN );
    bfweight_vec2_.resize( unitN );
  }
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    bfweight_vec_[unitX]  = new BeamformerWeights( fftLen_, dim_, halfBandShift_, NC_ );
    bfweight_vec2_[unitX] = new BeamformerWeights( fftLen_, chanN(), halfBandShift_, NC_ );
  }
  
  return true;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::calc_steering_unit_()
 */
void DualSphericalDSBeamformer::calc_weights_( unsigned fbinX,
					  gsl_vector_complex *weights )
{
  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, theta_, phi_ ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx,  gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==weights_normalized_ )
    normalize_weights_( weights, wgain_ );

  {//  construct the delay-and-sum beamformer in the normal subband domain
    gsl_vector* delays = gsl_vector_alloc( chanN() );

    calc_time_delays_of_spherical_array_( theta_, phi_, chanN(), a_, theta_s_, phi_s_, delays );
    bfweight_vec2_[0]->calcMainlobe( samplerate_, delays, false );
    bfweight_vec2_[0]->setTimeAlignment();
    gsl_vector_free( delays );
  }

  return;
}

// ----- definition for class DOAEstimatorSRPSphDSB' -----
// 

DOAEstimatorSRPSphDSB::DOAEstimatorSRPSphDSB( unsigned nBest, unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
}

DOAEstimatorSRPSphDSB::~DOAEstimatorSRPSphDSB()
{
}

void DOAEstimatorSRPSphDSB::calc_steering_unit_table_()
{
  int nChan = (int)chanN();
  if( nChan == 0 ){
    return;
  }

  this->alloc_image_( false );

  nTheta_ = (unsigned)( ( maxTheta_ - minTheta_ ) / widthTheta_ + 0.5 );
  nPhi_   = (unsigned)( ( maxPhi_ - minPhi_ ) / widthPhi_ + 0.5 );
  int maxUnit  = nTheta_ * nPhi_;

  svTbl_.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    svTbl_[i] = (gsl_vector_complex **)malloc((fbinMax_+1)*sizeof(gsl_vector_complex *));
    if( NULL == svTbl_[i] ){
      fprintf(stderr,"could not allocate image : %d\n", maxUnit );
    }
    for(unsigned fbinX=0;fbinX<=fbinMax_;fbinX++)
      svTbl_[i][fbinX] = gsl_vector_complex_calloc( dim_ );
  }

  accRPs_ = gsl_vector_calloc( maxUnit );
   
  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=minPhi_;phiIdx<nPhi_;phi+=widthPhi_,phiIdx++){
      gsl_vector_complex *weights;

      set_look_direction( theta, phi );
      weights = svTbl_[unitX][0];
      for(unsigned n=0;n<weights->size;n++)
	gsl_vector_complex_set( weights, n, gsl_complex_rect(1,0) );
      for(unsigned fbinX=fbinMin_;fbinX<=fbinMax_;fbinX++){
	weights = svTbl_[unitX][fbinX];
	calc_weights_( fbinX, weights ); // call the function through the pointer
	//for(unsigned n=0;n<weights->size;n++)
	//gsl_vector_complex_set( weights, n, gsl_complex_conjugate( gsl_vector_complex_get( weights, n ) ) );
      }
      unitX++;
    }
    
  }
  
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  table_initialized_ = true;
}

float DOAEstimatorSRPSphDSB::calc_response_power_( unsigned unitX )
{
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  const gsl_vector_complex* F;       /* spherical transformation coefficient */
  gsl_complex val;
  double rp  = 0.0;

  if( halfBandShift_ == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = fbinMin_; fbinX <= fbinMax_; fbinX++) {
      F = st_snapshot_array_->snapshot(fbinX);
      weights = svTbl_[unitX][fbinX];
      gsl_blas_zdotc( weights, F, &val ); // x^H y
      
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

  return rp / ( fbinMax_ - fbinMin_ + 1 ); // ( X0^2 + X1^2 + ... + XN^2 )
}

const gsl_vector_complex* DOAEstimatorSRPSphDSB::next( int frame_no )
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<nBest_;n++){
    gsl_vector_set( nBestRPs_, n, -10e10 );
    gsl_matrix_set( argMaxDOAs_, n, 0, -M_PI);
    gsl_matrix_set( argMaxDOAs_, n, 1, -M_PI);
  }

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

  // update the spherical harmonics transformation coefficients
  if( halfBandShift_ == false ){
    const gsl_vector_complex* XVec;    /* snapshot at each frequency */

    for (unsigned fbinX = fbinMin_; fbinX <=  fbinMax_; fbinX++) {
      XVec    = snapshot_array_->snapshot(fbinX);
      spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
      st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, fbinX );
    }
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=minTheta_;thetaIdx<nTheta_;theta+=widthTheta_,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=minPhi_;phiIdx<nPhi_;phi+=widthPhi_,phiIdx++){
      //set_look_direction( theta, phi );
      rp = calc_response_power_( unitX );
      gsl_vector_set( accRPs_, unitX, gsl_vector_get( accRPs_, unitX ) + rp );
      unitX++;
#ifdef __MBDEBUG__
      gsl_matrix_set( rpMat_, thetaIdx, phiIdx, rp);
#endif /* #ifdef __MBDEBUG__ */
      //   fprintf( stderr, "t=%0.8f p=%0.8f rp=%e\n" , theta, phi, rp );
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

  increment_();
  return vector_;
}

void DOAEstimatorSRPSphDSB::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( snapshot_array_ != NULL )
    snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

// ----- definition for class `SphericalDSBeamformer' -----
//

SphericalHWNCBeamformer::SphericalHWNCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, float ratio, const String& nm ):
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  ratio_ = ratio;
}

SphericalHWNCBeamformer::~SphericalHWNCBeamformer()
{
}

gsl_vector *SphericalHWNCBeamformer::calc_wng()
{
  double nrm = theta_s_->size / ( 16 * M_PI * M_PI );

  if( NULL == WNG_ )
    WNG_ = gsl_vector_calloc( fftLen_/2+1 );

  for (unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++) {
    double val = 0.0;
    double wng;

    for(int n=0;n<maxOrder_;n++){
      double bn2 = gsl_complex_abs2( gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ) );
      val += ( ( 2 * n + 1 ) * bn2 );
    }
    wng = nrm * val * ratio_;
    gsl_vector_set( WNG_, fbinX, wng );
  }

  //fprintf(stderr,"%e\n", wng);
  return WNG_;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::calc_steering_unit_()
 */
void SphericalHWNCBeamformer::calc_weights_( unsigned fbinX,
					    gsl_vector_complex *weights )
{
  unsigned nChan = (unsigned)phi_s_->size;
  unsigned norm = dim_ * nChan;

  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn ) + sigma2_;
    double      de  = norm * bn2;
    gsl_complex in;
    gsl_complex inbn;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    inbn = gsl_complex_mul( in, bn );
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
      gsl_complex YmnA = gsl_complex_conjugate(sphericalHarmonic( m, n, theta_, phi_ ));

      weight = gsl_complex_div_real( gsl_complex_mul( gsl_complex_mul_real( YmnA, 4 * M_PI ), inbn ), de ); // HMDI beamfomrer; see S Yan's paper
      gsl_vector_complex_set( weights, idx, weight ); 
      idx++;
    }
  }

  if( ratio_ > 0.0 ){ 
    // control the white noise gain
    if( NULL == WNG_ ){ calc_wng();}
    double wng = gsl_vector_get( WNG_, fbinX );
    normalize_weights_( weights, 2 * sqrt( M_PI / ( nChan * wng) ) );
  }
  else{
    double coeff = ( 16 * M_PI * M_PI ) / ( nChan * maxOrder_ * maxOrder_ );
    gsl_blas_zdscal( coeff, weights );
    //for(unsigned i=0;i<weights->size;i++)
    //  gsl_vector_complex_set( weights, i, gsl_complex_mul_real( gsl_vector_complex_get( weights, i ), coeff ) );
  }

  return;
}

// ----- definition for class `SphericalGSCBeamformer' -----
// 

SphericalGSCBeamformer::SphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalGSCBeamformer::~SphericalGSCBeamformer()
{}

const gsl_vector_complex* SphericalGSCBeamformer::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_vector_complex* wl_f;
  gsl_complex val;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();
  
  if( halfBandShift_ == false ){
    // calculate a direct component.
    XVec    = snapshot_array_->snapshot(0);
    spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
    st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, 0 );
    wq_f = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, F_, &val );
    gsl_vector_complex_set(vector_, 0, val);
    //gsl_vector_complex_set(vector_, 0, gsl_vector_complex_get( XVec, XVec->size/2) );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++) {
      XVec  = snapshot_array_->snapshot(fbinX);
      spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
      st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, fbinX );
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);

      calc_gsc_output( F_, wl_f, wq_f, &val, weights_normalized_ );
      if( fbinX < fftLen2_ ){
        gsl_vector_complex_set(vector_, fbinX, val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2_, val);
    }
  }
  else{
    throw j_error("halfBandShift_ == true is not implemented yet\n");
  }

  increment_();
  return vector_;
}

void SphericalGSCBeamformer::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( NULL != snapshot_array_ )
    snapshot_array_->zero();
  
  if( NULL != st_snapshot_array_ )
    st_snapshot_array_->zero();
  
  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void SphericalGSCBeamformer::set_look_direction( double theta, double phi )
{
  //fprintf(stderr," SphericalGSCBeamformer::set_look_direction()\n");

  theta_ = theta;
  phi_   = phi;

  if( NULL == mode_mplitudes_ )
    calc_mode_amplitudes_();

  if( NULL == bfweight_vec_[0] )
    alloc_steering_unit_(1);

  calc_steering_unit_( 0, true /* isGSC */  );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SphericalGSCBeamformer::set_active_weights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call set_look_direction() once\n");
  }

  bfweight_vec_[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 

SphericalHWNCGSCBeamformer::SphericalHWNCGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, float ratio, const String& nm )
  :SphericalHWNCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm )
{}

SphericalHWNCGSCBeamformer::~SphericalHWNCGSCBeamformer()
{}

const gsl_vector_complex* SphericalHWNCGSCBeamformer::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_vector_complex* wl_f;
  gsl_complex val;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == false ){
    // calculate a direct component.
    XVec    = snapshot_array_->snapshot(0);
    spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
    st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, 0 );
    wq_f = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, F_, &val );
    gsl_vector_complex_set(vector_, 0, val);
    //gsl_vector_complex_set(vector_, 0, gsl_vector_complex_get( XVec, XVec->size/2) );
    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++) {
      XVec  = snapshot_array_->snapshot(fbinX);
      spherical_harmonics_transformation_( maxOrder_, XVec, sh_s_, F_ );
      st_snapshot_array_->set_snapshots( (const gsl_vector_complex*)F_, fbinX );
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      wl_f = bfweight_vec_[0]->wl_f(fbinX);

      calc_gsc_output( F_, wl_f, wq_f, &val, weights_normalized_ );
      if( fbinX < fftLen2_ ){
        gsl_vector_complex_set(vector_, fbinX, val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2_, val);
    }
  }
  else{
    throw j_error("halfBandShift_ == true is not implemented yet\n");
  }

  increment_();
  return vector_;
}

void SphericalHWNCGSCBeamformer::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( NULL != snapshot_array_ )
    snapshot_array_->zero();
  
  if( NULL != st_snapshot_array_ )
    st_snapshot_array_->zero();
  
  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void SphericalHWNCGSCBeamformer::set_look_direction( double theta, double phi )
{
  //fprintf(stderr," SphericalGSCBeamformer::set_look_direction()\n");

  theta_ = theta;
  phi_   = phi;

  if( NULL == mode_mplitudes_ )
    calc_mode_amplitudes_();

  if( NULL == bfweight_vec_[0] )
    alloc_steering_unit_(1);

  calc_steering_unit_( 0, true /* isGSC */  );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SphericalHWNCGSCBeamformer::set_active_weights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == bfweight_vec_.size() ){
    throw  j_error("call set_look_direction() once\n");
  }

  bfweight_vec_[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

// ----- definition for class `DualSphericalGSCBeamformer' -----
// 

DualSphericalGSCBeamformer::DualSphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalGSCBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm)
{}

DualSphericalGSCBeamformer::~DualSphericalGSCBeamformer()
{}

bool DualSphericalGSCBeamformer::alloc_steering_unit_( int unitN )
{
  for(unsigned unitX=0;unitX<bfweight_vec_.size();unitX++){
    if( NULL != bfweight_vec_[unitX] ){
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;

    if( NULL != bfweight_vec2_[unitX] ){
      delete bfweight_vec2_[unitX];
    }
    bfweight_vec2_[unitX] = NULL;
  }

  if( bfweight_vec_.size() != unitN ){
    bfweight_vec_.resize( unitN );
    bfweight_vec2_.resize( unitN );
  }
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    bfweight_vec_[unitX]  = new BeamformerWeights( fftLen_, dim_, halfBandShift_, NC_ );
    bfweight_vec2_[unitX] = new BeamformerWeights( fftLen_, chanN(), halfBandShift_, NC_ );
  }
  
  return true;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::calc_steering_unit_()
 */
void DualSphericalGSCBeamformer::calc_weights_( unsigned fbinX, gsl_vector_complex *weights )
{
  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, theta_, phi_ ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx,  gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==weights_normalized_ )
    normalize_weights_( weights, wgain_ );

  {//  construct the delay-and-sum beamformer in the normal subband domain
    gsl_vector* delays = gsl_vector_alloc( chanN() );

    calc_time_delays_of_spherical_array_( theta_, phi_, chanN(), a_, theta_s_, phi_s_, delays );
    bfweight_vec2_[0]->calcMainlobe( samplerate_, delays, false );
    bfweight_vec2_[0]->setTimeAlignment();
    gsl_vector_free( delays );
  }

  return;
}


// ----- definition for class `SphericalMOENBeamformer' -----
// 

SphericalMOENBeamformer::SphericalMOENBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  dthreshold_ = 1.0E-8;
  is_term_fixed_ = false;
  CN_ = 2.0 / ( maxOrder * maxOrder ); // maxOrder = N + 1
  bf_order_ = maxOrder;

  A_      = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  fixedW_ = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  if( A_ == NULL || fixedW_ == NULL ){
    throw jallocation_error("SphericalMOENBeamformer: gsl_matrix_complex_alloc failed\n");
  }

  BN_ = (gsl_vector_complex** )malloc( (fftLen/2+1) * sizeof(gsl_vector_complex*) );
  if( BN_ == NULL ){
    throw jallocation_error("SphericalMOENBeamformer: gsl_vector_complex_alloc failed\n");
  }

  diagonal_weights_ = (float *)calloc( (fftLen/2+1), sizeof(float) );
  if( diagonal_weights_ == NULL ){
    throw jallocation_error("SphericalMOENBeamformer: cannot allocate RAM\n");
  }

  for( unsigned fbinX=0;fbinX<=fftLen/2;fbinX++){
    A_[fbinX]       = NULL;
    fixedW_[fbinX]  = NULL;
    BN_[fbinX]      = NULL;
    diagonal_weights_[fbinX] = 0.0;
  }

}

SphericalMOENBeamformer::~SphericalMOENBeamformer()
{
  unsigned fftLen2 = fftLen_ / 2;

  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( A_[fbinX] != NULL )
      gsl_matrix_complex_free( A_[fbinX] );
    if( fixedW_[fbinX] != NULL )
      gsl_matrix_complex_free( fixedW_[fbinX] );
    if( BN_[fbinX] != NULL )
      gsl_vector_complex_free( BN_[fbinX] );
  }
  free(A_);
  free(fixedW_);
  free(BN_);
  free(diagonal_weights_);
}

const gsl_vector_complex* SphericalMOENBeamformer::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_complex val;

  this->alloc_image_();
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no);
    if( true==(*itr)->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  if( halfBandShift_ == false ){
    // calculate a direct component.
    XVec = snapshot_array_->snapshot(0);
    wq_f = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, XVec, &val );
    //gsl_blas_zdotu( wq_f, XVec, &val );
    gsl_vector_complex_set(vector_, 0, val);

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++) {
      XVec = snapshot_array_->snapshot(fbinX);
      wq_f = bfweight_vec_[0]->wq_f(fbinX);
      gsl_blas_zdotc( wq_f, XVec, &val );
      //gsl_blas_zdotu( wq_f, XVec, &val );

      if( fbinX < fftLen2_ ){
        gsl_vector_complex_set(vector_, fbinX, val);
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2_, val);
    }
  }
  else{
    throw j_error("halfBandShift_ == true is not implemented yet\n");
  }

  increment_();
  return vector_;
}

void SphericalMOENBeamformer::reset()
{
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if ( NULL != snapshot_array_ )
    snapshot_array_->zero();

  if( NULL != st_snapshot_array_ )
    st_snapshot_array_->zero();

  VectorComplexFeatureStream::reset();
  is_end_ = false;
}

void SphericalMOENBeamformer::set_diagonal_looading( unsigned fbinX, float diagonalWeight )
{
  if( fbinX > fftLen_/2 ){
    jparameter_error("SphericalMOENBeamformer::set_diagonal_looading() : Invalid freq. bin %d\n", fbinX);
  }
  diagonal_weights_[fbinX] = diagonalWeight;
}

/*
  @brief calcualte matrices, A_ and BN_.

  @note this function is called by calc_steering_unit_() which is called by set_look_direction().
 */
void SphericalMOENBeamformer::calc_weights_( unsigned fbinX, gsl_vector_complex *weights )
{
  unsigned nChan = theta_s_->size;

  if( A_[fbinX] == NULL ){
    A_[fbinX]  = gsl_matrix_complex_alloc( dim_, nChan );
    BN_[fbinX] = gsl_vector_complex_calloc( dim_ );
  }
  else
    gsl_vector_complex_set_zero( BN_[fbinX] );
  
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;
    
    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }

    for( int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
        if( false == is_term_fixed_ ) {
          gsl_complex YAmn_s = gsl_vector_complex_get( sh_s_[idx], chanX );
          gsl_complex val = gsl_complex_mul( YAmn_s, gsl_complex_mul( in, bn ) );
          //gsl_complex val = gsl_complex_div( gsl_complex_conjugate(YAmn_s), gsl_complex_mul( in, bn ) );
          gsl_matrix_complex_set( A_[fbinX], idx, chanX, gsl_complex_mul_real( val, 4 * M_PI ) );
        }
        if( n < bf_order_ ){
	  //gsl_vector_complex_set( BN_[fbinX], idx, gsl_complex_mul_real( sphericalHarmonic( m, n, theta_, phi_ ), 2 * M_PI ) );
          gsl_vector_complex_set( BN_[fbinX], idx, gsl_complex_mul_real( gsl_complex_conjugate( sphericalHarmonic( m, n, theta_, phi_ ) ), 2 * M_PI ) );
        }
      }
      idx++;
    }
  }

  calc_moen_weights_( fbinX, weights, dthreshold_, is_term_fixed_, 0 );

  return;
}


bool SphericalMOENBeamformer::calc_moen_weights_( unsigned fbinX, gsl_vector_complex *weights, double dThreshold, bool calcFixedTerm, unsigned unitX )
{
  unsigned nChan = theta_s_->size;
  
  if( NULL != fixedW_[fbinX] ){
    gsl_matrix_complex_free( fixedW_[fbinX] );
  }
  fixedW_[fbinX] = gsl_matrix_complex_calloc( nChan, dim_ );
  
#if 0
  for(unsigned chanX=0;chanX<weights->size;chanX++)
    gsl_vector_complex_set( weights, chanX, gsl_complex_rect( 1.0, 0.0 ) );
#endif

  if( false == calcFixedTerm ){
    gsl_matrix_complex* tmp = gsl_matrix_complex_calloc( nChan, nChan );
    //gsl_matrix_complex* AH  = gsl_matrix_complex_calloc( nChan, dim_ );
    for(unsigned chanX=0;chanX<nChan;chanX++)
      gsl_matrix_complex_set( tmp, chanX, chanX, gsl_complex_rect( 1.0, 0.0 ) );

    gsl_blas_zherk( CblasUpper, CblasConjTrans, 1.0, A_[fbinX], diagonal_weights_[fbinX], tmp ); // A^H A + l^2 I
    // can be implemented in the faster way
    for(unsigned chanX=0;chanX<nChan;chanX++)
      for(unsigned chanY=chanX;chanY<nChan;chanY++)
        gsl_matrix_complex_set( tmp, chanY, chanX, gsl_complex_conjugate( gsl_matrix_complex_get( tmp, chanX, chanY) ) );
    if( false==pseudoinverse( tmp, tmp, dThreshold )){ //( A^H A + l^2 I )^{-1}
      fprintf(stderr,"fbinX %d : pseudoinverse() failed\n",fbinX);
#if 1
      for(unsigned chanX=0;chanX<nChan;chanX++){
        for(unsigned chanY=0;chanY<dim_;chanY++){
          fprintf(stderr,"%0.2e + i %0.2e, ", GSL_REAL(gsl_matrix_complex_get(tmp, chanX, chanY)), GSL_IMAG(gsl_matrix_complex_get(tmp, chanX, chanY)) );
        }
        fprintf(stderr,"\n");
      }
#endif
    }

    gsl_blas_zgemm( CblasNoTrans, CblasConjTrans, gsl_complex_rect( 1.0, 0.0 ), tmp, A_[fbinX], gsl_complex_rect( 0.0, 0.0 ), fixedW_[fbinX] ); //( A^H A + l^2 I )^{-1} A^H
    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( CN_, 0.0 ), (const gsl_matrix_complex*)fixedW_[fbinX], (const gsl_vector_complex*)BN_[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );// ( A^H A + l^2 I )^{-1} A^H BN
    gsl_matrix_complex_free( tmp );
  }
  else{
    //gsl_blas_zhemv( CblasUpper, gsl_complex_rect( CN_, 0.0 ), (const gsl_matrix_complex*)fixedW_[fbinX], (const gsl_matrix_complex*)BN_[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );
    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( CN_, 0.0 ), (const gsl_matrix_complex*)fixedW_[fbinX], (const gsl_vector_complex*)BN_[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );// ( A^H A + l^2 I )^{-1} A^H BN
  }

  if( true==weights_normalized_ )
    normalize_weights_( weights, wgain_ );

  return true;
}

bool SphericalMOENBeamformer::alloc_steering_unit_( int unitN )
{
  for(unsigned unitX=0;unitX<bfweight_vec_.size();unitX++){
    if( NULL != bfweight_vec_[unitX] ){
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;
  }

  if( bfweight_vec_.size() != unitN )
    bfweight_vec_.resize( unitN );

  for(unsigned unitX=0;unitX<unitN;unitX++){
    bfweight_vec_[unitX] = new BeamformerWeights( fftLen_, theta_s_->size, halfBandShift_, NC_ );
  }

  return true;
}

/**
   @brief compute the beam pattern at a frequnecy
   @param unsigned fbinX[in] frequency bin
   @param double theta[in] the look direction
   @param double phi[in]   the look direction
   @return the matrix of the beam patters where each column and row indicate the direction of the plane wave impinging on the sphere.
 */
gsl_matrix *SphericalMOENBeamformer::beampattern( unsigned fbinX, double theta, double phi,
						     double minTheta, double maxTheta, double minPhi, double maxPhi,
						     double widthTheta, double widthPhi )
{
  float nTheta = ( maxTheta - minTheta ) / widthTheta + 0.5 + 1;
  float nPhi   = ( maxPhi - minPhi ) / widthPhi + 0.5 + 1;
  double ka = 2.0 * M_PI * fbinX * a_ * samplerate_ / ( fftLen_ * SSPEED );
  gsl_vector_complex *p = gsl_vector_complex_alloc( theta_s_->size );

  if( NULL != beampattern_ )
    gsl_matrix_free( beampattern_ );
  beampattern_ = gsl_matrix_alloc( (int)nTheta, (int)nPhi );
						
  set_look_direction( theta, phi );
  unsigned thetaIdx = 0;
  for(double theta=minTheta;thetaIdx<(int)nTheta;theta+=widthTheta,thetaIdx++){
    unsigned phiIdx = 0;;
    for(double phi=minPhi;phiIdx<(int)nPhi;phi+=widthPhi,phiIdx++){
      gsl_complex val;

      planeWaveOnSphericalAperture( ka, theta, phi, theta_s_, phi_s_, p );
      gsl_vector_complex *weights = bfweight_vec_[0]->wq_f(fbinX);
      //gsl_blas_zdotc( weights, p, &val );
      gsl_blas_zdotu( weights, p, &val );
      gsl_matrix_set( beampattern_, thetaIdx, phiIdx, gsl_complex_abs( val ) );
    }
  }

  gsl_vector_complex_free( p );

  return beampattern_;
}



// ----- definition for class `SphericalSpatialDSBeamformer' -----
//

SphericalSpatialDSBeamformer::SphericalSpatialDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalSpatialDSBeamformer::~SphericalSpatialDSBeamformer()
{}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::calc_steering_unit_()
 */
void SphericalSpatialDSBeamformer::calc_weights_( unsigned fbinX,
						 gsl_vector_complex *weights )
{
  int nChan = (int)chanN();

  for ( int s = 0; s < nChan; s++ ) { /* channnel */
    /* compute the approximation of the sound pressure at sensor s with the spherical harmonics coefficients, */
    /* G(Omega_s,ka,Omega) = 4pi \sum_{n=0}^{N} i^n b_n(ka) \sum_{m=-n}^{n} Ymn(Omega_s) Ymn(Omega)^*         */
    gsl_complex weight = gsl_complex_rect( 0, 0 );
    for ( int n = 0, idx = 0; n < maxOrder_; n++ ) { /* order */
      gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); // bn = modeAmplitude( order, ka );
      gsl_complex in, inbn;
      gsl_complex tmp_weight;

      if ( 0 == ( n % 4 ) ) {
	in = gsl_complex_rect( 1, 0 );
      }
      else if ( 1 == ( n % 4 ) ) {
	in = gsl_complex_rect( 0, 1 );
      }
      else if ( 2 == ( n % 4 ) ) {
	in = gsl_complex_rect( -1, 0 );
      }
      else {
	in = gsl_complex_rect( 0, -1 );
      }
      inbn = gsl_complex_mul( in, bn );

      tmp_weight = gsl_complex_rect( 0, 0 );
      for ( int m = -n; m <= n; m++ ) { /* degree */
	tmp_weight = gsl_complex_add( tmp_weight, 
				      gsl_complex_mul( gsl_complex_conjugate( gsl_vector_complex_get( sh_s_[idx], s ) ), /* this has to be conjugated to get Ymn */
						       gsl_complex_conjugate( sphericalHarmonic( m, n, theta_, phi_ ) ) ) );
	idx++;
      }
      weight = gsl_complex_add( weight, gsl_complex_mul( inbn, tmp_weight ) );
    }
    gsl_vector_complex_set( weights, s, gsl_complex_mul_real( weight, 4 * M_PI / nChan ) );// the Euclidean norm gsl_blas_dznrm2( weights )
  }

  /* Normalization */
  // if ( true == weights_normalized_ )
  // normalize_weights_( weights, wgain_ ); /* <- Correct? */

  //double ka = 2.0 * M_PI * fbinX * a_ * samplerate_ / ( fftLen_ * SSPEED );
  //gsl_complex Gpw = gsl_complex_polar( 1.0, ka * cos( theta_ ) );
  //for ( unsigned i = 0; i < nChan; i++) {
  //gsl_vector_complex_set( weights, i, gsl_complex_conjugate(gsl_complex_mul( Gpw, norm_weight ) ) );
  //gsl_complex norm_weight = gsl_complex_div_real( gsl_vector_complex_get( weights, i ), nrm * nChan );
  //gsl_vector_complex_set( weights, i, gsl_complex_conjugate(gsl_complex_mul( Gpw, norm_weight ) ) );
  //}

  return;
}

const gsl_vector_complex* SphericalSpatialDSBeamformer::next( int frame_no )
{
  if ( frame_no == frame_no_ ) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->alloc_image_();
  for ( ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++ ) {
    const gsl_vector_complex* samp = ( *itr )->next( frame_no );
    if ( true == ( *itr )->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX );
    chanX++;
  }
  snapshot_array_->update();
  
  if ( halfBandShift_ == false ) {
    // calculate a direct component.

    XVec    = snapshot_array_->snapshot(0);
    weights = bfweight_vec_[0]->wq_f(0);

    gsl_blas_zdotc( weights, XVec, &val );
    gsl_vector_complex_set(vector_, 0, val );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for ( unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++ ) {
      XVec    = snapshot_array_->snapshot( fbinX );
      weights = bfweight_vec_[0]->wq_f( fbinX );

      gsl_blas_zdotc( weights, XVec, &val );      
      if( fbinX < fftLen2_ ){
	gsl_vector_complex_set(vector_, fbinX, val );
	gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(vector_, fftLen2_, val );
    }
  }
  else{
    throw j_error( "halfBandShift_ == true is not implemented yet\n" );
  }

  increment_();
  return vector_;
}

bool SphericalSpatialDSBeamformer::alloc_steering_unit_( int unitN )
{
  for ( unsigned unitX = 0; unitX < bfweight_vec_.size(); unitX++ ) {
    if ( NULL != bfweight_vec_[unitX] ) {
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;
  }

  if ( bfweight_vec_.size() != unitN )
    bfweight_vec_.resize( unitN );

  for ( unsigned unitX = 0; unitX < unitN; unitX++ ) {
    bfweight_vec_[unitX]
      = new BeamformerWeights( fftLen_, (int)chanN(), halfBandShift_, NC_ );
  }

  return true;
}

bool SphericalSpatialDSBeamformer::calc_steering_unit_( int unitX, bool isGSC )
{
  gsl_vector_complex* weights;

  if( unitX >= bfweight_vec_.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  // weights = bfweight_vec_[unitX]->wq_f(0); 
  // calcDCWeights( maxOrder_, weights );

  // for(unsigned fbinX=1;fbinX<=fftLen2_;fbinX++){
  for ( unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++ ) {
    // fprintf(stderr, "calc_weights_(%d)\n", fbinX);
    weights = bfweight_vec_[unitX]->wq_f(fbinX); 
    calc_weights_( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      bfweight_vec_[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  bfweight_vec_[unitX]->setTimeAlignment();

  return true;
}


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
//
SphericalSpatialHWNCBeamformer::SphericalSpatialHWNCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, float ratio, const String& nm ):
  SphericalHWNCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm ),
  SigmaSI_(NULL),dthreshold_(1.0E-8)
{
  unsigned fftLen2 = fftLen/2;

  SigmaSI_ = new gsl_matrix_complex*[fftLen2];
  for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++)
    SigmaSI_[fbinX] = NULL;
}

SphericalSpatialHWNCBeamformer::~SphericalSpatialHWNCBeamformer()
{
  unsigned fftLen2 = fftLen_/2;
    
  for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++)
    if( NULL != SigmaSI_[fbinX] ){
      gsl_matrix_complex_free( SigmaSI_[fbinX] );
    }

  delete [] SigmaSI_;
}

/*
  @brief compute the coherence matrix for the diffuse noise field
  @param unsigned fbinX[in]
*/
gsl_matrix_complex *SphericalSpatialHWNCBeamformer::calc_diffuse_noise_model_( unsigned fbinX )
{
  int nChan = (int)chanN();
  gsl_matrix_complex *A = gsl_matrix_complex_alloc( nChan, dim_ );
  gsl_matrix_complex *SigmaSIp = gsl_matrix_complex_calloc( dim_, dim_ );
  gsl_matrix_complex *A_SigmaSIp = gsl_matrix_complex_calloc( nChan, dim_ );

  if( NULL == SigmaSI_[fbinX] ){
    SigmaSI_[fbinX] = gsl_matrix_complex_calloc( nChan, nChan );
  }

  /* set the left-side matrix, A */
  /* Note: this matrix A is correct! Eq. (180) in the book chapter is incorrect! */
  for(int chanX=0;chanX<nChan;chanX++){
    for(int n=0,idx=0;n<maxOrder_;n++){/* order */
      for(int m=-n;m<=n;m++){/* degree */
	gsl_complex YmnA_s = gsl_vector_complex_get( sh_s_[idx], chanX );
	gsl_matrix_complex_set( A, chanX, idx, gsl_complex_conjugate(YmnA_s) );
	idx++;
      }
    }
  }

  /* compute the covariance matrix in the spherical harmonics domain, SigmaSIp */
  for(int n=0,idx=0;n<maxOrder_;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn );
    
    //if( (n%2)!=0 ) bn2 = -bn2;
    for( int m=-n;m<=n;m++){/* degree */
      gsl_matrix_complex_set( SigmaSIp, idx, idx, gsl_complex_rect( bn2, 0.0 ) );
      idx++;
    }
  }

  /* compute the covariance matrix SigmaSI = A * SigmaSIp * A^H */
  gsl_blas_zgemm( CblasNoTrans, CblasNoTrans,   gsl_complex_rect(1,0), A, SigmaSIp, gsl_complex_rect(0,0), A_SigmaSIp );
  gsl_blas_zgemm( CblasNoTrans, CblasConjTrans, gsl_complex_rect(1,0), A_SigmaSIp, A, gsl_complex_rect(0,0), SigmaSI_[fbinX] );

  gsl_matrix_complex_free( A );
  gsl_matrix_complex_free( SigmaSIp );
  gsl_matrix_complex_free( A_SigmaSIp );

  // add the diagonal component
  for(unsigned chanX=0;chanX<nChan;chanX++){
    gsl_matrix_complex_set( SigmaSI_[fbinX], chanX, chanX, gsl_complex_add_real( gsl_matrix_complex_get( SigmaSI_[fbinX], chanX, chanX ), sigma2_ ) );
  }

  return SigmaSI_[fbinX];
}

void SphericalSpatialHWNCBeamformer::calc_weights_( unsigned fbinX, gsl_vector_complex *weights )
{
  int nChan = chanN();

  for ( int s = 0; s < nChan; s++ ) { /* channnel */
    /* compute the approximation of the sound pressure at sensor s with the spherical harmonics coefficients, */
    /* G(Omega_s,ka,Omega) = 4pi \sum_{n=0}^{N} i^n b_n(ka) \sum_{m=-n}^{n} Ymn(Omega_s) Ymn(Omega)^*         */
    gsl_complex weight = gsl_complex_rect( 0, 0 );
    for ( int n = 0, idx = 0; n < maxOrder_; n++ ) { /* order */
      gsl_complex bn = gsl_matrix_complex_get( mode_mplitudes_, fbinX, n ); // bn = modeAmplitude( order, ka );
      gsl_complex in, inbn;
      gsl_complex tmp_weight;

      if ( 0 == ( n % 4 ) ) {
	in = gsl_complex_rect( 1, 0 );
      }
      else if ( 1 == ( n % 4 ) ) {
	in = gsl_complex_rect( 0, 1 );
      }
      else if ( 2 == ( n % 4 ) ) {
	in = gsl_complex_rect( -1, 0 );
      }
      else {
        in = gsl_complex_rect( 0, -1 );
      }
      inbn = gsl_complex_mul( in, bn );

      tmp_weight = gsl_complex_rect( 0, 0 );
      for ( int m = -n; m <= n; m++ ) { /* degree */
        tmp_weight = gsl_complex_add( tmp_weight,
                                      gsl_complex_mul( gsl_complex_conjugate( gsl_vector_complex_get( sh_s_[idx], s ) ), /* this has to be conjugated to get Ymn */
                                                       gsl_complex_conjugate( sphericalHarmonic( m, n, theta_, phi_ ) ) ) );
        idx++;
      }
      weight = gsl_complex_add( weight, gsl_complex_mul( inbn, tmp_weight ) );
    }
    //gsl_vector_complex_set( weights, s, gsl_complex_mul_real( weight, 4 * M_PI ) );
  }

  {// compute the coherence matrix of the diffuse noise field; the result is set to SigmaSI_.
    calc_diffuse_noise_model_( fbinX );

    gsl_matrix_complex *iSigmaSI   = gsl_matrix_complex_calloc( nChan, nChan );
    gsl_vector_complex *iSigmaSI_v = gsl_vector_complex_calloc( nChan );
    gsl_complex lambda, ilambda;
    double norm = gsl_blas_dznrm2( weights );

    gsl_blas_zdscal( 1/norm, weights );

    if ( false == pseudoinverse( SigmaSI_[fbinX], iSigmaSI, dthreshold_ ) )
      fprintf( stderr, "fbinX %d : pseudoinverse() failed\n", fbinX );

    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( 1.0, 0.0 ), iSigmaSI, weights,
                    gsl_complex_rect( 0.0, 0.0), iSigmaSI_v ); // SigmaSI^{-1} v
    gsl_blas_zdotc( weights, iSigmaSI_v, &lambda ); // lambda = vH * SigmaSI^{-1} * v
    ilambda = gsl_complex_inverse( lambda ); // ilambda = 1 / lambda
    gsl_blas_zscal( ilambda, iSigmaSI_v );   // iSigmaSI_v = ilambda * iSigmaSI_v
    gsl_vector_complex_memcpy( weights, iSigmaSI_v );

    gsl_matrix_complex_free( iSigmaSI );
    gsl_vector_complex_free( iSigmaSI_v );
  }
  {
    if( ratio_ > 0.0 ){
      // control the white noise gain
      if( NULL == WNG_ ){ calc_wng();}
      double wng = gsl_vector_get( WNG_, fbinX );
      normalize_weights_( weights, 2 * sqrt( M_PI / ( nChan * wng) ) );
    }
    else{
      double coeff = ( 16 * M_PI * M_PI ) / ( nChan * maxOrder_ * maxOrder_ );
      gsl_blas_zdscal( coeff, weights );
    }
  }

  return;
}

const gsl_vector_complex* SphericalSpatialHWNCBeamformer::next( int frame_no )
{
  if ( frame_no == frame_no_ ) return vector_;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->alloc_image_();
  for ( ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++ ) {
    const gsl_vector_complex* samp = ( *itr )->next( frame_no );
    if ( true == ( *itr )->is_end() ) is_end_ = true;
    snapshot_array_->set_samples( samp, chanX );
    chanX++;
  }
  snapshot_array_->update();

  if ( halfBandShift_ == false ) {
    // calculate a direct component.

    XVec    = snapshot_array_->snapshot(0);
    weights = bfweight_vec_[0]->wq_f(0);
    gsl_blas_zdotc( weights, XVec, &val );
    gsl_vector_complex_set(vector_, 0, val );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for ( unsigned fbinX = 1; fbinX <= fftLen2_; fbinX++ ) {
      XVec    = snapshot_array_->snapshot( fbinX );
      weights = bfweight_vec_[0]->wq_f( fbinX );

      gsl_blas_zdotc( weights, XVec, &val );

      if( fbinX < fftLen2_ ){
        gsl_vector_complex_set(vector_, fbinX, val );
        gsl_vector_complex_set(vector_, fftLen_ - fbinX, gsl_complex_conjugate(val) );
      }
      else
        gsl_vector_complex_set(vector_, fftLen2_, val );
    }
  }
  else{
    throw j_error( "halfBandShift_ == true is not implemented yet\n" );
  }

  increment_();
  return vector_;
}

bool SphericalSpatialHWNCBeamformer::alloc_steering_unit_( int unitN )
{
  for ( unsigned unitX = 0; unitX < bfweight_vec_.size(); unitX++ ) {
    if ( NULL != bfweight_vec_[unitX] ) {
      delete bfweight_vec_[unitX];
    }
    bfweight_vec_[unitX] = NULL;
  }

  if ( bfweight_vec_.size() != unitN )
    bfweight_vec_.resize( unitN );

  for ( unsigned unitX = 0; unitX < unitN; unitX++ ) {
    bfweight_vec_[unitX] = new BeamformerWeights( fftLen_, (int)chanN(), halfBandShift_, NC_ );
  }

  return true;
}

bool SphericalSpatialHWNCBeamformer::calc_steering_unit_( int unitX, bool isGSC )
{
  gsl_vector_complex* weights;

  if( unitX >= bfweight_vec_.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  for ( unsigned fbinX = 0; fbinX <= fftLen2_; fbinX++ ) {
    weights = bfweight_vec_[unitX]->wq_f(fbinX);
    calc_weights_( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      bfweight_vec_[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  bfweight_vec_[unitX]->setTimeAlignment();

  return true;
}
