/*
 * @file tracker.cc
 * @brief Beamforming and speaker tracking with a spherical microphone array.
 * @author John McDonough
 */
#include "beamformer/tracker.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <algorithm>


// ----- methods for class `BaseDecomposition::SubbandEntry' -----
//
BaseDecomposition::SubbandEntry::SubbandEntry(unsigned subbandX, const gsl_complex& Bkl)
  : subbandX_(subbandX), bkl_(Bkl)
{
}

MemoryManager<BaseDecomposition::SubbandEntry>& BaseDecomposition::SubbandEntry::memoryManager()
{
  static MemoryManager<SubbandEntry> _MemoryManager("SubbandEntry::memoryManager");
  return _MemoryManager;
}


// ----- methods for class `BaseDecomposition::SubbandEntry' -----
//
BaseDecomposition::SubbandList::SubbandList(const gsl_vector_complex* bkl, unsigned useSubbandsN)
  : subbandsN_(bkl->size), useSubbandsN_(useSubbandsN == 0 ? subbandsN_ : useSubbandsN), subbands_(new SubbandEntry*[subbandsN_])
{
  // create and sort a new list of subband entries
  for (unsigned subX = 0; subX < subbandsN_; subX++)
    subbands_[subX] = new SubbandEntry(subX, gsl_vector_complex_get(bkl, subX));

  sort(subbands_, subbands_ + subbandsN_, GreaterThan());

  /*
  for (unsigned subX = 0; subX < useSubbandsN_; subX++) {
    const SubbandEntry& subbandEntry(*(subbands_[subX]));
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();
    printf("Subband %4d : Magnitude %8.2f\n", subbandX, gsl_complex_abs(Bkl));
  }
  fflush(stdout);
  */
}

BaseDecomposition::SubbandList::~SubbandList()
{
  // cout << "Deallocating 'SubbandList'." << endl;

  for (unsigned subX = 0; subX < subbandsN_; subX++)
    delete subbands_[subX];
  delete[] subbands_;
}


// ----- methods for class `BaseDecomposition' -----
//
const unsigned    BaseDecomposition::StateN_		= 2;
const unsigned    BaseDecomposition::ChannelsN_		= 32;
const double      BaseDecomposition::SpeedOfSound_	= 343740.0;
const gsl_complex BaseDecomposition::ComplexZero_	= gsl_complex_rect(0.0, 0.0);
const gsl_complex BaseDecomposition::ComplexOne_	= gsl_complex_rect(1.0, 0.0);

BaseDecomposition::BaseDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN, bool spatial)
  : orderN_(orderN), modesN_((orderN_ + 1) * (orderN_ + 1)), subbandsN_(subbandsN), subbandsN2_(subbandsN / 2),
    useSubbandsN_((useSubbandsN == 0) ? (subbandsN2_ + 1) : useSubbandsN), subbandLengthN_(spatial ? ChannelsN_ : modesN_),
    samplerate_(sampleRate),
    a_(a),
    bn_(new gsl_vector_complex*[subbandsN2_ + 1]),
    theta_s_(gsl_vector_calloc(ChannelsN_)),
    phi_s_(gsl_vector_calloc(ChannelsN_)),
    spherical_component_(new gsl_vector_complex*[modesN_]),
    bkl_(gsl_vector_complex_calloc(subbandsN2_ + 1)),
    dbkl_dtheta_(gsl_vector_complex_calloc(subbandsN2_ + 1)),
    dbkl_dphi_(gsl_vector_complex_calloc(subbandsN2_ + 1)),
    gkl_(new gsl_vector_complex*[subbandsN2_ + 1]),
    dgkl_dtheta_(new gsl_vector_complex*[subbandsN2_ + 1]),
    dgkl_dphi_(new gsl_vector_complex*[subbandsN2_ + 1]),
    vkl_(gsl_vector_complex_calloc(subbandLengthN_)),
    Hbar_k_(gsl_matrix_complex_calloc(useSubbandsN_ * subbandLengthN_, StateN_)),
    yhat_k_(gsl_vector_complex_calloc(useSubbandsN_ * subbandLengthN_)),
    subbandList_(NULL)
{
  set_eigenmike_geometry_();

  // bn_ contains the factors 4 \pi i^n b_n(ka)
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    double ka = 2.0 * M_PI * subbandX * a_ * samplerate_ / (subbandsN_ * SpeedOfSound_);
    bn_[subbandX] = gsl_vector_complex_calloc(orderN_ + 1);
    for (int n = 0; n <= orderN_; n++) {
      gsl_complex in = calc_in_(n);
      const gsl_complex factor = gsl_complex_mul(gsl_complex_mul(gsl_complex_rect(4.0 * M_PI, 0.0), in), modal_coefficient(n, ka));
      gsl_vector_complex_set(bn_[subbandX], n, factor);
    }
  }

  // spherical_component_[idx] contains the spherical components Y^m_n(\theta_s, \phi_s) for each channel s
  unsigned idx = 0;
  for (int n = 0; n <= orderN_; n++) {
    for (int m = -n; m <= n; m++) {
      spherical_component_[idx] = gsl_vector_complex_calloc(ChannelsN_);
      for (unsigned chanX = 0; chanX < ChannelsN_; chanX++) {
	double theta_s = gsl_vector_get(theta_s_, chanX);
	double phi_s   = gsl_vector_get(phi_s_, chanX);
	gsl_complex Ynm = gsl_complex_conjugate(harmonic(n, m, theta_s, phi_s));
	gsl_vector_complex_set(spherical_component_[idx], chanX, Ynm);
      }
      idx++;
    }
  }

  // work space for storing $\mathbf{g}_{k,l}(\theta, \phi)$ and its derivatives w.r.t. $\theta$ and $\phi$
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    gkl_[subbandX] = gsl_vector_complex_calloc(subbandLengthN_);
    dgkl_dtheta_[subbandX] = gsl_vector_complex_calloc(subbandLengthN_);
    dgkl_dphi_[subbandX] = gsl_vector_complex_calloc(subbandLengthN_);
  }
}

BaseDecomposition::~BaseDecomposition()
{
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++)
    gsl_vector_complex_free(bn_[subbandX]);
  delete[] bn_;

  gsl_vector_free(theta_s_);
  gsl_vector_free(phi_s_);

  unsigned idx = 0;
  for (int n = 0; n <= orderN_; n++) {
    for (int m = -n; m <= n; m++) {
      gsl_vector_complex_free(spherical_component_[idx]);
      idx++;
    }
  }

  gsl_vector_complex_free(bkl_);
  gsl_vector_complex_free(dbkl_dtheta_);
  gsl_vector_complex_free(dbkl_dphi_);
  gsl_vector_complex_free(vkl_);
  gsl_matrix_complex_free(Hbar_k_);
  gsl_vector_complex_free(yhat_k_);

  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    gsl_vector_complex_free(gkl_[subbandX]);
    gsl_vector_complex_free(dgkl_dtheta_[subbandX]);
    gsl_vector_complex_free(dgkl_dphi_[subbandX]);
  }
  delete[] gkl_;  delete[] dgkl_dtheta_;  delete[] dgkl_dphi_;
}

gsl_complex BaseDecomposition::modal_coefficient(unsigned order, unsigned subbandX) const
{
  return gsl_vector_complex_get(bn_[subbandX], order);
}

gsl_complex BaseDecomposition::calc_Gnm_(unsigned subbandX, int n, int m, double theta, double phi)
{
  gsl_complex Gnm = gsl_complex_mul(gsl_vector_complex_get(bn_[subbandX], n), harmonic(n, m, theta, phi));

  return Gnm;
}

gsl_complex BaseDecomposition::calc_dGnm_dtheta_(unsigned subbandX, int n, int m, double theta, double phi)
{
  gsl_complex Gnm;

  return Gnm;
}


/**
   @brief set the geometry of the EigenMikeR
*/
void BaseDecomposition::set_eigenmike_geometry_()
{ 
   gsl_vector_set(theta_s_, 0, 69.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   0, 0.0);

   gsl_vector_set(theta_s_, 1, 90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   1, 32.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 2, 111.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   2, 0.0);

   gsl_vector_set(theta_s_, 3,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   3, 328.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 4, 32.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   4, 0.0);

   gsl_vector_set(theta_s_, 5, 55.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   5, 45.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 6, 90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   6, 69.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 7, 125.0 * M_PI / 180);
   gsl_vector_set(phi_s_,   7, 45.0  * M_PI / 180);

   gsl_vector_set(theta_s_, 8, 148.0 * M_PI / 180);
   gsl_vector_set(phi_s_,   8, 0.0);

   gsl_vector_set(theta_s_, 9, 125.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   9, 315.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 10,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   10, 291.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 11,  55.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   11, 315.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 12, 21.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   12, 91.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 13, 58.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   13, 90.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 14, 121.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   14,  90.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 15, 159.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   15,  89.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 16,  69.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   16, 180.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 17,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   17, 212.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 18, 111.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   18, 180.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 19,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   19, 148.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 20,  32.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   20, 180.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 21,  55.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   21, 225.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 22,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   22, 249.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 23, 125.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   23, 225.0 * M_PI / 180.0);
   
   gsl_vector_set(theta_s_, 24, 148.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   24, 180.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 25, 125.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   25, 135.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 26,  90.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   26, 111.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 27,  55.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   27, 135.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 28,  21.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   28, 269.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 29,  58.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   29, 270.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 30, 122.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   30, 270.0 * M_PI / 180.0);

   gsl_vector_set(theta_s_, 31, 159.0 * M_PI / 180.0);
   gsl_vector_set(phi_s_,   31, 271.0 * M_PI / 180.0);

   /*
   for (unsigned s = 0; s < 32; s++)
     fprintf(stderr,"%02d : %6.2f %6.2f\n", s, gsl_vector_get(theta_s_, s),gsl_vector_get(phi_s_,s));
   */
}

gsl_complex BaseDecomposition::calc_in_(int n)
{
  int modulo = n % 4;
  switch (modulo) {
  case 0:
    return gsl_complex_rect(1.0, 0.0);
  case 1:
    return gsl_complex_rect(0.0, 1.0);
  case 2:
    return gsl_complex_rect(-1.0, 0.0);
  case 3:
    return gsl_complex_rect(0.0, -1.0);
  }
  throw jtype_error("BaseDecomposition::calc_in_(): Invalid arg %d\n", n);
}

void BaseDecomposition::reset() { }

#if 1

// calculate \bar{Y}^m_n(\theta, \phi)
gsl_complex BaseDecomposition::harmonic(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);

  int status;
  gsl_sf_result sphPnm;
  if (degree >= 0) {
    // \sqrt{(2n+1)/(4\pi)} \sqrt{(n-m)!/(n+m)!} P_n^m(x), and derivatives, m >= 0, n >= m, |x| <= 1
    status = gsl_sf_legendre_sphPlm_e(order /* =n */,  degree /* =m */, cos(theta), &sphPnm);
  } else {
    status = gsl_sf_legendre_sphPlm_e(order /* =n */, -degree /* =m */, cos(theta), &sphPnm);
    if(((-degree) % 2) == 1)
      sphPnm.val = -sphPnm.val;
  }

  gsl_complex Ynm = gsl_complex_mul_real(gsl_complex_polar(1.0, -degree*phi), sphPnm.val);
  //fprintf(stderr,"%e %e \n", sphPnm.val, sphPnm.err); 

  return Ynm;
}

#else

// calculate \bar{Y}^m_n(\theta, \phi)
gsl_complex BaseDecomposition::harmonic(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);

  gsl_sf_result sphPnm;
  int deg	  = abs(degree);
  // \sqrt{(2n+1)/(4\pi)} \sqrt{(n-m)!/(n+m)!} P_n^m(x), and derivatives, m >= 0, n >= m, |x| <= 1
  int status	  = gsl_sf_legendre_sphPlm_e(order /* =n */,  deg /* =m */, cos(theta), &sphPnm);
  gsl_complex Ynm = gsl_complex_mul_real(gsl_complex_polar(1.0, -deg * phi), sphPnm.val);
  if (degree < 0) {
    // gsl_complex Ynm = gsl_complex_conjugate(Ynm);
    if (((-degree) % 2) == 1)
      Ynm = gsl_complex_mul_real(Ynm, -1.0);
  }

  return Ynm;
}

#endif

// retrieve previously stored spherical harmonic for 'channelX'
gsl_complex BaseDecomposition::harmonic(int order, int degree, unsigned channelX) const
{
  unsigned idx = 0;
  for (int n = 0; n <= orderN_; n++) {
    for (int m = -n; m <= n; m++) {
      if (n == order && m == degree)
        return gsl_vector_complex_get(spherical_component_[idx], channelX);
      idx++;
    }
  }
  throw jtype_error("BaseDecomposition::harmonic(): Invalid args %d %d\n", order, degree);
}

#if 1

// calculate the normalization factor \sqrt{(4\pi)/(2n+1)} \sqrt{(n-m)!/(n+m)!}
double BaseDecomposition::calc_normalization_(int order, int degree)
{
  double norm	= sqrt((2 * order + 1) / (4.0 * M_PI));
  double factor = 1.0;

  if (degree >= 0) {
    int m = degree;
    while (m > -degree) {
      factor *= (order + m);
      m--;
    }
    norm /= sqrt(factor);
  } else {
    int m = -degree;
    while (m > degree) {
      factor *= (order + m);
      m--;
    }
    norm *= sqrt(factor);
  }

  return norm;
}

#else

// calculate the normalization factor \sqrt{(4\pi)/(2n+1)} \sqrt{(n-m)!/(n+m)!}
double BaseDecomposition::calc_normalization_(int order, int degree)
{
  double norm	= sqrt((2 * order + 1) / (4.0 * M_PI));
  double factor = sqrt(tgamma(order - abs(degree) + 1.0) / tgamma(order + abs(degree) + 1.0));

  if ((degree < 0) && ((-degree) % 2 == 1))
    norm *= -1;

  return norm * factor;
}

#endif

double BaseDecomposition::calc_Pnm_(int order, int degree, double theta)
{
  double result = gsl_sf_legendre_Plm(order, abs(degree), cos(theta));
  if (degree < 0) {
    int m = -degree;
    double factor = 1.0;
    while (m > degree) {
      factor *= (order + m);
      m--;
    }
    result /= factor;
    if (((-degree) % 2) == 1)
      result *= -1;
  }

  return result;
}

// calculate $d P^m_n(\cos\theta) / d \theta$
double BaseDecomposition::calc_dPnm_dtheta_(int order, int degree, double theta) 
{
  // still need to test for the case |theta| = 1
  double cosTheta  = cos(theta);
  double cosTheta2 = cosTheta * cosTheta;
  double dPnm_dx   = ((degree - order - 1) * calc_Pnm_(order + 1, degree, theta)
		      + (order + 1) * cosTheta * calc_Pnm_(order, degree, theta)) / (1.0 - cosTheta2);

  return dPnm_dx;
}

// calculate \partial \bar{Y}^m_n(\theta, \phi) / \partial \theta
gsl_complex BaseDecomposition::harmonic_deriv_polar_angle(int order, int degree, double theta, double phi)
{
  double      dPnm_dx	= calc_dPnm_dtheta_(order, degree, theta);
  double      norm	= calc_normalization_(order, degree);
  double      factor	= -norm * dPnm_dx * sin(theta);
  gsl_complex result	= gsl_complex_mul_real(gsl_complex_polar(1.0, -degree * phi), factor);

  return result;
}

// calculate \partial \bar{Y}^m_n(\theta, \phi) / \partial \phi
gsl_complex BaseDecomposition::harmonic_deriv_azimuth(int order, int degree, double theta, double phi)
{
  if (order < degree || order < -degree)
    fprintf(stderr, "The order must be less than the degree but %d > %dn", order, degree);
  
  gsl_complex Ynm	= harmonic(order, degree, theta, phi);
  gsl_complex dYmn_dphi	= gsl_complex_mul_real(gsl_complex_mul(Ynm, gsl_complex_rect(0.0, -1.0)), degree);

  return dYmn_dphi;
}

gsl_complex BaseDecomposition::modal_coefficient(unsigned order, double ka)
{
  if(ka == 0.0)
    return gsl_complex_rect(1.0, 0.0);
  
  gsl_complex bn;
  switch (order) {
  case 0:
    {
      double      j0 = gsl_sf_sinc(ka / M_PI);
      gsl_complex h0 = gsl_complex_rect(j0, - cos(ka) / ka);
      double      val1 = ka * cos(ka) - sin(ka);
      gsl_complex val2 = gsl_complex_mul(gsl_complex_rect(ka, 1), gsl_complex_polar(1, ka));
      gsl_complex grad = gsl_complex_div(gsl_complex_rect(val1, 0), val2);
      bn = gsl_complex_sub(gsl_complex_rect(j0, 0), gsl_complex_mul(grad, h0));
      
      // bn = ( i * cos(ka) + sin(x) ) ./ ( i + ka );
      // bn = gsl_complex_div( gsl_complex_rect( sin(ka), cos(ka) ), gsl_complex_rect( ka, 1.0 ) );
    }
    break;
  case 1:
    {
      // bn = x .* ( - cos(x) + i * sin(x) ) ./ (-2 + 2 * i * x + x.^2);
      double ka2 = ka * ka;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(-cos(ka), sin(ka)),
						gsl_complex_rect(ka2 - 2, 2 * ka)), ka);
      //printf("E %e %e\n",ka, gsl_complex_abs(bn));
    }
    break;
  case 2:
    {
      // bn = i * x.^2 .* (cos(x) - i * sin(x)) ./ (-9*i - 9*x + 4*i*x.^2 + x.^3);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      bn = gsl_complex_mul_imag(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka)),
						gsl_complex_rect(ka3 - 9*ka, 4*ka2 - 9)), ka2);
    }
    break;
  case 3:
    {
      // bn = x.^3 .* (cos(x) - i * sin(x)) ./ (60 - 60 * i * x - 27 * x.^2 + 7 * i * x.^3 + x.^4);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka) ),
						gsl_complex_rect(ka4 - 27 * ka2 + 60, 7 * ka3 - 60 * ka)), ka3);
    }
    break;
  case 4:
    {
      //  bn = x.^4 .* (i * cos(x) + sin(x)) ./ (525*i + 525*x - 240 * i * x.^2 - 65 * x.^3 + 11 * i * x.^4 + x.^5);
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(sin(ka), cos(ka)),
						gsl_complex_rect(ka5 - 65 * ka3 + 525 * ka, 11 * ka4 - 240 * ka2 + 525)), ka4);
    }
    break;
  case 5:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      // bn = x.^5 .* (cos(x) - i * sin(x)) ./ (-5670 + 5670 * i * x + 2625 * x.^2 - 735 * i * x.^3 - 135 * x.^4 + 16 * i * x.^5 + x.^6);
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(cos(ka), -sin(ka) ),
						gsl_complex_rect(ka6 - 135 * ka4 + 2625 * ka2 - 5670,
								 16 * ka5 - 735 * ka3 + 5670 * ka)), ka5);
    }
    break;
  case 6:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      //  bn = i*x.^6 .* (cos(x) - i * sin(x)) ./ (-72765 * i - 72765 * x + 34020 * i * x.^2 + 9765 * x.^3 - 1890 * i * x.^4 - 252 * x.^5 + 22 * i * x.^6 + x.^7);
      bn = gsl_complex_mul_imag( gsl_complex_div( gsl_complex_rect(cos(ka), -sin(ka)),
						  gsl_complex_rect(ka7 - 252 * ka5 + 9765 * ka3 - 72765 * ka ,
								   22 * ka6 - 1890 * ka4 + 34020 * ka2 -72765)), ka6);
    }
    break;
  case 7:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      double ka8 = ka7 * ka;
      //bn = x.^7 .* (cos(x) - i * sin(x)) ./ (1081080 - 1081080 * i * x - 509355 * x.^2 + 148995 * i * x.^3 + 29925 * x.^4 + -4284 * i * x.^5 - 434 * x.^6 + 29 * i * x.^7 + x.^8);
      bn = gsl_complex_mul_real( gsl_complex_div( gsl_complex_rect( cos(ka), -sin(ka) ),
						  gsl_complex_rect( 1081080- 509355 * ka2 + 29925 * ka4 - 434 * ka6 + ka8, - 1081080 * ka + 148995 * ka3 - 4284 * ka5 + 29 * ka7)), ka7);
    }
    break;
  case 8:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double ka6 = ka5 * ka;
      double ka7 = ka6 * ka;
      double ka8 = ka7 * ka;
      double ka9 = ka8 * ka;
      //bn = x.^8 .* (i * cos(x) + sin(x)) ./ (18243225 * i + 18243225 * x - 8648640 * i * x.^2 - 2567565 * x.^3 + 530145 * i * x.^4 + 79695 * x.^5 - 8820 * i * x.^6 - 702 * x.^7 + 37 * i * x.^8 + x.^9);
      bn = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(sin(ka), cos(ka)),
						gsl_complex_rect(18243225 * ka - 2567565 * ka3 + 79695 * ka5 - 702 * ka7 + ka9,
								 18243225 - 8648640 * ka2 + 530145 * ka4 - 8820 * ka6 + 37 * ka8)), ka8);
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
      gsl_complex val, grad;
      
      status = gsl_sf_bessel_jl_e( order, ka, &jn);// the (regular) spherical Bessel function of the first kind
      status = gsl_sf_bessel_yl_e( order, ka, &yn);// the (irregular) spherical Bessel function of the second kind
      hn = gsl_complex_rect(jn.val, yn.val); // Spherical Hankel function of the first kind
      
      status = gsl_sf_bessel_jl_e( order-1, ka, &jn_p );
      status = gsl_sf_bessel_jl_e( order+1, ka, &jn_n );
      djn = ( jn_p.val - jn_n.val ) / 2;

      status = gsl_sf_bessel_yl_e( order-1, ka, &yn_p );
      status = gsl_sf_bessel_yl_e( order+1, ka, &yn_n );
      dyn = ( yn_p.val - yn_n.val ) / 2;

      hn_p = gsl_complex_rect( jn_p.val, yn_p.val );
      hn_n = gsl_complex_rect( jn_n.val, yn_n.val );

      val = gsl_complex_div_real( gsl_complex_add( hn, gsl_complex_mul_real( hn_n, ka ) ), ka );
      dhn = gsl_complex_mul_real( gsl_complex_sub( hn_p, val ), 0.5 );

      grad = gsl_complex_div( gsl_complex_rect( djn, 0 ), dhn );
      bn   = gsl_complex_add_real( gsl_complex_negative( gsl_complex_mul( grad, hn ) ), jn.val );
    }
    break;
  }

  return bn;
}


// ----- methods for class `ModalDecomposition' -----
//
ModalDecomposition::ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN)
  : BaseDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN, /* spatial= */ false) { }

void ModalDecomposition::estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX)
{
  calculate_gkl(theta, phi, subbandX);
  transform(snapshot, vkl_);
  gsl_complex eta, del;
  gsl_blas_zdotc(gkl_[subbandX], vkl_, &eta);
  gsl_blas_zdotc(gkl_[subbandX], gkl_[subbandX], &del);

  // calculate $B_{k,l}$
  double 	delta		= GSL_REAL(del);
  gsl_complex	Bkl		= gsl_complex_div_real(eta, delta);
  gsl_vector_complex_set(bkl_, subbandX, Bkl);

  // calculate $\partial B_{k,l} / \partial \theta$
  gsl_complex deta_dtheta;
  gsl_blas_zdotc(dgkl_dtheta_[subbandX], vkl_, &deta_dtheta);

  gsl_complex deta_dphi;
  gsl_blas_zdotc(dgkl_dphi_[subbandX], vkl_, &deta_dphi);

  double ddelta_dtheta = 0.0;
  for (int n = 0; n <= orderN_; n++) {
    gsl_complex bn		= gsl_vector_complex_get(bn_[subbandX], n);
    for (int m = -n; m <= n; m++) {
      double norm2		= M_PI * calc_normalization_(n, m) * gsl_complex_abs(bn);
      double Pnm		= calc_Pnm_(n, m, theta);
      double dPnm_dtheta	= calc_dPnm_dtheta_(n, m, theta);

      double dG2nm_dtheta	= -32.0 * norm2 * norm2 * Pnm * dPnm_dtheta * sin(theta);
      ddelta_dtheta		+= dG2nm_dtheta;
    }
  }
  gsl_complex dBkl_dtheta	= gsl_complex_sub(gsl_complex_mul_real(deta_dtheta, delta), gsl_complex_mul_real(eta, ddelta_dtheta));
  dBkl_dtheta			= gsl_complex_div_real(dBkl_dtheta, delta * delta);
  gsl_vector_complex_set(dbkl_dtheta_, subbandX, dBkl_dtheta);

  gsl_complex dBkl_dphi		= gsl_complex_div_real(deta_dphi, delta);
  gsl_vector_complex_set(dbkl_dphi_, subbandX, dBkl_dphi);

  // printf("Estimated subband %d out of %d\n", subbandX, subbandsN2_);
  if (subbandX == subbandsN2_)
    subbandList_ = new SubbandList(bkl_, useSubbandsN_);
}

void ModalDecomposition::transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed)
{
  gsl_complex Fmn;
  unsigned idx = 0;
  for (int n = 0; n <= orderN_; n++) {				// order
    for (int m = -n ; m <= n; m++) {				// degree
      gsl_blas_zdotc(spherical_component_[idx], initial, &Fmn);	// Ynm^H X; see GSL documentation
      gsl_vector_complex_set(transformed, idx, Fmn);
      idx++;
    }
  }
}

const gsl_matrix_complex* ModalDecomposition::linearize(gsl_vector* xk, int frame_no)
{
  unsigned rowX = 0;

  for (Iterator itr(subbandList_); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();
    gsl_complex dBkl_dtheta	= gsl_vector_complex_get(dbkl_dtheta_, subbandX);
    gsl_complex dBkl_dphi	= gsl_vector_complex_get(dbkl_dphi_, subbandX);

    unsigned idx = 0;
    for (int n = 0; n <= orderN_; n++) {
      for (int m = -n; m <= n; m++) {
	gsl_complex dtheta	= gsl_complex_add(gsl_complex_mul(Bkl, gsl_vector_complex_get(dgkl_dtheta_[subbandX], idx)),
						  gsl_complex_mul(gsl_vector_complex_get(gkl_[subbandX], idx), dBkl_dtheta));
	gsl_matrix_complex_set(Hbar_k_, rowX, /* colX= */ 0, dtheta);

	gsl_complex dphi	= gsl_complex_add(gsl_complex_mul(Bkl, gsl_vector_complex_get(dgkl_dphi_[subbandX], idx)),
						  gsl_complex_mul(gsl_vector_complex_get(gkl_[subbandX], idx), dBkl_dphi));
	gsl_matrix_complex_set(Hbar_k_, rowX, /* colX= */ 1, dphi);
	idx++;  rowX++;
      }
    }
  }

  return Hbar_k_;
}

const gsl_vector_complex* ModalDecomposition::predicted_observation(gsl_vector* xk, int frame_no)
{
  double theta	= gsl_vector_get(xk, 0);
  double phi	= gsl_vector_get(xk, 1);

  unsigned rowX = 0;
  for (Iterator itr(subbandList_); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();

    for (int n = 0; n <= orderN_; n++) {
      for (int m = -n; m <= n; m++) {
	gsl_complex Gnm = calc_Gnm_(subbandX, n, m, theta, phi);
	gsl_complex Vkl = gsl_complex_mul(Gnm, Bkl);
	gsl_vector_complex_set(yhat_k_, rowX, Vkl);
	rowX++;
      }
    }
  }

  return yhat_k_;
}

// estimate vectors $g_{k,l}$, $\partial g_{k,l} / \partial \theta$, and, $\partial g_{k,l} / \partial \phi$
void ModalDecomposition::calculate_gkl(double theta, double phi, unsigned subbandX)
{
  unsigned idx = 0;
  for (int n = 0; n <= orderN_; n++) {
    for (int m = -n; m <= n; m++) {
      gsl_complex bn = gsl_vector_complex_get(bn_[subbandX], n);
      gsl_complex coefficient = gsl_complex_mul(bn, harmonic(n, m, theta, phi));
      gsl_vector_complex_set(gkl_[subbandX], idx, coefficient);

      gsl_complex d_dtheta = gsl_complex_mul(bn, harmonic_deriv_polar_angle(n, m, theta, phi));
      gsl_vector_complex_set(dgkl_dtheta_[subbandX], idx, d_dtheta);

      gsl_complex d_dphi = gsl_complex_mul(bn, harmonic_deriv_azimuth(n, m, theta, phi));
      gsl_vector_complex_set(dgkl_dphi_[subbandX], idx, d_dphi);
      idx++;
    }
  }
}


// ----- methods for class `SpatialDecomposition' -----
//
SpatialDecomposition::SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN)
  : BaseDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN, /* spatial= */ true) { }

void SpatialDecomposition::estimate_Bkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX)
{
  calculate_gkl(theta, phi, subbandX);
  gsl_vector_complex_memcpy(vkl_, snapshot);
  gsl_complex eta, del;
  gsl_blas_zdotc(gkl_[subbandX], vkl_, &eta);
  gsl_blas_zdotc(gkl_[subbandX], gkl_[subbandX], &del);

  // calculate $B_{k,l}$
  double 	delta		= GSL_REAL(del);
  gsl_complex	Bkl		= gsl_complex_div_real(eta, delta);
  gsl_vector_complex_set(bkl_, subbandX, Bkl);

  // printf("Estimated subband %d out of %d\n", subbandX, subbandsN2_);
  if (subbandX == subbandsN2_)
    subbandList_ = new SubbandList(bkl_, useSubbandsN_);
}

const gsl_matrix_complex* SpatialDecomposition::linearize(gsl_vector* xk, int frame_no)
{
  unsigned rowX = 0;
  for (Iterator itr(subbandList_); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned subbandX		= subbandEntry.subbandX();

    for (unsigned s = 0; s < ChannelsN_; s++) {
      gsl_complex dtheta	= gsl_complex_mul(Bkl, gsl_vector_complex_get(dgkl_dtheta_[subbandX], s));
      gsl_matrix_complex_set(Hbar_k_, rowX, /* colX= */ 0, dtheta);

      gsl_complex dphi		= gsl_complex_mul(Bkl, gsl_vector_complex_get(dgkl_dphi_[subbandX], s));
      gsl_matrix_complex_set(Hbar_k_, rowX, /* colX= */ 1, dphi);
      rowX++;
    }
  }

  return Hbar_k_;
}

const gsl_vector_complex* SpatialDecomposition::predicted_observation(gsl_vector* xk, int frame_no)
{
  unsigned rowX = 0;
  for (Iterator itr(subbandList_); itr.more(); itr++) {
    const SubbandEntry& subbandEntry(*itr);
    gsl_complex Bkl		= subbandEntry.bkl();
    unsigned    subbandX	= subbandEntry.subbandX();

    for (unsigned s = 0; s < ChannelsN_; s++) {
      gsl_complex gkl		= gsl_complex_mul(gsl_vector_complex_get(gkl_[subbandX], s), Bkl);
      gsl_vector_complex_set(yhat_k_, rowX, gkl);
      rowX++;
    }
  }

  return yhat_k_;
}

void SpatialDecomposition::calculate_gkl(double theta, double phi, unsigned subbandX)
{
  for (unsigned s = 0; s < ChannelsN_; s++) {
    double theta_s		= gsl_vector_get(theta_s_, s);
    double phi_s		= gsl_vector_get(phi_s_, s);
    gsl_complex sum_n		= ComplexZero_;
    gsl_complex dsum_n_dtheta	= ComplexZero_;
    gsl_complex dsum_n_dphi	= ComplexZero_;
    for (int n = 0; n <= orderN_; n++) {
      gsl_complex b_n 		= gsl_vector_complex_get(bn_[subbandX], n);
      gsl_complex sum_m		= ComplexZero_;
      gsl_complex dsum_m_dtheta	= ComplexZero_;
      gsl_complex dsum_m_dphi	= ComplexZero_;
      for (int m = -n; m <= n; m++) {
	gsl_complex Ynm_s		= gsl_complex_conjugate(harmonic(n, m, theta_s, phi_s));
	gsl_complex coef_m		= gsl_complex_mul(Ynm_s, harmonic(n, m, theta, phi));
	sum_m				= gsl_complex_add(sum_m, coef_m);

	gsl_complex dcoef_m_dtheta	= gsl_complex_mul(Ynm_s, harmonic_deriv_polar_angle(n, m, theta, phi));
	dsum_m_dtheta			= gsl_complex_add(dsum_m_dtheta, dcoef_m_dtheta);

	gsl_complex dcoef_m_dphi		= gsl_complex_mul(Ynm_s, harmonic_deriv_azimuth(n, m, theta, phi));
	dsum_m_dphi			= gsl_complex_add(dsum_m_dphi, dcoef_m_dphi);
      }
      sum_n				= gsl_complex_add(sum_n,        gsl_complex_mul(b_n, sum_m));
      dsum_n_dtheta			= gsl_complex_add(dsum_n_dtheta, gsl_complex_mul(b_n, dsum_m_dtheta));
      dsum_n_dphi			= gsl_complex_add(dsum_n_dphi,   gsl_complex_mul(b_n, dsum_m_dphi));
    }
    gsl_vector_complex_set(gkl_[subbandX],         s, sum_n);
    gsl_vector_complex_set(dgkl_dtheta_[subbandX], s, dsum_n_dtheta);
    gsl_vector_complex_set(dgkl_dphi_[subbandX],   s, dsum_n_dphi);
  }
}


// ----- methods for class `BaseSphericalArrayTracker' -----
//
const unsigned    BaseSphericalArrayTracker::StateN_		= 2;
const gsl_complex BaseSphericalArrayTracker::ComplexZero_	= gsl_complex_rect(0.0, 0.0);
const gsl_complex BaseSphericalArrayTracker::ComplexOne_	= gsl_complex_rect(1.0, 0.0);
const double      BaseSphericalArrayTracker::Epsilon_		= 0.01;
const double	  BaseSphericalArrayTracker::Tolerance_		= 1.0e-04;

BaseSphericalArrayTracker::
BaseSphericalArrayTracker(BaseDecompositionPtr& baseDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			  unsigned maxLocalN, const String& nm)
  : VectorFloatFeatureStream(StateN_, nm), firstFrame_(true),
    subbandsN_(baseDecomposition->subbandsN()), subbandsN2_(baseDecomposition->subbandsN2()),
    useSubbandsN_(baseDecomposition->useSubbandsN()),
    modesN_(baseDecomposition->modesN()), subbandLengthN_(baseDecomposition->subbandLengthN()),
    observationN_(baseDecomposition->useSubbandsN() * baseDecomposition->subbandLengthN()),
    maxLocalN_(maxLocalN), is_end_(false),
    sigma_init_(sqrt(sigma2_init)),
    base_decomposition_(baseDecomposition),
    U_(gsl_matrix_calloc(StateN_, StateN_)),
    V_(gsl_matrix_calloc(2 * (subbandsN2_ + 1) * subbandLengthN_, 2 * (subbandsN2_ + 1) * subbandLengthN_)),
    K_k_k1_(gsl_matrix_calloc(StateN_, StateN_)),
    prearray_(gsl_matrix_calloc((2 * observationN_ + StateN_), (2 * observationN_ + 2 * StateN_))),
    vk_(gsl_vector_complex_calloc(observationN_)),
    Hbar_k_(gsl_matrix_calloc(2 * observationN_, StateN_)), yhat_k_(gsl_vector_calloc(2 * observationN_)),
    correction_(gsl_vector_calloc(StateN_)),
    position_(gsl_vector_calloc(StateN_)), eta_i_(gsl_vector_calloc(StateN_)), delta_(gsl_vector_calloc(StateN_)),
    residual_(gsl_vector_complex_calloc(observationN_)), residual_real_(gsl_vector_calloc(2 * observationN_)),
    scratch_(gsl_vector_calloc(2 * observationN_))
{
  next_speaker();

  // initialize covariance matrices
  for (unsigned n = 0; n < StateN_; n++) {
    gsl_matrix_set(U_, n, n, sqrt(sigma2_u));
    gsl_matrix_set(K_k_k1_, n, n, sqrt(sigma_init_));
  }

  for (unsigned n = 0; n < (2 * (subbandsN2_ + 1) * subbandLengthN_); n++)
    gsl_matrix_set(V_, n, n, sqrt(sigma2_v));
}

void BaseSphericalArrayTracker::next_speaker()
{
  reset();

  firstFrame_ = true;

  // set initial position: theta = 0.5, phi = 0
  gsl_vector_set(position_, 0, 0.5);
  gsl_vector_set(position_, 1, 0.0);

  // re-initialize state estimation error covariance matrix
  gsl_matrix_set_zero(K_k_k1_);
  for (unsigned n = 0; n < StateN_; n++)
    gsl_matrix_set(K_k_k1_, n, n, sqrt(sigma_init_));
}

void BaseSphericalArrayTracker::set_initial_position(double theta, double phi)
{
  gsl_vector_set(position_, 0, theta);
  gsl_vector_set(position_, 1, phi);
}

BaseSphericalArrayTracker::~BaseSphericalArrayTracker()
{
  gsl_matrix_free(U_);
  gsl_matrix_free(V_);
  gsl_matrix_free(K_k_k1_);
  gsl_matrix_free(prearray_);
  gsl_vector_complex_free(vk_);
  gsl_matrix_free(Hbar_k_);
  gsl_vector_free(yhat_k_);
  gsl_vector_free(correction_);
  gsl_vector_free(position_);
  gsl_vector_free(eta_i_);
  gsl_vector_free(delta_);
  gsl_vector_complex_free(residual_);
  gsl_vector_free(residual_real_);
  gsl_vector_free(scratch_);
}

void BaseSphericalArrayTracker::set_channel(VectorComplexFeatureStreamPtr& chan)
{
  channelList_.push_back(chan);
}

// "realify" 'Vk' and place into 'V'
void BaseSphericalArrayTracker::set_V(const gsl_matrix_complex* Vk, unsigned subbandX)
{
  gsl_matrix_view  Vk_ = gsl_matrix_submatrix(V_, /* m1= */ 2 * subbandX * subbandLengthN_, /* n1= */ 2 * subbandX * subbandLengthN_,
					      2 * subbandLengthN_, 2 * subbandLengthN_);
  for (unsigned m = 0; m < subbandLengthN_; m++) {
    for (unsigned n = 0; n <= m; n++) {
      double c = GSL_REAL(gsl_matrix_complex_get(Vk, m, n));
      gsl_matrix_set(&Vk_.matrix, m, n, c);
      gsl_matrix_set(&Vk_.matrix, m + subbandLengthN_, n + subbandLengthN_, c);

      double s = GSL_IMAG(gsl_matrix_complex_get(Vk, m, n));
      gsl_matrix_set(&Vk_.matrix, m + subbandLengthN_, n,  s);
    }
  }
  gsl_linalg_cholesky_decomp(&Vk_.matrix);

  // zero out upper triangular portion of Cholesky decomposition
  for (unsigned m = 0; m < 2 * subbandLengthN_; m++)
    for (unsigned n = m + 1; n < 2 * subbandLengthN_; n++)
      gsl_matrix_set(&Vk_.matrix, m, n, 0.0);
}

// calculate squared-error with current state estimate
double BaseSphericalArrayTracker::calc_residual_()
{
  const gsl_vector_complex* yhat_k = base_decomposition_->predicted_observation(eta_i_, frame_no_ + 1);
  gsl_vector_complex_memcpy(residual_, vk_);
  
  gsl_vector_complex_sub(residual_, yhat_k);
  double residual = 0.0;
  for (unsigned observationX = 0; observationX < observationN_; observationX++)
    residual += gsl_complex_abs2(gsl_vector_complex_get(residual_, observationX));
  residual /= observationN_;

  return residual;
}
    
void BaseSphericalArrayTracker::copy_position_()
{
  for (unsigned positionX = 0; positionX < StateN_; positionX++)
    gsl_vector_float_set(vector_, positionX, gsl_vector_get(position_, positionX));
}

void BaseSphericalArrayTracker::printMatrix_(const gsl_matrix_complex* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      gsl_complex value = gsl_matrix_complex_get(mat, m, n);
      printf("%8.4f %8.4f  ", GSL_REAL(value), GSL_IMAG(value));
    }
    printf("\n");
  }
}

void BaseSphericalArrayTracker::printMatrix_(const gsl_matrix* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      double value = gsl_matrix_get(mat, m, n);
      printf("%8.4f ", value);
    }
    printf("\n");
  }
}

void BaseSphericalArrayTracker::printVector_(const gsl_vector_complex* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    gsl_complex value = gsl_vector_complex_get(vec, n);
    printf("%8.4f %8.4f\n", GSL_REAL(value), GSL_IMAG(value));
  }
}

void BaseSphericalArrayTracker::printVector_(const gsl_vector* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    double value = gsl_vector_get(vec, n);
    printf("%8.4f\n", value);
  }
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
double BaseSphericalArrayTracker::calc_givens_rotation_(double v1, double v2, double& c, double& s)
{
  double norm = sqrt(v1 * v1 + v2 * v2);

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = v1 / norm;
  // gsl_complex_div_real(v1, norm);

  s = v2 / norm;
  // gsl_complex_div_real(gsl_complex_conjugate(v2), norm);

  // return gsl_complex_rect(norm, 0.0);
  return norm;
}

// apply a previously calculated Givens rotation
void BaseSphericalArrayTracker::apply_givens_rotation_(double v1, double v2, double c, double s, double& v1p, double& v2p)
{
  v1p = c * v1 + s * v2;
  /*
    gsl_complex_add(gsl_complex_mul(gsl_complex_conjugate(c), v1),
		    gsl_complex_mul(s, v2));
  */

  v2p = c * v2 - s * v1;
  /*
    gsl_complex_sub(gsl_complex_mul(c, v2),
		    gsl_complex_mul(gsl_complex_conjugate(s), v1));
  */
}

void BaseSphericalArrayTracker::realify_(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k)
{
  for (unsigned subX = 0; subX < useSubbandsN_; subX++) {
    for (unsigned obsX = 0; obsX < subbandLengthN_; obsX++) {
      for (unsigned stateX = 0; stateX < BaseDecomposition::StateN_; stateX++) {
	gsl_matrix_set(Hbar_k_,
		        2 * subX      * subbandLengthN_ + obsX, stateX, GSL_REAL(gsl_matrix_complex_get(Hbar_k, subX * subbandLengthN_ + obsX, stateX)));
	gsl_matrix_set(Hbar_k_,
		       (2 * subX + 1) * subbandLengthN_ + obsX, stateX, GSL_IMAG(gsl_matrix_complex_get(Hbar_k, subX * subbandLengthN_ + obsX, stateX)));
      }
      gsl_vector_set(yhat_k_,  2 * subX      * subbandLengthN_ + obsX, GSL_REAL(gsl_vector_complex_get(yhat_k, subX * subbandLengthN_ + obsX)));
      gsl_vector_set(yhat_k_, (2 * subX + 1) * subbandLengthN_ + obsX, GSL_IMAG(gsl_vector_complex_get(yhat_k, subX * subbandLengthN_ + obsX)));
    }
  }
}

void BaseSphericalArrayTracker::realify_residual_()
{
  for (unsigned subX = 0; subX < useSubbandsN_; subX++) {
    for (unsigned obsX = 0; obsX < subbandLengthN_; obsX++) {
      gsl_vector_set(residual_real_,  2 * subX      * subbandLengthN_ + obsX, GSL_REAL(gsl_vector_complex_get(residual_, subX * subbandLengthN_ + obsX)));
      gsl_vector_set(residual_real_, (2 * subX + 1) * subbandLengthN_ + obsX, GSL_IMAG(gsl_vector_complex_get(residual_, subX * subbandLengthN_ + obsX)));
    }
  }
}

void BaseSphericalArrayTracker::update_(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k, const SubbandListPtr& subbandList)
{
  // copy matrix components into prearray
  gsl_matrix_set_zero(prearray_);
  
  // copy subband-dependent observation noise covariance matrices into prearray
  unsigned subX = 0;
  for (Iterator itr(subbandList); itr.more(); itr++) {
    const    SubbandEntry& subbandEntry(*itr);
    unsigned subbandX	= subbandEntry.subbandX();

    // cout << "subX = " << subX << " subbandX = " << subbandX << endl;

    gsl_matrix_view  Vk = gsl_matrix_submatrix(prearray_, /* m1= */ 2 * subX * subbandLengthN_, /* n1= */ 2 * subX * subbandLengthN_,
					       2 * subbandLengthN_, 2 * subbandLengthN_);
    gsl_matrix_view Vk_ = gsl_matrix_submatrix(V_, /* m1= */ 2 * subbandX * subbandLengthN_, /* n1= */ 2 * subbandX * subbandLengthN_,
					       2 * subbandLengthN_, 2 * subbandLengthN_);
    gsl_matrix_memcpy(&Vk.matrix, &Vk_.matrix);
    subX++;
  }
  realify_(Hbar_k, yhat_k);
  gsl_matrix_view HkKkk1 = gsl_matrix_submatrix(prearray_, /* m1= */ 0, 2 * observationN_, 2 * observationN_, StateN_);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Hbar_k_, K_k_k1_, 0.0, &HkKkk1.matrix);

  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_, StateN_, StateN_);
  gsl_matrix_memcpy(&K_k_k1.matrix, K_k_k1_);

  gsl_matrix_view Uk = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_ + StateN_, StateN_, StateN_);
  gsl_matrix_memcpy(&Uk.matrix, U_);

  // calculate postarray by imposing lower triangular form on prearray
  lower_triangularize_();

  // conventional innovation vector
  gsl_vector_complex_memcpy(residual_, vk_);  realify_residual_();
  gsl_vector_sub(residual_real_, yhat_k_);

  // extra term required by IEKF
  gsl_vector_memcpy(delta_, position_);
  gsl_vector_sub(delta_, eta_i_);
  gsl_blas_dgemv(CblasNoTrans, 1.0, Hbar_k_, delta_, 0.0, scratch_);
  gsl_vector_sub(residual_real_, scratch_);

  // perform (local) state update
  gsl_matrix_view  Vk = gsl_matrix_submatrix(prearray_, /* m1= */ 0, /* n1= */ 0, 2 * observationN_, 2 * observationN_);
  gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, &Vk.matrix, residual_real_);
  gsl_matrix_view B21 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 0, StateN_, 2 * observationN_);
  gsl_blas_dgemv(CblasNoTrans, 1.0, &B21.matrix, residual_real_, 0.0, correction_);

  cout << endl << "Correction:" << endl;
  printVector_(correction_);
  gsl_vector_add(eta_i_, correction_);
  check_physical_constraints_();
}

// maintain the spherical coordinates within physical bounds
void BaseSphericalArrayTracker::check_physical_constraints_()
{
  double theta = gsl_vector_get(eta_i_, 0);
  double phi   = gsl_vector_get(eta_i_, 1);

  // constrain polar angle to 0 < theta < pi
  if (theta < Epsilon_) {
    gsl_vector_set(eta_i_, 0, Epsilon_);
    printf("Limiting polar angle %6.2f to %6.2f\n", theta, Epsilon_);
  } else if (theta > M_PI - Epsilon_) {
    gsl_vector_set(eta_i_, 0, M_PI - Epsilon_);
    printf("Limiting polar angle %6.2f to %6.2f\n", theta, M_PI - Epsilon_);
  } else
    gsl_vector_set(eta_i_, 0, theta);

  // not necessary to constrain the azimuth
  gsl_vector_set(eta_i_, 1, phi);
}

// calculate postarray by imposing lower triangular form on prearray
void BaseSphericalArrayTracker::lower_triangularize_()
{
  // zero out upper portion of A12 row by row
  gsl_matrix_view A11 = gsl_matrix_submatrix(prearray_, 0, 0, 2 * observationN_ + StateN_, 2 * observationN_);
  gsl_matrix_view A12 = gsl_matrix_submatrix(prearray_, 0, 2 * observationN_, 2 * observationN_ + StateN_, StateN_);

  for (unsigned rowX = 0; rowX < 2 * observationN_; rowX++) {
    for (unsigned colX = 0; colX < StateN_; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A11.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A12.matrix, rowX, colX);
      gsl_matrix_set(&A11.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
      gsl_matrix_set(&A12.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < 2 * observationN_ + StateN_; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A11.matrix, n, rowX);
	v2 = gsl_matrix_get(&A12.matrix, n, colX);
	apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A11.matrix, n, rowX, v1p);
	gsl_matrix_set(&A12.matrix, n, colX, v2p);
      }
    }
  }

  // lower triangularize A22 row by row
  gsl_matrix_view A22 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_, StateN_, StateN_);
  for (unsigned rowX = 0; rowX < StateN_; rowX++) {
    for (unsigned colX = rowX + 1; colX < StateN_; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A22.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A22.matrix, rowX, colX);
      gsl_matrix_set(&A22.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
      gsl_matrix_set(&A22.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < StateN_; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A22.matrix, n, rowX);
	v2 = gsl_matrix_get(&A22.matrix, n, colX);
	apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A22.matrix, n, rowX, v1p);
	gsl_matrix_set(&A22.matrix, n, colX, v2p);
      }
    }
  }

  // zero out all of A23 by rotating it into A22 row by row
  gsl_matrix_view A23 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_ + StateN_, StateN_, StateN_);
  for (unsigned rowX = 0; rowX < StateN_; rowX++) {
    for (unsigned colX = 0; colX < StateN_; colX++) {
      // element to be annihilated
      double c, s;
      double v1 = gsl_matrix_get(&A22.matrix, rowX, rowX);
      double v2 = gsl_matrix_get(&A23.matrix, rowX, colX);
      gsl_matrix_set(&A22.matrix, rowX, rowX, calc_givens_rotation_(v1, v2, c, s));
      gsl_matrix_set(&A23.matrix, rowX, colX, 0.0);

      // complete rotation on remainder of column
      for (unsigned n = rowX + 1; n < StateN_; n++) {
	double v1p, v2p;
	v1 = gsl_matrix_get(&A22.matrix, n, rowX);
	v2 = gsl_matrix_get(&A23.matrix, n, colX);
	apply_givens_rotation_(v1, v2, c, s, v1p, v2p);
	gsl_matrix_set(&A22.matrix, n, rowX, v1p);
	gsl_matrix_set(&A23.matrix, n, colX, v2p);
      }
    }
  }
}

void BaseSphericalArrayTracker::reset()
{
  base_decomposition_->reset();

  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++)
    (*itr)->reset();

  if (snapshot_array_ != NULL)
    snapshot_array_->zero();

  VectorFloatFeatureStream::reset();
  is_end_ = false;
}

void BaseSphericalArrayTracker::alloc_image_()
{
  if(snapshot_array_ == NULL)
    snapshot_array_ = new SnapShotArray(subbandsN_, chanN());
}

// ----- methods for class `ModalSphericalArrayTracker' -----
//
ModalSphericalArrayTracker::
ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			   unsigned maxLocalN, const String& nm)
  : BaseSphericalArrayTracker(modalDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm) { }

const gsl_vector_float* ModalSphericalArrayTracker::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  // get new snapshots
  this->alloc_image_();
  unsigned chanX = 0;
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no_ + 1);
    if((*itr)->is_end() == true) is_end_ = true;
    snapshot_array_->set_samples(samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  gsl_vector_memcpy(eta_i_, position_);
  for (unsigned localX = 0; localX < maxLocalN_; localX++) {
    double theta = gsl_vector_get(eta_i_, 0);
    double phi   = gsl_vector_get(eta_i_, 1);

    // estimate and sort Bkl factors
    for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
      const gsl_vector_complex* snapshot = snapshot_array_->snapshot(subbandX);
      base_decomposition_->estimate_Bkl(theta, phi, snapshot, subbandX);
    }

    unsigned subX = 0;
    for (BaseDecomposition::Iterator itr(base_decomposition_->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      unsigned    subbandX			= subbandEntry.subbandX();

      const gsl_vector_complex* snapshot	= snapshot_array_->snapshot(subbandX);
      gsl_vector_complex_view   vk		= gsl_vector_complex_subvector(vk_, subX * modesN_, modesN_);
      base_decomposition_->transform(snapshot, &vk.vector);
      subX++;
    }

    // perform position estimate update
    const gsl_matrix_complex* Hbar_k = base_decomposition_->linearize(eta_i_, frame_no_ + 1);
    const gsl_vector_complex* yhat_k = base_decomposition_->predicted_observation(eta_i_, frame_no_ + 1);

    double residualBefore = calc_residual_();

    update_(Hbar_k, yhat_k, base_decomposition_->subbandList());

    double residualAfter = calc_residual_();

    printf("Before local update %2d : Residual = %10.4e\n", localX, residualBefore);
    printf("After  local update %2d : Residual = %10.4e\n", localX, residualAfter);

    // test for convergence
    if ((residualBefore - residualAfter) / (residualBefore + residualAfter) < Tolerance_) break;
  }
  gsl_vector_memcpy(position_, eta_i_);
  copy_position_();

  // copy out updated state estimation error covariance matrix
  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_, StateN_, StateN_);
  gsl_matrix_memcpy(K_k_k1_, &K_k_k1.matrix);

  cout << "K_k_k1" << endl;
  printMatrix_(K_k_k1_);

  increment_();
  return vector_;
}


// ----- methods for class `SpatialSphericalArrayTracker' -----
//
SpatialSphericalArrayTracker::
SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u, double sigma2_v, double sigma2_init,
			     unsigned maxLocalN, const String& nm)
  : BaseSphericalArrayTracker(spatialDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm) { }


const gsl_vector_float* SpatialSphericalArrayTracker::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  // get new snapshots
  this->alloc_image_();
  unsigned chanX = 0;
  for (ChannelIterator_ itr = channelList_.begin(); itr != channelList_.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frame_no_ + 1);
    if((*itr)->is_end() == true) is_end_ = true;
    snapshot_array_->set_samples(samp, chanX);  chanX++;
  }
  snapshot_array_->update();

  gsl_vector_memcpy(eta_i_, position_);
  for (unsigned localX = 0; localX < maxLocalN_; localX++) {
    double theta = gsl_vector_get(eta_i_, 0);
    double phi   = gsl_vector_get(eta_i_, 1);

    // estimate and sort Bkl factors
    for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
      const gsl_vector_complex* snapshot = snapshot_array_->snapshot(subbandX);
      base_decomposition_->estimate_Bkl(theta, phi, snapshot, subbandX);
    }

    unsigned subX = 0;
    for (BaseDecomposition::Iterator itr(base_decomposition_->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      unsigned	  subbandX			= subbandEntry.subbandX();

      const gsl_vector_complex* snapshot 	= snapshot_array_->snapshot(subbandX);
      gsl_vector_complex_view   vk		= gsl_vector_complex_subvector(vk_, subX * BaseDecomposition::ChannelsN_, BaseDecomposition::ChannelsN_);
      gsl_vector_complex_memcpy(&vk.vector, snapshot);
      subX++;
    }

    double residualBefore = calc_residual_();
    printf("Before local update %2d : Residual = %12.6e\n", localX, residualBefore);

    // perform position estimate update
    const gsl_matrix_complex* Hbar_k = base_decomposition_->linearize(eta_i_, frame_no_ + 1);
    const gsl_vector_complex* yhat_k = base_decomposition_->predicted_observation(eta_i_, frame_no_ + 1);

    update_(Hbar_k, yhat_k, base_decomposition_->subbandList());

    theta = gsl_vector_get(eta_i_, 0);  phi = gsl_vector_get(eta_i_, 1);
    for (BaseDecomposition::Iterator itr(base_decomposition_->subbandList()); itr.more(); itr++) {
      const BaseDecomposition::SubbandEntry& subbandEntry(*itr);
      unsigned   subbandX                      = subbandEntry.subbandX();

      base_decomposition_->calculate_gkl(theta, phi, subbandX);
    }

    double residualAfter = calc_residual_();
    printf("After  local update %2d : Residual = %12.6e\n", localX, residualAfter);

    // test for convergence
    if ((residualBefore - residualAfter) / (residualBefore + residualAfter) < Tolerance_) break;
  }
  gsl_vector_memcpy(position_, eta_i_);
  copy_position_();

  // re-estimate and re-sort Bkl factors
  double theta = gsl_vector_get(eta_i_, 0);
  double phi   = gsl_vector_get(eta_i_, 1);
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    const gsl_vector_complex* snapshot = snapshot_array_->snapshot(subbandX);
    base_decomposition_->estimate_Bkl(theta, phi, snapshot, subbandX);
  }

  // copy out updated state estimation error covariance matrix
  gsl_matrix_view K_k_k1 = gsl_matrix_submatrix(prearray_, 2 * observationN_, 2 * observationN_, StateN_, StateN_);
  gsl_matrix_memcpy(K_k_k1_, &K_k_k1.matrix);

  cout << "K_k_k1" << endl;
  printMatrix_(K_k_k1_);

  increment_();
  return vector_;
}



// ----- methods for class `PlaneWaveSimulator' -----
//
const gsl_complex PlaneWaveSimulator::ComplexZero_ = gsl_complex_rect(0.0, 0.0);

PlaneWaveSimulator::PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
				       unsigned channelX, double theta, double phi, const String& nm)
  : VectorComplexFeatureStream(modalDecomposition->subbandsN(), nm),
    subbandsN_(modalDecomposition->subbandsN()), subbandsN2_(modalDecomposition->subbandsN2()), channelX_(channelX),
    theta_(theta), phi_(phi),
    source_(source), modalDecomposition_(modalDecomposition),
    subbandCoefficients_(gsl_vector_complex_calloc(subbandsN2_ + 1))
{
  // cout << "Initializing 'PlaneWaveSimulator' ... " << endl;
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    gsl_complex coefficient = ComplexZero_;
    for (int n = 0; n <= modalDecomposition_->orderN(); n++) {			// order
      gsl_complex bn = modalDecomposition_->modal_coefficient(n, subbandX);	// scaled modal coefficient
      gsl_complex coeff_n = ComplexZero_;
      for (int m = -n ; m <= n; m++) {						// degree
	coeff_n = gsl_complex_add(coeff_n, gsl_complex_mul(modalDecomposition_->harmonic(n, m, channelX_),
							   modalDecomposition_->harmonic(n, m, theta_, phi_)));
      }
      coefficient = gsl_complex_add(coefficient, gsl_complex_mul(bn, coeff_n));
    }
    gsl_vector_complex_set(subbandCoefficients_, subbandX, coefficient);
  }
  // cout << "Done." << endl;
}

PlaneWaveSimulator::~PlaneWaveSimulator()
{
  gsl_vector_complex_free(subbandCoefficients_);
}

const gsl_vector_complex* PlaneWaveSimulator::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  const gsl_vector_complex* block = source_->next(frame_no_ + 1);
  for (unsigned subbandX = 0; subbandX <= subbandsN2_; subbandX++) {
    gsl_complex component = gsl_complex_mul(gsl_vector_complex_get(subbandCoefficients_, subbandX), gsl_vector_complex_get(block, subbandX));
    gsl_vector_complex_set(vector_, subbandX, component);
    if (subbandX != 0 && subbandX != subbandsN2_)
      gsl_vector_complex_set(vector_, subbandsN_ - subbandX, gsl_complex_conjugate(component));
  }

  increment_();
  return vector_;
}

void PlaneWaveSimulator::reset()
{
  source_->reset();  VectorComplexFeatureStream::reset();
}
