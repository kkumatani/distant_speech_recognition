/*
 * @file feature.cc
 * @brief Speech recognition front end.
 * @author John McDonough, Tobias Gehrig, Kenichi Kumatani, Friedrich Faubel
 */

#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "common/mach_ind_io.h"
#include "feature/feature.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include "matrix/gslmatrix.h"

#include "common/jpython_error.h"

#include <sndfile.h>

/**
   @brief unpack the data from half- to gsl complex array (with symmetrical conjugate componnent)
 */
void unpack_half_complex(gsl_vector_complex* tgt, const double* src)
{
  int len  = tgt->size;
  int len2 = (len + 1) / 2;

  gsl_vector_complex_set(tgt, 0,    gsl_complex_rect(src[0],    0));
  if ((len & 1) == 0) // len == even number
    gsl_vector_complex_set(tgt, len2, gsl_complex_rect(src[len2], 0));
  for (int m = 1; m < len2; m++) {
    gsl_vector_complex_set(tgt, m,
                           gsl_complex_rect(src[m],  src[len-m]));
    gsl_vector_complex_set(tgt, len-m,
                           gsl_complex_rect(src[m], -src[len-m])); // symmetrical
  }
}

/**
   @brief unpack the data from half- to specified-size complex array
   @param gsl_vector_complex* tgt[out]
   @param const unsigned N2[in] the number of element to be copied into tgt
   @param const double* src[in]
   @param const unsigned N[in] size of "*src"
*/
void unpack_half_complex(gsl_vector_complex* tgt, const unsigned N2, const double* src, const unsigned N)
{
  for (unsigned m = 0; m <= N2; m++) {
    if (m == 0 || m == N2) {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], 0.0));
    } else {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], src[N-m]));
    }
  }
}

/**
   @brief pack the data from standard to half-complex form
*/
void pack_half_complex(double* tgt, const gsl_vector_complex* src, unsigned size)
{
  unsigned len  = (size == 0)?(src->size):size;
  unsigned len2 = (len+1) / 2;

  gsl_complex entry = gsl_vector_complex_get(src, 0);
  tgt[0]    = GSL_REAL(entry);

  for (unsigned m = 1; m < len2; m++) {
    entry      = gsl_vector_complex_get(src, m);
    tgt[m]     = GSL_REAL(entry);
    tgt[len-m] = GSL_IMAG(entry);
  }

  if ((len & 1) == 0) {
    entry     = gsl_vector_complex_get(src, len2);
    tgt[len2] = GSL_REAL(entry);
  }
}

#ifdef HAVE_LIBFFTW3

// unpack the data from fftw format to standard complex form
//
void fftwUnpack(gsl_vector_complex* tgt, const fftw_complex* src)
{
  int len  = tgt->size;
  int len2 = len / 2;

  for (unsigned i = 0; i <= len2; i++) {
    gsl_vector_complex_set(tgt, i, gsl_complex_rect(src[i][0], src[i][1]));
    if (i != 0 && i != len2)
      gsl_vector_complex_set(tgt, len - i , gsl_complex_rect(src[i][0], -src[i][1]));
  }
}

#endif


// ----- methods for class `FileFeature' -----
//
const gsl_vector_float* FileFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  increment_();
  if (frame_no_ >= feature_->size1) {
    throw jiterator_error("end of samples!");
  }

  float* feat = gsl_matrix_float_ptr(feature_, frame_no_, 0);
  for (int i = 0; i < GSL_MATRIX_NCOLS(feature_); i++)
    gsl_vector_float_set(vector_, i, feat[i]);

  return vector_;
}

void FileFeature::bload(const String& fileName, bool old)
{
  cout << "Loading features from " << fileName << "." << endl;

  if (feature_ != NULL) {
    gsl_matrix_float_free(feature_);
    feature_ = NULL;
  }

  feature_ = gsl_matrix_float_load(feature_, fileName.chars(), old);

  reset();

  printf("Matrix is %lu x %lu\n", feature_->size1, GSL_MATRIX_NCOLS(feature_));
}

void FileFeature::copy(gsl_matrix_float* matrix)
{
  if ((feature_ == NULL) || (feature_->size1 != matrix->size1) || (GSL_MATRIX_NCOLS(feature_) != matrix->size2)) {
    gsl_matrix_float_free(feature_);
    feature_ = gsl_matrix_float_calloc(matrix->size1, matrix->size2);
  }

  gsl_matrix_float_memcpy(feature_, matrix);
}

FileFeature& FileFeature::operator=(const FileFeature& f)
{
  //  fmatrixCopy(feature_, f.feature_);
  gsl_matrix_float_memcpy(feature_, f.feature_);

  return *this;
}


// ----- methods for Conversion24bit2Short -----
//
const gsl_vector_short* Conversion24bit2Short::next(int frame_no) {
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();

  const gsl_vector_char* newBlock = src_->next(frame_no_);
  unsigned short ii;
  for (int i=0; i<size(); i++) {

    // this is how array3conversion_dsp.c from NSFS does it:
    // *buffer++ = (inbuf[(i*64+j)*3+1])|(inbuf[(i*64+j)*3]<<8);

    ii = ((unsigned char)gsl_vector_char_get(newBlock, i*3+1) |
          ((unsigned char)gsl_vector_char_get(newBlock, i*3)) << 8);
    gsl_vector_short_set(vector_, i, ii);
  }
  return vector_;
}


// ----- methods for Conversion24bit2Float -----
//
const gsl_vector_float* Conversion24bit2Float::next(int frame_no) {
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frame_no - 1, frame_no_);

  increment_();

  const gsl_vector_char* newBlock = src_->next(frame_no_);
  signed int ii;
  unsigned char *p = (unsigned char*) &ii;
  unsigned char sample;
  for (int i=0; i<size(); i++) {
    sample = (unsigned char)gsl_vector_char_get(newBlock, i*3);
    /* Code to put on a 32 bits range */
    /* From Big Endian (CMA's data) to Linux Little Endian numbers */
    if (sample & 128)
      *(p+3) = 0xFF;
    else
      *(p+3) = 0x00;
    *(p+2) = sample;
    *(p+1) = (unsigned char)gsl_vector_char_get(newBlock, i*3+1);
    *(p)   = (unsigned char)gsl_vector_char_get(newBlock, i*3+2);
    gsl_vector_float_set( vector_, i, ii*1.0 );
  }
  return vector_;
}


// ----- methods for class `SampleFeature' -----
//
SampleFeature::SampleFeature(const String& fn, unsigned blockLen,
                             unsigned shiftLen, bool padZeros, const String& nm) :
  VectorFloatFeatureStream(blockLen, nm),
  samples_(NULL), ttlsamples_(0), shiftLen_(shiftLen), cur_(0), pad_zeros_(padZeros),
  copy_fsamples_(NULL),copy_dsamples_(NULL)
{
  if (fn != "") read(fn);
  is_end_ = false;
}

SampleFeature::~SampleFeature()
{
    if (samples_ != NULL) delete[] samples_;
    if (copy_fsamples_ != NULL) gsl_vector_float_free(copy_fsamples_);
    if (copy_dsamples_ != NULL) gsl_vector_free(copy_dsamples_);
}

unsigned SampleFeature::
read(const String& fn, int format, int samplerate, int chX, int chN, int cfrom, int to, int outsamplerate, float norm)
{
  using namespace sndfile;
  SNDFILE* sndfile;
  SF_INFO sfinfo;
  float* tmpsamples;
  int nsamples;

  norm_ = norm;
  ttlsamples_ = 0;

  if ( NULL != samples_ ){ delete[] samples_;}
  samples_ = NULL; // avoid double deletion if file cannot be read

  sfinfo.format = format;
  sfinfo.samplerate = samplerate;
  sfinfo.channels = chN;
  sndfile = sf_open(fn.c_str(), SFM_READ, &sfinfo);
  if (!sndfile)
    throw jio_error("Could not open file %s.", fn.c_str());

  if (sf_error(sndfile)) {
    sf_close(sndfile);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile));
  }

  if (norm == 0.0) {
#ifdef DEBUG
    cout << "Disabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  } else {
#ifdef DEBUG
    cout << "Enabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  }

  if (outsamplerate == -1) outsamplerate = sfinfo.samplerate;

#ifdef DEBUG
  cout << "channels: " << sfinfo.channels << endl;
  cout << "frames: " << sfinfo.frames << endl;
  cout << "samplerate: " << sfinfo.samplerate << endl;
#endif

  if ((to < 0) || (to >= sfinfo.frames))
    to = sfinfo.frames - 1;
  if (cfrom < 0)
    cfrom = 0;
  if ((cfrom > to) || (cfrom > sfinfo.frames)) {
    sf_close(sndfile);
    throw jio_error("Cannot load samples from %d to %d.", cfrom, to);
  }
  nsamples = to - cfrom + 1;

  // Allocating memory for samples
  tmpsamples = new float[nsamples*sfinfo.channels];
  if (tmpsamples == NULL) {
    sf_close(sndfile);
    throw jallocation_error("Error when allocating memory for samples.");
  }

  if (sf_seek(sndfile, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  ttlsamples_ = sf_readf_float(sndfile, tmpsamples, nsamples);
#ifdef DEBUG
  cout << "read frames: " << ttlsamples_ << endl;
#endif

  samplerate_ = sfinfo.samplerate;
  chN = sfinfo.channels;
  nChan_ = chN;

  if (chX > sfinfo.channels || chX < 1) {
    if (chX == 0)
      throw jconsistency_error("Multi-channel read is not yet supported.");

    // for now just allow one channel to be loaded
    throw jconsistency_error("Selected channel out of range of available channels.");
    //    chX = 1;
  }
  chX--;
  if (chX < 0 || chN == 1 )
    samples_ = tmpsamples;
  else {
    samples_ = new float[nsamples];
    if (samples_ == NULL) {
      delete[] tmpsamples;
      sf_close(sndfile);
      throw jallocation_error("Error when allocating memory for samples");
    }
    // Copy the selected channel to samples_
    for (int i=0; i < nsamples; i++)
      samples_[i] = tmpsamples[i*sfinfo.channels + chX];
    delete[] tmpsamples;
  }

  if (samplerate_ <= outsamplerate)
    if (norm != 1.0 && norm != 0.0)
      for (int i=0; i < nsamples; i++)
	samples_[i] *= norm;

#ifdef SRCONV
  if (samplerate_ != outsamplerate) {
#ifdef DEBUG
    cout << "sample rate converting to " << outsamplerate<<endl;
#endif
    SRC_DATA data;
    data.input_frames = ttlsamples_;
    data.src_ratio = (float)outsamplerate / (float)samplerate_;
    data.output_frames = (long)ceil(data.src_ratio*(float)ttlsamples_);
#ifdef DEBUG
    cout << "src_ratio: " << data.src_ratio << endl;
    cout << "output_frames: " << data.output_frames << endl;
    char normalisation = sf_command (sndfile, SFC_GET_NORM_FLOAT, NULL, 0) ;
    cout << "norm: " << (normalisation?"true":"false") << endl;
#endif
    data.data_in = samples_;
    data.data_out = new float[data.output_frames];
#ifdef DEBUG
    cout << "channels "<<((chX<0)?sfinfo.channels:1) <<endl;
#endif
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, (chX<0)?sfinfo.channels:1)) {    
      sf_close(sndfile);
      throw jconsistency_error("Error during samplerate conversion.");
    }
    delete[] samples_;
    samples_ = data.data_out;
    ttlsamples_ = data.output_frames_gen;
#ifdef DEBUG
    cout << "output_frames_gen: " << ttlsamples_ << endl;
#endif
  }
#endif

  if (samplerate_ > outsamplerate)
    if (norm != 1.0 && norm != 0.0)
      for (int i=0; i < nsamples; i++)
	samples_[i] *= norm;

  format_ = sfinfo.format;

  cur_        = 0;
  reset();
  is_end_ = false;

  sf_close(sndfile);
  return ttlsamples_;
}

void SampleFeature::addWhiteNoise(float snr)
{
  double desiredNoA;
  double avgSig = 0.0, avgNoi = 0.0;
  struct timeval now_time;
  short *noiseSamp = new short[ ttlsamples_];
  int max = INT_MIN;

  for(int i=0; i < ttlsamples_; i++)
    avgSig += fabsf(samples_[i]);
  avgSig = avgSig / (double)ttlsamples_;

  gettimeofday( &now_time, NULL);
  srand( now_time.tv_usec );

  for(int i=0;i<ttlsamples_;i++){
    noiseSamp[i] = rand();
    if( noiseSamp[i] > max )
      max = noiseSamp[i];
  }
  for(int i=0;i<ttlsamples_;i++){
    noiseSamp[i] = (noiseSamp[i]/(float)max) - 0.5;
    avgNoi += std::abs(noiseSamp[i]);
  }
  avgNoi = avgNoi / (double)ttlsamples_;

  //desiredNoA = pow( 10, ( log10(avgSig) - ( snr / 20.0 ) ) );
  desiredNoA = avgSig / pow( 10.0, snr/20.0 );

  for(int i=0;i<ttlsamples_;i++)
    noiseSamp[i] = desiredNoA * noiseSamp[i] / avgNoi;

  for(int i=0;i<ttlsamples_;i++)
    samples_[i] += noiseSamp[i];

  delete [] noiseSamp;
}

void SampleFeature::write(const String& fn, int format, int sampleRate)
{
  using namespace sndfile;
  SNDFILE* sndfile;
  SF_INFO sfinfo;
  int nsamp, frames = ttlsamples_;
  float norm;
  float *samplesorig = NULL;

  if (sampleRate == -1) sampleRate = samplerate_;
#ifdef SRCONV
  sfinfo.samplerate = sampleRate;
#else
  sfinfo.samplerate = samplerate_;
#endif
  sfinfo.format = format;
  sfinfo.channels = 1;
  sfinfo.frames = 0;
  sfinfo.sections = 0;
  sfinfo.seekable = 0;

  sndfile = sf_open(fn.c_str(), SFM_WRITE, &sfinfo);
  if (!sndfile)
    throw jio_error("Error opening file %s.", fn.c_str());

  if (norm_ == 0.0) {
#ifdef DEBUG
    cout << "Disabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  } else {
#ifdef DEBUG
    cout << "Enabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  }
 
  if (norm_ != 1.0 && norm_ != 0.0) {
    norm = 1/norm_;
    samplesorig = samples_;
    samples_ = new float[ttlsamples_];
    for (int i=0; i < ttlsamples_; i++)
      samples_[i] = samplesorig[i]*norm;
  }

#ifdef SRCONV
  if (samplerate_ != sfinfo.samplerate) {
    SRC_DATA data;
#ifdef DEBUG
    cout << "sample rate converting to " << sfinfo.samplerate << endl;
#endif
    data.input_frames = ttlsamples_;
    data.src_ratio = (float)sfinfo.samplerate / (float)samplerate_;
    data.output_frames = (long)ceil(data.src_ratio*(float)ttlsamples_);
#ifdef DEBUG
    cout << "src_ratio: " << data.src_ratio << endl;
    cout << "output_frames: " << data.output_frames << endl;
    char normalisation = sf_command (sndfile, SFC_GET_NORM_FLOAT, NULL, 0) ;
    cout << "norm: " << (normalisation?"true":"false") << endl;
#endif
    data.data_in = samples_; //new float[data.input_frames];
    data.data_out = new float[data.output_frames];
    /*
    for (unsigned int i=0; i < ttlsamples_ ; i++)
      data.data_in[i] = (float)samples_[i] / (float)0x8000;
    */
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, 1)) {    
      sf_close(sndfile);
      if (samplesorig != NULL) {
	delete[] samples_;
	samples_ = samplesorig;
      }
      throw jconsistency_error("Error during samplerate conversion.");
    }
    nsamp = sf_writef_float(sndfile, data.data_out, data.output_frames_gen);
    frames = data.output_frames_gen;
#ifdef DEBUG
    cout << "output_frames_gen: " << frames << endl;
#endif
  } else
#endif
    nsamp = sf_writef_float(sndfile, samples_, ttlsamples_);
  if(nsamp != int(frames))
    cerr << "unable to write " << (frames - nsamp) << " samples" << endl;
  sf_close(sndfile);
  if (samplesorig != NULL) {
    delete[] samples_;
    samples_ = samplesorig;
  }
}

const gsl_vector_float* SampleFeature::data()
{
  if (NULL == copy_fsamples_) {
    copy_fsamples_ = gsl_vector_float_calloc(ttlsamples_);
  } else {
    gsl_vector_float_free(copy_fsamples_);
    copy_fsamples_ = gsl_vector_float_calloc(ttlsamples_);
  }

  for (unsigned i = 0; i < ttlsamples_; i++)
    gsl_vector_float_set(copy_fsamples_, i, samples_[i]);

  return copy_fsamples_;
}

const gsl_vector* SampleFeature::dataDouble()
{
  if( NULL == copy_dsamples_ )
    copy_dsamples_ = gsl_vector_calloc(ttlsamples_);
  else{
    gsl_vector_free( copy_dsamples_ );
    copy_dsamples_ = gsl_vector_calloc(ttlsamples_);
  }

  for (unsigned i = 0; i < ttlsamples_; i++)
    gsl_vector_set(copy_dsamples_, i, samples_[i]);

  return copy_dsamples_;
}


#define SLOW    -32768
#define SHIGH    32767
#define SLIMIT(x) \
  ((((x) < SLOW)) ? (SLOW) : (((x) < (SHIGH)) ? (x) : (SHIGH)))

void SampleFeature::zeroMean()
{
  double mean = 0.0;

  if (samples_ == NULL)
    throw jconsistency_error("Must first load data before setting mean to zero.");

  for (unsigned i = 0; i < ttlsamples_; i++)
    mean += samples_[i];

  mean /= ttlsamples_;

  for (unsigned i = 0; i < ttlsamples_; i++)
    samples_[i] = int(SLIMIT(samples_[i] - mean));
}

void SampleFeature::cut(unsigned cfrom, unsigned cto)
{
  if (cfrom >= cto)
    throw j_error("Cut bounds (%d,%d) do not match.", cfrom, cto);

  if (cto >= ttlsamples_)
    throw j_error("Do not have enough samples (%d,%d).", cto, ttlsamples_);

  ttlsamples_ = cto - cfrom + 1;
  float* newSamples = new float[ttlsamples_];
  memcpy(newSamples, samples_ + cfrom, ttlsamples_ * sizeof(float));

  delete[] samples_;
  samples_ = newSamples;
}

// create a generator chosen by the environment variable GSL_RNG_TYPE
void SampleFeature::randomize(int startX, int endX, double sigma2)
{
  const gsl_rng_type* rng_type;
  gsl_rng*            rnd_gen;

  gsl_rng_env_setup();
  rng_type = gsl_rng_default;
  rnd_gen  = gsl_rng_alloc(rng_type);

  printf("Randomizing from %6.2f to %6.2f\n", startX / 16000.0, endX / 16000.0);

  for (int n = startX; n <= endX; n++) {
    samples_[n] = gsl_ran_gaussian(rnd_gen, sigma2);
  }
}

const gsl_vector_float* SampleFeature::next(int frame_no)
{

  if (is_end_) {
    throw jiterator_error("end of samples!");
  }

  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_){
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);
  }

  if (cur_ >= ttlsamples_) {
    is_end_ = true;

    if( NULL != samples_ )
      delete [] samples_;
    samples_ = NULL;
    throw jiterator_error("end of samples!");
  }

  if (cur_ + size() >= ttlsamples_) {
    if (pad_zeros_) {
      gsl_vector_float_set_zero(vector_);
      unsigned remainingN = ttlsamples_ - cur_;
      for (unsigned i = 0; i < remainingN; i++)
        gsl_vector_float_set(vector_, i, samples_[cur_ + i]);
    } else {
      is_end_ = true;
      if( NULL != samples_ )
        delete [] samples_;
      samples_ = NULL;
      throw jiterator_error("end of samples!");
    }
  } else {
    for (unsigned i = 0; i < size(); i++)
      gsl_vector_float_set(vector_, i, samples_[cur_ + i]);
  }

  cur_ += shiftLen_;

  increment_();
  return vector_;
}

void SampleFeature::copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to)
{
  if( NULL != samples_ ){ delete [] samples_; }
  if (to == 0) {
    ttlsamples_ = src->ttlsamples_;
  } else {
    if (to <= cfrom)
      throw jindex_error("cfrom = %d and to = %d are inconsistent.", cfrom, to);
    if (to >= src->ttlsamples_)
      to = src->ttlsamples_ - 1;

    ttlsamples_ = to - cfrom;
  }
    
  samples_ = new float[ttlsamples_];
  memcpy(samples_, src->samples_ + cfrom, ttlsamples_ * sizeof(float));
}

void SampleFeature::setSamples(const gsl_vector* samples, unsigned sampleRate)
{
  if( NULL != samples_ ){ delete [] samples_; }
  samplerate_ = sampleRate;
  ttlsamples_ = samples->size;
  samples_ = new float[ttlsamples_];
  for (unsigned sampleX = 0; sampleX < ttlsamples_; sampleX++)
    samples_[sampleX] = float(gsl_vector_get(samples, sampleX));

  reset();
}

// ----- methods for class `IterativeSingleChannelSampleFeature' -----
//
IterativeSingleChannelSampleFeature::IterativeSingleChannelSampleFeature( unsigned blockLen, const String& nm )
  : VectorFloatFeatureStream(blockLen, nm), blockLen_(blockLen), cur_(0)
{ 
  interval_	 = 30;
  sndfile_ = NULL;
  samples_ = NULL;
  last_ = false;
}

IterativeSingleChannelSampleFeature::~IterativeSingleChannelSampleFeature()
{
  if (sndfile_ != NULL) 
    sf_close(sndfile_);
  if( samples_ != NULL )
    delete[] samples_;
  sndfile_ = NULL;
  samples_ = NULL;
}

void IterativeSingleChannelSampleFeature::reset()
{
  ttlsamples_ = cur_ = 0;  last_ = false;  VectorFloatFeatureStream::reset();
}

void IterativeSingleChannelSampleFeature::read(const String& fileName, int format, int samplerate, int cfrom, int cto )
{
  using namespace sndfile;

  delete[] samples_;  samples_ = NULL;
  if (sndfile_ != NULL) {  sf_close(sndfile_); sndfile_ = NULL; }

  sfinfo_.channels   = 1;
  sfinfo_.samplerate = samplerate;
  sfinfo_.format     = format;

  sndfile_ = sf_open(fileName.c_str(), SFM_READ, &sfinfo_);
  if (!sndfile_)
    throw jio_error("Could not open file %s.", fileName.c_str());

  if (sf_error(sndfile_)) {
    sf_close(sndfile_);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile_));
  }

  if( sfinfo_.channels > 1 ){
    sf_close(sndfile_);
    throw j_error("IterativeSingleChannelSampleFeature is for the single channel file only\n");
  }
#ifdef DEBUG
  cout << "channels: "   << sfinfo_.channels   << endl;
  cout << "frames: "     << sfinfo_.frames     << endl;
  cout << "samplerate: " << sfinfo_.samplerate << endl;
#endif

  sf_command(sndfile_, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  blockN_     = interval_ * sfinfo_.samplerate / blockLen_ + 1;
  sampleN_    = blockN_   * blockLen_;
  samples_ = new float[sampleN_];

  for (unsigned i = 0; i < sampleN_; i++)
    samples_[i] = 0.0;

  if (sf_seek(sndfile_, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  if (cto > 0 and cto < cfrom)
    throw jconsistency_error("Segment cannot start at %d and end at %d", cfrom, cto);
  cto_ = cto - cfrom;
}

const gsl_vector_float* IterativeSingleChannelSampleFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  unsigned currentFrame = cur_ % blockN_;
  if ( currentFrame == 0) {

    if (last_ || (cto_ > 0 && cur_ * blockLen_ > cto_)){
      delete[] samples_;  samples_ = NULL;
      if (sndfile_ != NULL) {  sf_close(sndfile_); sndfile_ = NULL; }
      throw jiterator_error("end of samples!");
    }

    for (unsigned i = 0; i < sampleN_; i++)
      samples_[i] = 0.0;
    unsigned readN = sf_readf_float(sndfile_, samples_, sampleN_);

    ttlsamples_ += readN;

    if (readN < sampleN_) last_ = true;
  }

  unsigned offset = currentFrame * blockLen_;
  for (unsigned i = 0; i < blockLen_; i++)
    gsl_vector_float_set(vector_, i, samples_[offset+i]);

  cur_++;
  increment_();
  return vector_;
}


// ----- methods for class `IterativeSampleFeature' -----
//
float*            IterativeSampleFeature::allSamples_    = NULL;
sndfile::SNDFILE* IterativeSampleFeature::sndfile_       = NULL;
sndfile::SF_INFO  IterativeSampleFeature::sfinfo_;
unsigned          IterativeSampleFeature::interval_	 = 30;
unsigned          IterativeSampleFeature::blockN_;
unsigned          IterativeSampleFeature::sampleN_;
unsigned          IterativeSampleFeature::allSampleN_;
unsigned          IterativeSampleFeature::ttlsamples_;

IterativeSampleFeature::IterativeSampleFeature(unsigned chX, unsigned blockLen, unsigned firstChanX, const String& nm)
  : VectorFloatFeatureStream(blockLen, nm), blockLen_(blockLen), chanX_(chX),  firstChanX_(firstChanX), cur_(0){ }

IterativeSampleFeature::~IterativeSampleFeature()
{
  if (sndfile_ != NULL)
    sf_close(sndfile_);
  sndfile_ = NULL;
  if( allSamples_ != NULL )
    delete[] allSamples_;
  allSamples_ = NULL;
}

void IterativeSampleFeature::reset()
{
  ttlsamples_ = cur_ = 0;  last_ = false;  VectorFloatFeatureStream::reset();
}

void IterativeSampleFeature::read(const String& fileName, int format, int samplerate, int chN, int cfrom, int cto )
{
  using namespace sndfile;

  if ( chanX_ != firstChanX_ ) return;

  delete[] allSamples_;  allSamples_ = NULL;
  if (sndfile_ != NULL) { sf_close(sndfile_); sndfile_ = NULL; }

  sfinfo_.channels   = chN;
  sfinfo_.samplerate = samplerate;
  sfinfo_.format     = format;

  sndfile_ = sf_open(fileName.c_str(), SFM_READ, &sfinfo_);
  if (!sndfile_)
    throw jio_error("Could not open file %s.", fileName.c_str());

  if (sf_error(sndfile_)) {
    sf_close(sndfile_);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile_));
  }

#ifdef DEBUG
  cout << "channels: "   << sfinfo_.channels   << endl;
  cout << "frames: "     << sfinfo_.frames     << endl;
  cout << "samplerate: " << sfinfo_.samplerate << endl;
#endif

  sf_command(sndfile_, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  blockN_     = interval_ * sfinfo_.samplerate / blockLen_ + 1;
  sampleN_    = blockN_   * blockLen_;
  allSampleN_ = sampleN_  * sfinfo_.channels;
  allSamples_ = new float[allSampleN_];

  if (sf_seek(sndfile_, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  if (cto > 0 and cto < cfrom)
    throw jconsistency_error("Segment cannot start at %d and end at %d", cfrom, cto);
  cto_ = cto - cfrom;
}

const gsl_vector_float* IterativeSampleFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  unsigned currentFrame = cur_ % blockN_;
  if (chanX_ == firstChanX_ && currentFrame == 0) {

    if (last_ || (cto_ > 0 && cur_ * blockLen_ > cto_)){
      throw jiterator_error("end of samples!");
    }

    for (unsigned i = 0; i < allSampleN_; i++)
      allSamples_[i] = 0.0;
    unsigned readN = sf_readf_float(sndfile_, allSamples_, sampleN_);

    ttlsamples_ += readN;

    if (readN < sampleN_) last_ = true;
  }

  unsigned offset = currentFrame * sfinfo_.channels * blockLen_;
  for (unsigned i = 0; i < blockLen_; i++)
    gsl_vector_float_set(vector_, i, allSamples_[offset + i * sfinfo_.channels + chanX_]);

  cur_++;
  increment_();
  return vector_;
}


// ----- methods for class `BlockSizeConversionFeature' -----
//
BlockSizeConversionFeature::
BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
                           unsigned blockLen,
                           unsigned shiftLen, const String& nm)
  : VectorFloatFeatureStream(blockLen, nm), src_(src),
    inputLen_(src_->size()), blockLen_(blockLen), shiftLen_(shiftLen),
    overlapLen_(blockLen_ - shiftLen_), curin_(0), curout_(0), src_frame_no_(-1)
{
  if (blockLen_ < shiftLen_)
    throw jdimension_error("Block length (%d) is less than shift length (%d).\n",
                           blockLen_, shiftLen_);
}

void BlockSizeConversionFeature::inputLonger_()
{
  if (frame_no_ == frame_reset_no_) {
    src_frame_no_++;
    src_feat_ = src_->next(src_frame_no_);
    memcpy(vector_->data, src_feat_->data, blockLen_ * sizeof(float));
    curin_ += blockLen_;
    return;
  }

  if (overlapLen_ > 0)
    memmove(vector_->data, vector_->data + shiftLen_, overlapLen_ * sizeof(float));

  if (curin_ + shiftLen_ < inputLen_) {
    memcpy(vector_->data + overlapLen_, src_feat_->data + curin_, shiftLen_ * sizeof(float));
    curin_ += shiftLen_;
  } else {
    int remaining = inputLen_ - curin_;

    if (remaining < 0)
      throw jconsistency_error("Remaining sample (%d) cannot be negative.\n", remaining);

    if (remaining > 0)
      memcpy(vector_->data + overlapLen_, src_feat_->data + curin_, remaining * sizeof(float));

    curin_ = 0;
    src_frame_no_++;
    src_feat_ = src_->next(src_frame_no_);
    unsigned fromNew = shiftLen_ - remaining;

    memcpy(vector_->data + overlapLen_ + remaining, src_feat_->data + curin_, fromNew * sizeof(float));
    curin_ += fromNew;
  }
}

void BlockSizeConversionFeature::outputLonger_()
{
  if (frame_no_ == frame_reset_no_) {
    while (curout_ + inputLen_ <= blockLen_) {
      src_frame_no_++;
      src_feat_ = src_->next(src_frame_no_);
      memcpy(vector_->data + curout_, src_feat_->data, inputLen_ * sizeof(float));
      curout_ += inputLen_;
    }

    int remaining = blockLen_ - curout_;
    if (remaining > 0) {
      src_frame_no_++;
      src_feat_ = src_->next(src_frame_no_);
      memcpy(vector_->data + curout_, src_feat_->data, remaining * sizeof(float));
      curin_ += remaining;
    }
    curout_ = 0;
    return;
  }

  if (overlapLen_ > 0) {
    memmove(vector_->data, vector_->data + shiftLen_, overlapLen_ * sizeof(float));
    curout_ += overlapLen_;
  }

  if (curin_ > 0) {
    int remaining = inputLen_ - curin_;
    if (remaining > 0) {
      memcpy(vector_->data + curout_, src_feat_->data + curin_, remaining * sizeof(float));
      curout_ += remaining;
    }
    curin_ = 0;
  }

  while (curout_ + inputLen_ <= blockLen_) {
    src_frame_no_++;
    src_feat_ = src_->next(src_frame_no_);
    memcpy(vector_->data + curout_, src_feat_->data, inputLen_ * sizeof(float));
    curout_ += inputLen_;
  }

  int remaining = blockLen_ - curout_;
  if (remaining > 0) {
    src_frame_no_++;
    src_feat_ = src_->next(src_frame_no_);
    memcpy(vector_->data + curout_, src_feat_->data, remaining * sizeof(float));
    curin_ += remaining;
  }
  curout_ = 0;
}

const gsl_vector_float* BlockSizeConversionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (inputLen_ > shiftLen_)
    inputLonger_();
  else
    outputLonger_();

  increment_();
  return vector_;
}


// ----- methods for class `BlockSizeConversionFeatureShort' -----
//
BlockSizeConversionFeatureShort::
BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
                                unsigned blockLen,
                                unsigned shiftLen, const String& nm)
  : VectorShortFeatureStream(blockLen, nm), src_(src),
    inputLen_(src_->size()), blockLen_(blockLen), shiftLen_(shiftLen),
    overlapLen_(blockLen_ - shiftLen_), curin_(0), curout_(0)
{
  if (blockLen_ < shiftLen_)
    throw jdimension_error("Block length (%d) is less than shift length (%d).\n",
                           blockLen_, shiftLen_);
}

void BlockSizeConversionFeatureShort::inputLonger_()
{
  if (frame_no_ == frame_reset_no_) {
    src_feat_ = src_->next();
    memcpy(vector_->data, src_feat_->data, blockLen_ * sizeof(short));
    curin_ += blockLen_;
    return;
  }

  if (overlapLen_ > 0)
    memmove(vector_->data, vector_->data + shiftLen_, overlapLen_ * sizeof(short));

  if (curin_ + shiftLen_ < inputLen_) {
    memcpy(vector_->data + overlapLen_, src_feat_->data + curin_, shiftLen_ * sizeof(short));
    curin_ += shiftLen_;
  } else {
    int remaining = inputLen_ - curin_;

    if (remaining < 0)
      throw jconsistency_error("Remaining sample (%d) cannot be negative.\n", remaining);

    if (remaining > 0)
      memcpy(vector_->data + overlapLen_, src_feat_->data + curin_, remaining * sizeof(short));

    curin_ = 0; src_feat_ = src_->next();
    unsigned fromNew = shiftLen_ - remaining;

    memcpy(vector_->data + overlapLen_ + remaining, src_feat_->data + curin_, fromNew * sizeof(short));
    curin_ += fromNew;
  }
}

void BlockSizeConversionFeatureShort::outputLonger_()
{
  if (frame_no_ == frame_reset_no_) {
    while (curout_ + inputLen_ <= blockLen_) {
      src_feat_ = src_->next();
      memcpy(vector_->data + curout_, src_feat_->data, inputLen_ * sizeof(short));
      curout_ += inputLen_;
    }

    int remaining = blockLen_ - curout_;
    if (remaining > 0) {
      src_feat_ = src_->next();
      memcpy(vector_->data + curout_, src_feat_->data, remaining * sizeof(short));
      curin_ += remaining;
    }
    curout_ = 0;
    return;
  }

  if (overlapLen_ > 0) {
    memmove(vector_->data, vector_->data + shiftLen_, overlapLen_ * sizeof(short));
    curout_ += overlapLen_;
  }

  if (curin_ > 0) {
    int remaining = inputLen_ - curin_;
    if (remaining > 0) {
      memcpy(vector_->data + curout_, src_feat_->data + curin_, remaining * sizeof(short));
      curout_ += remaining;
    }
    curin_ = 0;
  }

  while (curout_ + inputLen_ <= blockLen_) {
    src_feat_ = src_->next();
    memcpy(vector_->data + curout_, src_feat_->data, inputLen_ * sizeof(short));
    curout_ += inputLen_;
  }

  int remaining = blockLen_ - curout_;
  if (remaining > 0) {
    src_feat_ = src_->next();
    memcpy(vector_->data + curout_, src_feat_->data, remaining * sizeof(short));
    curin_ += remaining;
  }
  curout_ = 0;
}

const gsl_vector_short* BlockSizeConversionFeatureShort::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (inputLen_ > shiftLen_)
    inputLonger_();
  else
    outputLonger_();

  increment_();
  return vector_;
}


// ----- methods for class `PreemphasisFeature' -----
//
PreemphasisFeature::PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp), prior_(0.0), mu_(mu)
{
}

void PreemphasisFeature::reset()
{
  samp_->reset(); VectorFloatFeatureStream::reset();
}

void PreemphasisFeature::next_speaker()
{
  prior_ = 0.0;
}

const gsl_vector_float* PreemphasisFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < size(); i++) {
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(block, i) - mu_ * prior_);
    prior_ = gsl_vector_float_get(block, i);
  }

  return vector_;
}


// ----- methods for class `HammingFeatureShort' -----
//
HammingFeatureShort::HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp), windowLen_(samp->size()),
    window_(new double[windowLen_])
{
  double temp = 2. * M_PI / (double)(windowLen_ - 1);
  for ( unsigned i = 0 ; i < windowLen_; i++ )
    window_[i] = 0.54 - 0.46*cos(temp*i);
}

const gsl_vector_float* HammingFeatureShort::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_short* block = samp_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < windowLen_; i++)
    gsl_vector_float_set(vector_, i, window_[i] * gsl_vector_short_get(block, i));

  return vector_;
}


// ----- methods for class `HammingFeature' -----
//
HammingFeature::HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp), windowLen_(samp->size()),
    window_(new double[windowLen_])
{
  double temp = 2. * M_PI / (double)(windowLen_ - 1);
  for ( unsigned i = 0 ; i < windowLen_; i++ )
    window_[i] = 0.54 - 0.46*cos(temp*i);
}

const gsl_vector_float* HammingFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < windowLen_; i++)
    gsl_vector_float_set(vector_, i, window_[i] * gsl_vector_float_get(block, i));

  return vector_;
}


// ----- methods for class `FFTFeature' -----
//
FFTFeature::FFTFeature(const VectorFloatFeatureStreamPtr& samp, unsigned fftLen, const String& nm)
  : VectorComplexFeatureStream(fftLen, nm), samp_(samp), fftLen_(fftLen), windowLen_(samp_->size()),
#ifdef HAVE_LIBFFTW3
    samples_(static_cast<double*>(fftw_malloc(sizeof(double) * fftLen_))), output_(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (fftLen_ / 2 + 1))))
#else
  samples_(new double[fftLen_])
#endif
{
  for (unsigned i = 0; i < fftLen_; i++)
    samples_[i] = 0.0;

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  fftwPlan_ = fftw_plan_dft_r2c_1d(fftLen_, samples_, output_, FFTW_MEASURE);
#endif
}

FFTFeature::~FFTFeature()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(fftwPlan_);
  fftw_free(samples_);
  fftw_free(output_);
#else
  delete[] samples_;
#endif
}

const gsl_vector_complex* FFTFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < windowLen_; i++)
    samples_[i] = gsl_vector_float_get(block, i);
  for (unsigned i = windowLen_; i < fftLen_; i++)
    samples_[i] = 0.0;

#ifdef HAVE_LIBFFTW3
  fftw_execute(fftwPlan_);
  fftwUnpack(vector_, output_);
#else
  gsl_fft_real_radix2_transform(samples_, /*stride=*/ 1, fftLen_);
  unpack_half_complex(vector_, samples_);
#endif

  return vector_;
}


// ----- methods for class `SpectralPowerFloatFeature' -----
//
SpectralPowerFloatFeature::
SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN, const String& nm) :
  VectorFloatFeatureStream(powN == 0 ? fft->size() : powN, nm), fft_(fft)
{
  if (size() != fft->size() && size() != (fft->size() / 2) + 1)
    throw jconsistency_error("Number of power coefficients %d does not match FFT length %d.",
			     size(), fft->size());
}

const gsl_vector_float* SpectralPowerFloatFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* fftVec = fft_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_float_set(vector_, i, gsl_complex_abs2(gsl_vector_complex_get(fftVec, i)));

  return vector_;
}


// ----- methods for class `SpectralPowerFeature' -----
//
SpectralPowerFeature::
SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN, const String& nm) :
  VectorFeatureStream(powN == 0 ? fft->size() : powN, nm), fft_(fft)
{
  if (size() != fft->size() && size() != (fft->size() / 2) + 1)
    throw jconsistency_error("Number of power coefficients %d does not match FFT length %d.",
			     size(), fft->size());
}

const gsl_vector* SpectralPowerFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_complex* fftVec = fft_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_set(vector_, i, gsl_complex_abs2(gsl_vector_complex_get(fftVec, i)));

  return vector_;
}


// ----- methods for class `SignalPowerFeature' -----
//
const gsl_vector_float* SignalPowerFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  double power = 0.0;
  for (unsigned i = 0; i < samp_->size(); i++) {
    double val = gsl_vector_float_get(block, i);
    power += val * val;
  }
  gsl_vector_float_set(vector_, 0, power / samp_->size() / range_);

  return vector_;
}


// ----- methods for class `ALogFeature' -----
//
const gsl_vector_float* ALogFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  if (minMaxFound_ == false)
    find_min_max_(block);
  float b = max_ / pow(10.0, a_);

  for (unsigned i = 0; i < size(); i++) {
    double val = b + gsl_vector_float_get(block, i);

    if (val <= 0.0) val = 1.0;

    gsl_vector_float_set(vector_, i, m_ * log10(val));
  }

  return vector_;
}

void ALogFeature::reset()
{
  samp_->reset(); VectorFloatFeatureStream::reset();

  if (runon_) return;

  min_ = HUGE; max_ = -HUGE; minMaxFound_ = false;
}

void ALogFeature::find_min_max_(const gsl_vector_float* block)
{
  if (runon_) {
    for (unsigned i = 0; i < block->size; i++) {
      float val = gsl_vector_float_get(block, i);
      if (val < min_) min_ = val;
      if (val > max_) max_ = val;
    }
    return;
  }

  int frame_no = 0;
  while (true) {
    try {
      block = samp_->next(frame_no);
      for (unsigned i = 0; i < block->size; i++) {
        float val = gsl_vector_float_get(block, i);
        if (val < min_) min_ = val;
        if (val > max_) max_ = val;
      }
      frame_no++;
    } catch (jiterator_error& e) {
      minMaxFound_ = true;
      return;
    }
  }
}


// ----- methods for class `NormalizeFeature' -----
//
NormalizeFeature::
NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min, double max, bool runon, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp), min_(min), max_(max), range_(max_ - min_),
    xmin_(HUGE), xmax_(-HUGE), minMaxFound_(false), runon_(runon) { }

const gsl_vector_float* NormalizeFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  /*------------------------------------------------------------
  ;  y = ((x - xmin)/xrange) * yrange + ymin
  ;    =   x * (yrange/xrange)  - xmin*yrange/xrange + ymin
  ;    =   x * factor           - xmin*factor + ymin
  ;-----------------------------------------------------------*/
  if (minMaxFound_ == false) find_min_max_(block);
  double xrange = xmax_ - xmin_;
  double factor = range_ / xrange;
  double add    = min_ - xmin_ * factor;

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(block, i) * factor + add);

  return vector_;
}

void NormalizeFeature::reset()
{
  samp_->reset();  VectorFloatFeatureStream::reset();

  if (runon_) return;

  xmin_ = HUGE; xmax_ = -HUGE; minMaxFound_ = false;
}

void NormalizeFeature::find_min_max_(const gsl_vector_float* block)
{
  if (runon_) {
    for (unsigned i = 0; i < block->size; i++) {
      float val = gsl_vector_float_get(block, i);
      if (val < xmin_) xmin_ = val;
      if (val > xmax_) xmax_ = val;
    }
    return;
  }

  int frame_no = 0;
  while (true) {
    try {
      block = samp_->next(frame_no);
      for (unsigned i = 0; i < block->size; i++) {
	float val = gsl_vector_float_get(block, i);
	if (val < xmin_) xmin_ = val;
	if (val > xmax_) xmax_ = val;
      }
      frame_no++;
    } catch (jiterator_error& e) {
      minMaxFound_ = true;
      return;
    }
  }
}


// ----- methods for class `ThresholdFeature' -----
//
ThresholdFeature::
ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value, double thresh,
		 const String& mode, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), samp_(samp), value_(value), thresh_(thresh)
{
  if (mode == "upper")
    compare_ =  1;
  else if (mode == "lower")
    compare_ = -1;
  else if (mode == "both")
    compare_ =  0;
  else
    throw jkey_error("Mode %s is not supported", mode.c_str());
}

const gsl_vector_float* ThresholdFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < size(); i++) {
    double v = gsl_vector_float_get(block, i);

    if (compare_ > 0) {
      if (v >= thresh_) v = value_;
    } else if (compare_ == 0) {
      if (v >= thresh_) v = value_;
      else if (v <= -thresh_) v = -value_;
    } else if (compare_ < 0) {
      if (v <= thresh_) v = value_;
    }

    gsl_vector_float_set(vector_, i, v);
  }

  return vector_;
}


// ----- methods for class `SpectralResamplingFeature' -----
//
const double SpectralResamplingFeature::SampleRatio = 16.0 / 22.05;

SpectralResamplingFeature::
SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio, unsigned len,
		  const String& nm)
  : VectorFeatureStream((len == 0 ? src->size() : len), nm), src_(src), ratio_(ratio * float(src->size()) / float(size()))
{
  if (ratio_ > 1.0)
    throw jconsistency_error("Must resample the spectrum to a higher rate (ratio = %10.4f < 1.0).",
			     ratio_);
}

SpectralResamplingFeature::~SpectralResamplingFeature() { }

const gsl_vector* SpectralResamplingFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* srcVec = src_->next(frame_no_ + 1);
  increment_();

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {
    float    exact = coeffX * ratio_;
    unsigned low   = unsigned(coeffX * ratio_);
    unsigned high  = low + 1;

    float    wgt   = high - exact;
    float    coeff = wgt * gsl_vector_get(srcVec, low)
      + (1.0 - wgt) * gsl_vector_get(srcVec, high);

    gsl_vector_set(vector_, coeffX, coeff);
  }

  return vector_;
}


#ifdef SRCONV

// ----- methods for class `SamplerateConversionFeature' -----
//
SamplerateConversionFeature::
SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate, unsigned destrate,
			    unsigned len, const String& method, const String& nm)
  : VectorFloatFeatureStream((len == 0 ? unsigned(unsigned(src->size() * float(destrate)/ float(sourcerate))) : len), nm),
    src_(src), dataInSamplesN_(0), dataOutStartX_(0), dataOutSamplesN_(0)
{
  if (method == "best")
    state_ = src_new(SRC_SINC_BEST_QUALITY, 1, &error_);
  else if (method == "medium")
    state_ = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &error_);
  else if (method == "fastest")
    state_ = src_new(SRC_SINC_FASTEST, 1, &error_);
  else if (method == "zoh")
    state_ = src_new(SRC_ZERO_ORDER_HOLD, 1, &error_);
  else if (method == "linear")
    state_ = src_new(SRC_LINEAR, 1, &error_);
  else
    throw jconsistency_error("Cannot recognize type (%s) of sample rate converter.", method.c_str());

  if (state_ == NULL)
    throw j_error("Error while initializing the samplerate converter: %s", src_strerror(error_));

  data_.src_ratio     = float(destrate) / float(sourcerate);
  data_.input_frames  = src_->size();
  data_.output_frames = size();
  data_.end_of_input  = 0;

  data_.data_in  = new float[2 * data_.input_frames];
  data_.data_out = new float[data_.output_frames];
}

SamplerateConversionFeature::~SamplerateConversionFeature()
{
  state_ = src_delete(state_);

  delete[] data_.data_in;
  delete[] data_.data_out;
}

void SamplerateConversionFeature::reset()
{
  src_reset(state_);  src_->reset(); VectorFloatFeatureStream::reset();
}

const gsl_vector_float* SamplerateConversionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  // copy samples remaining in 'data_.data_out' from prior iteration
  unsigned outputBufferSamplesN = 0;
  if (dataOutSamplesN_ > 0) {
    memcpy(vector_->data, data_.data_out + dataOutStartX_, dataOutSamplesN_ * sizeof(float));
    outputBufferSamplesN += dataOutSamplesN_;
    dataOutStartX_ = dataOutSamplesN_ = 0;
  }

  // iterate until '_vector' is full
  while (outputBufferSamplesN < size()) {

    // copy samples from 'src_->next()'
    if (dataInSamplesN_ < src_->size()) {
      const gsl_vector_float* srcVec = src_->next();
      memcpy(data_.data_in + dataInSamplesN_, srcVec->data, src_->size() * sizeof(float));
      dataInSamplesN_ += src_->size();
    }

    // process the current buffer
    error_ = src_process(state_, &data_);

    // copy generated samples to 'vector_'
    unsigned outputSamplesCopied = size() - outputBufferSamplesN;
    if (outputSamplesCopied > data_.output_frames_gen)
      outputSamplesCopied = data_.output_frames_gen;
    memcpy(vector_->data + outputBufferSamplesN, data_.data_out, outputSamplesCopied * sizeof(float));
    outputBufferSamplesN += outputSamplesCopied;
    dataOutStartX_   = outputSamplesCopied;
    dataOutSamplesN_ = data_.output_frames_gen - outputSamplesCopied;

    // copy down remaining samples in 'data_.data_in'
    dataInSamplesN_ -= data_.input_frames_used;
    memmove(data_.data_in, data_.data_in + data_.input_frames_used, dataInSamplesN_ * sizeof(float));
  }

  increment_();
  return vector_;
}

#endif // SRCONV

// ----- methods for class 'VTLNFeature' -----
//
VTLNFeature::VTLNFeature(const VectorFeatureStreamPtr& pow,
			 unsigned coeffN, double ratio, double edge, int version, 
			 const String& nm)
  : VectorFeatureStream(coeffN == 0 ? pow->size() : coeffN, nm), pow_(pow), ratio_(ratio),
    edge_(edge), version_(version)
{
  auxV_ = gsl_vector_alloc(coeffN);
}


const gsl_vector* VTLNFeature::nextOrg(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* powVec = pow_->next(frame_no_ + 1);
  increment_();

  double yedge = (edge_ < ratio_) ? (edge_ / ratio_)              : 1.0;
  double b     = (yedge < 1.0)    ? (1.0 - edge_) / (1.0 - yedge) : 0;

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {

    double Y0 = double(coeffX)   / double(size());
    double Y1 = double(coeffX+1) / double(size());

    double X0 = ((Y0 < yedge) ? (ratio_ * Y0) :
		 (b     * Y0 +  1.0 - b)) * size();
    double X1 = ((Y1 < yedge) ? (ratio_ * Y1) :
		 (b     * Y1 +  1.0 - b)) * size();

    int    Lower_coeffY1 = int(X1);
    double alpha1        = X1 - Lower_coeffY1;

    int    Lower_coeffY0 = int(X0);
    double alpha0        = int(X0) + 1 - X0;

    double z             =  0.0;

    if (Lower_coeffY0 >= powVec->size)
      Lower_coeffY0 = powVec->size - 1;
         
    if (Lower_coeffY1 > powVec->size)
      Lower_coeffY1 = powVec->size;

    if ( Lower_coeffY0 == Lower_coeffY1) {
      z += (X1-X0) * gsl_vector_get(powVec, Lower_coeffY0);
    } else {
      z += alpha0  * gsl_vector_get(powVec, Lower_coeffY0);

      for (int i = Lower_coeffY0+1; i < Lower_coeffY1; i++)
	z += gsl_vector_get(powVec, i);

      if ( Lower_coeffY1 < int(powVec->size))
	z += alpha1 * gsl_vector_get(powVec, Lower_coeffY1);
    }

    gsl_vector_set(vector_, coeffX, z);
  }

  return vector_;
}

const gsl_vector* VTLNFeature::nextFF(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* powVec = pow_->next(frame_no_ + 1);
  increment_();

  unsigned N	= size();
  float b	= N * edge_;
  float slope1	= ratio_;
  float slope2	= ratio_;
  if (slope1 < 1.0)
    slope2 = (N - slope1 * b) / (N - b);

  gsl_vector_set_zero(vector_);
  gsl_vector_set_zero(auxV_);

  for (int sIdx = 0; sIdx < N; sIdx++) {
    float sIdx1	= sIdx - 0.5;
    float sIdx2	= sIdx + 0.5;
    float v	= gsl_vector_get(powVec, sIdx);
    float dIdx1 = sIdx1*slope1;
    if (sIdx1 > b)
      dIdx1 = b * slope1 + (sIdx1 - b) * slope2;
    float dIdx2 = sIdx2*slope1;
    if (sIdx2 > b)
      dIdx2 = b * slope1 + (sIdx2 - b) * slope2;

    int i1 = int(floor(dIdx1));
    int i2 = int(ceil(dIdx2));

    if (i1<=N-1) {
      double alpha = 1.0;
      double alpha1 = (1.0 - (dIdx1 - i1)) * alpha;
      double alpha2 = (i2  - dIdx2) * alpha;

      for (int j = i1; j <= i2; j++) {
        int k = j;
        if (k < 0)
          k = 0;
        if (k >= N)
          break;

        double a = alpha;
        if (j == i1)
          a = alpha1;
        if (j == i2)
          a = alpha2;

        gsl_vector_set(vector_, k, gsl_vector_get(vector_, k) + a * v);
        gsl_vector_set(auxV_, k, gsl_vector_get(auxV_, k) + a);
      }
    }
  }

  for (unsigned i = 0; i < N; i++) {
    double norm = gsl_vector_get(auxV_, i);
    if (norm > 1E-20)
      gsl_vector_set(vector_, i, gsl_vector_get(vector_, i)/norm);
  }

  return vector_;
}

const gsl_vector* VTLNFeature::next(int frame_no) {
  switch(version_) {
    case 1:
      return nextOrg(frame_no);
    case 2:
      return nextFF(frame_no);
    default:
      throw jtype_error("[ERROR] VTLNFeature::next >> unknown version number (%d)", version_);
  }
}

void VTLNFeature::matrix(gsl_matrix* mat) const
{
  if (mat->size1 != size() || mat->size2 != size())
    throw jdimension_error("Matrix (%d x %d) does not match (%d x %d)",
                           mat->size1, mat->size2, size(), size());

  gsl_matrix_set_zero(mat);

  double yedge = (edge_ < ratio_) ? (edge_ / ratio_)              : 1.0;
  double b     = (yedge < 1.0)    ? (1.0 - edge_) / (1.0 - yedge) : 0;

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {

    double Y0 = double(coeffX)   / double(size());
    double Y1 = double(coeffX+1) / double(size());

    double X0 = ((Y0 < yedge) ? (ratio_ * Y0) :
		 (b     * Y0 +  1.0 - b)) * size();
    double X1 = ((Y1 < yedge) ? (ratio_ * Y1) :
		 (b     * Y1 +  1.0 - b)) * size();

    int    Lower_coeffY1 = int(X1);
    double alpha1        = X1 - Lower_coeffY1;

    int    Lower_coeffY0 = int(X0);
    double alpha0        = int(X0) + 1 - X0;

    if (Lower_coeffY0 >= mat->size1)
      Lower_coeffY0 = mat->size1 - 1;

    if (Lower_coeffY1 > mat->size1)
      Lower_coeffY1 = mat->size1;

    if ( Lower_coeffY0 == Lower_coeffY1) {
      gsl_matrix_set(mat, coeffX, Lower_coeffY0, X1-X0);
    } else {
      gsl_matrix_set(mat, coeffX, Lower_coeffY0, alpha0);

      for (int i = Lower_coeffY0+1; i < Lower_coeffY1; i++)
	gsl_matrix_set(mat, coeffX, i, 1.0);

      if ( Lower_coeffY1 < int(size()))
	gsl_matrix_set(mat, coeffX, Lower_coeffY1, alpha1);	
    }
  }
}


// ----- methods for class `MelFeature::SparseMatrix_' -----
//
MelFeature::SparseMatrix_::SparseMatrix_(unsigned m, unsigned n, unsigned version)
  : data_(new float*[m]), m_(m), n_(n), offset_(new unsigned[m]), coefN_(new unsigned[m]), version_(version)
{
  for (unsigned i = 0; i < m_; i++) {
    data_[i]  = NULL;  offset_[i] = coefN_[i] = 0;
  }
}

MelFeature::SparseMatrix_::~SparseMatrix_()
{
  dealloc_();
}

void MelFeature::SparseMatrix_::alloc_(unsigned m, unsigned n)
{
  dealloc_();
  m_ = m;  n_ = n;

  data_   = new float*[m_];
  offset_ = new unsigned[m_];
  coefN_  = new unsigned[m_];

  for (unsigned i = 0; i < m_; i++) {
    data_[i]  = NULL;  offset_[i] = coefN_[i] = 0;
  }
}

void MelFeature::SparseMatrix_::dealloc_()
{
  for (unsigned i = 0; i < m_; i++)
    delete[] data_[i];

  delete[] data_;    data_   = NULL;
  delete[] offset_;  offset_ = NULL;
  delete[] coefN_;   coefN_  = NULL;
}

float MelFeature::SparseMatrix_::mel_(float hz)
{
   if (hz>=0) return (float)(2595.0 * log10(1.0 + (double)hz/700.0));
   else return 0.0;
}

float MelFeature::SparseMatrix_::hertz_(float m)
{
   double d = m / 2595.0;
   return (float)(700.0 * (pow(10.0,d) - 1.0));
}

void MelFeature::SparseMatrix_::melScaleOrg(int powN,  float rate, float low, float up, int filterN)
{
  float df = rate / (4.0 * (powN/2));   /* spacing between FFT points in Hz */
  float mlow = mel_(low);
  float mup  = mel_(up);
  float dm   = (mup - mlow)/(filterN+1);    /* delta mel */

  if (low<0.0 || 2.0*up>rate || low>up)
    throw j_error("mel: something wrong with\n");

  /* printf("lower = %fHz (%fmel), upper = %fHz (%fmel)\n",low,mlow,up,mup);*/

  /* -------------------------------------------
     free band matrix, allocate filterN pointer
     ------------------------------------------- */
  if (data_) {
    for (unsigned i = 0; i < m_; i++)
      delete[] data_[i];
    delete[] data_;
  }
  delete[] offset_;  delete[] coefN_;

  data_   = new float*[filterN];
  coefN_  = new unsigned[filterN];
  offset_ = new unsigned[filterN];

  /* ---------------------------
     loop over all filters
     --------------------------- */
  for (unsigned x = 0; x < filterN; x++) {

    /* ---- left, center and right edge ---- */
    float left   = hertz_( x     *dm + mlow);
    float center = hertz_((x+1.0)*dm + mlow);
    float right  = hertz_((x+2.0)*dm + mlow);
    /* printf("%3d: left = %fmel, center = %fmel, right = %fmel\n",
       x,x*dm+mlow,(x+1.0)*dm+mlow,(x+2.0)*dm+mlow); */
    /* printf("%3d: left = %fHz, center = %fHz, right = %fHz\n",
       x,left,center,right); */
      
    float height = 2.0 / (right - left);          /* normalized height = 2/width */
    float slope1 = height / (center - left);
    float slope2 = height / (center - right);
    int start    = (int)ceil(left / df);
    int end      = (int)floor(right / df);
      
    offset_[x] = start;
    coefN_[x]  = end - start + 1;
    n_         = end;
    data_[x] = new float[coefN_[x]];
    float freq=start*df;
    for (unsigned i=0; i < coefN_[x]; i++) {
      freq += df;
      if (freq <= center)
	data_[x][i] = slope1*(freq-left);
      else
	data_[x][i] = slope2*(freq-right);
    }
  }
  rate_ = rate;
}

void MelFeature::SparseMatrix_::melScaleFF(int powN,  float rate, float low, float up, int filterN)
{
  float df = rate / (4.0 * (powN/2));   /* spacing between FFT points in Hz */
  float mlow = mel_(low);
  float mup  = mel_(up);
  float dm   = (mup - mlow)/(filterN+1);    /* delta mel */

  if (low<0.0 || 2.0*up>rate || low>up)
    throw j_error("mel: something wrong with\n");

  /* printf("lower = %fHz (%fmel), upper = %fHz (%fmel)\n",low,mlow,up,mup);*/

  /* -------------------------------------------
     free band matrix, allocate filterN pointer
     ------------------------------------------- */
  if (data_) {
    for (unsigned i = 0; i < m_; i++)
      delete[] data_[i];
    delete[] data_;
  }
  delete[] offset_;  delete[] coefN_;

  data_   = new float*[filterN];
  coefN_  = new unsigned[filterN];
  offset_ = new unsigned[filterN];

  /* ---------------------------
     loop over all filters
     --------------------------- */
  for (unsigned x = 0; x < filterN; x++) {

    /* ---- left, center and right edge ---- */
    float left   = hertz_( x     *dm + mlow);
    float center = hertz_((x+1.0)*dm + mlow);
    float right  = hertz_((x+2.0)*dm + mlow);
    /* printf("%3d: left = %fmel, center = %fmel, right = %fmel\n",
       x,x*dm+mlow,(x+1.0)*dm+mlow,(x+2.0)*dm+mlow); */
    /* printf("%3d: left = %fHz, center = %fHz, right = %fHz\n",
       x,left,center,right); */

    float height = 2.0 / (right - left);          /* normalized height = 2/width */
    float slope1 = height / (center - left);
    float slope2 = height / (center - right);
    int start    = (int)ceil(left / df);
    int end      = (int)floor(right / df);

    offset_[x] = start;
    coefN_[x]  = end - start + 1;
    n_         = end;
    data_[x] = new float[coefN_[x]];
    float freq=start*df;
    // FF Print MFB --- START
    /*for (unsigned j=0; j<start; j++)
      printf("%f ", 0.0);*/
    // FF Print MFB --- END
    for (unsigned i=0; i < coefN_[x]; i++) {
      //freq += df; // better don't put it here FF fix
      if (freq <= center)
        data_[x][i] = slope1*(freq-left);
      else
        data_[x][i] = slope2*(freq-right);
      // FF Print MFB --- START
      //printf("%f ", data_[x][i]);
      // FF Print MFB --- END
      freq += df; // instead put it here FF fix
    }
    // FF Print MFB --- START
    /*for (unsigned j=end; j<powN; j++)
      printf("%f ", 0.0);
    printf("\r\n");*/
    // FF Print MFB --- END
  }
  rate_ = rate;
}

void MelFeature::SparseMatrix_::melScale(int powN,  float rate, float low, float up, int filterN)
{
  switch (version_) {
    case 1:
      melScaleOrg(powN,  rate, low, up, filterN);
    break;
    case 2:
      melScaleFF(powN,  rate, low, up, filterN);
    break;
    default:
      throw jtype_error("[ERROR] MelFeature::SparseMatrix_::fmatrixBMulot >> Unknown Version.\r\n");
  }
}

gsl_vector* MelFeature::SparseMatrix_::fmatrixBMulotOrg( gsl_vector* C, const gsl_vector* A) const
{
  if (C == A)
     throw jconsistency_error("matrix multiplication: result matrix must be different!\n");

  if ( int(A->size) < n_)   // n_ can be smaller
     throw jconsistency_error("Matrix columns differ: %d and %d.\n",
            A->size, n_);

  // fmatrixResize(C,A->m,m_);
  for (unsigned j = 0; j < C->size; j++) {
    double  sum = 0.0;
    double*  aP  = A->data + offset_[j];
    double*  eP  = A->data + offset_[j] + coefN_[j] - 3;
    float*   bP  = data_[j];
    while (aP < eP) {
      sum += aP[0]*bP[0] + aP[1]*bP[1] + aP[2]*bP[2] + aP[3]*bP[3];
      aP += 4; bP += 4;
    }
    eP += 3;
    while (aP < eP) sum += *(aP++) * *(bP++);
    gsl_vector_set(C, j, sum);
  }

  return C;
}

gsl_vector* MelFeature::SparseMatrix_::fmatrixBMulotFF( gsl_vector* C, const gsl_vector* A) const
{
  if (C == A)
     throw jconsistency_error("matrix multiplication: result matrix must be different!\n");

  if ( int(A->size) < n_)		// n_ can be smaller
     throw jconsistency_error("Matrix columns differ: %d and %d.\n",
			      A->size, n_);

  // fmatrixResize(C,A->m,m_);
  for (unsigned j = 0; j < C->size; j++) {
    double  sum = 0.0;
    double*  aP  = A->data + offset_[j];
    //double*  eP  = A->data + offset_[j] + coefN_[j] - 3; // "are you serious about this" FF fix
    double*  eP  = A->data + offset_[j] + (coefN_[j] & ~3); // "better do this" FF fix
    float*   bP  = data_[j];
    while (aP < eP) {
      sum += aP[0]*bP[0] + aP[1]*bP[1] + aP[2]*bP[2] + aP[3]*bP[3];
      aP += 4; bP += 4;
    }
    //eP += 3; // "are you serious about this" FF fix - part 2
    eP += (coefN_[j] & 3); // "better do this" FF fix - part 2
    while (aP < eP) sum += *(aP++) * *(bP++);
    gsl_vector_set(C, j, sum);
  }

  return C;
}

void MelFeature::SparseMatrix_::fmatrixBMulot(gsl_vector* C, const gsl_vector* A) const {
  switch (version_) {
  case 1:
    fmatrixBMulotOrg(C, A);
    break;
  case 2:
    fmatrixBMulotFF(C, A);
    break;
  default:
    throw jtype_error("[ERROR] MelFeature::SparseMatrix_::fmatrixBMulot >> Unknown Version.\r\n");
  }
}

void MelFeature::SparseMatrix_::readBuffer(const String& fb)
{
  std::list<String> scratch;
  split_list(fb, scratch);

  std::list<String>::iterator sitr = scratch.begin();
  for (unsigned i = 0; i < 2; i++)
    sitr++;

  std::list<String> rows;
  split_list(*sitr, rows);

  unsigned i = 0;
  for (std::list<String>::iterator itr = rows.begin(); itr != rows.end(); itr++) {

    // cout << "All " << *itr  << endl;

    std::list<String> values;
    split_list((*itr), values);

    if (i == 0)
      alloc_(rows.size(), values.size() - 1);

    // --- scan <offset> ---
    std::list<String>::iterator jitr = values.begin();
    if ( values.size() < 1 || sscanf(*jitr, "%d", &(offset_[i])) != 1) {
      dealloc_();
      throw jconsistency_error("expected integer value for <offset> not: %s\n", (*jitr).c_str());
    }

    // --- How many coefficients? Allocate memory! ---
    coefN_[i]      = values.size() - 1;
    unsigned maxN  = offset_[i] + coefN_[i];
    if (n_ < maxN) n_ = maxN;

    data_[i] = new float[coefN_[i]];

    if (data_[i] == NULL) {
      dealloc_();
      throw jallocation_error("could not allocate float band matrix");
    }

    // --- scan <coef0> <coef1> .. ---
    unsigned j = 0;
    for (jitr++; jitr != values.end(); jitr++) {

      // cout << "Value " << *jitr  << endl;

      float d;
      if ( sscanf((*jitr).c_str(), "%f", &d) != 1) {
	dealloc_();
	throw jconsistency_error("expected 'float' type elements.\n");
      }
      data_[i][j] = d;
      j++;
    }
    i++;
  }
}

void MelFeature::SparseMatrix_::matrix(gsl_matrix* mat) const
{
  if (mat->size1 != m_ || mat->size2 != n_)
    throw jdimension_error("Matrix (%d x %d) does not match (%d x %d)",
                           mat->size1, mat->size2, m_, n_);

  gsl_matrix_set_zero(mat);
  for (unsigned i = 0; i < m_; i++)
    for (int j = 0; j < coefN_[i]; j++)
      gsl_matrix_set(mat, i, j + offset_[i], data_[i][j]);
}


// ----- methods for class `MelFeature' -----
//
MelFeature::MelFeature(const VectorFeatureStreamPtr& mag, int powN,
                       float rate, float low, float up,
                       unsigned filterN, unsigned version, const String& nm)
  : VectorFeatureStream(filterN, nm), nmel_(filterN),
    powN_((powN == 0) ? mag->size() : powN),
    mag_(mag), mel_(0, 0, version)
{
  if (up <= 0) up = rate/2.0;
  mel_.melScale(powN_, rate, low, up, nmel_);
}

MelFeature::~MelFeature()
{
}

void MelFeature::read(const String& fileName)
{
  static size_t n      = 0;
  static char*  buffer = NULL;

  FILE* fp = btk_fopen(fileName, "r");
  getline(&buffer, &n, fp);
  btk_fclose(fileName, fp);

  // cout << "Buffer " << buffer << endl;

  mel_.readBuffer(buffer);
  free(buffer);
}

const gsl_vector* MelFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* fftVec = mag_->next(frame_no_ + 1);
  increment_();

  mel_.fmatrixBMulot(vector_, fftVec);

  return vector_;
}


// ----- methods for class `SphinxMelFeature' -----
//
SphinxMelFeature::SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN, unsigned powerN,
				   float sampleRate, float lowerF, float upperF,
				   unsigned filterN, const String& nm)
  : VectorFeatureStream(filterN, nm), fftN_(fftN), filterN_(filterN),
    powN_((powerN == 0) ? mag->size() : powerN), samplerate_(sampleRate),
    mag_(mag), filters_(gsl_matrix_calloc(filterN_, powN_))
{
  double dfreq = samplerate_ / fftN_;
  if (upperF > samplerate_ / 2)
    throw j_error("Upper frequency %f exceeds Nyquist %f", upperF, sampleRate / 2.0);

  double melmax = melFrequency_(upperF);
  double melmin = melFrequency_(lowerF);
  double dmelbw = (melmax - melmin) / (filterN_ + 1);

  // Filter edges, in Hz
  gsl_vector* edges(gsl_vector_calloc(filterN_ + 2));
  for (unsigned n = 0; n < filterN_ + 2; n++)
    gsl_vector_set(edges, n, melInverseFrequency_(melmin + dmelbw * n));

  // Set filter triangles, in DFT points
  for (unsigned filterX = 0; filterX < filterN_; filterX++) {
    const double left_freq	= gsl_vector_get(edges, filterX);
    const double center_freq	= gsl_vector_get(edges, filterX + 1);
    const double right_freq	= gsl_vector_get(edges, filterX + 2);

    unsigned min_k = 999999;
    unsigned max_k = 0;

    for (unsigned k = 1; k < powN_; k++) {
      double hz			= k * dfreq;
      if (hz < left_freq) continue;
      if (hz > right_freq) break;
      double left_value		= (hz - left_freq) / (center_freq - left_freq);
      double right_value	= (right_freq - hz) / (right_freq - center_freq);
      double filter_value	=  min(left_value, right_value);
      gsl_matrix_set(filters_, filterX, k, filter_value);
      min_k = min(k, min_k);
      max_k = max(k, max_k);
      boundaries_.push_back(Boundary_(min_k, max_k));
    }
  }
  gsl_vector_free(edges);
}

SphinxMelFeature::~SphinxMelFeature()
{
  gsl_matrix_free(filters_);
}

double SphinxMelFeature::melFrequency_(double frequency)
{
  return (2595.0 * log10(1.0 + (frequency / 700.0)));
}

double SphinxMelFeature::melInverseFrequency_(double mel)
{
  return (700.0 * (pow(10.0, mel / 2595.0) - 1.0));
}


const gsl_vector* SphinxMelFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* fftVec = mag_->next(frame_no_ + 1);
  increment_();

  gsl_blas_dgemv(CblasNoTrans, 1.0, filters_, fftVec, 0.0, vector_);

  return vector_;
}


// ----- methods for class `LogFeature' -----
//
LogFeature::LogFeature(const VectorFeatureStreamPtr& mel, double m, double a, bool sphinxFlooring,
                       const String& nm) :
  VectorFloatFeatureStream(mel->size(), nm),
  nmel_(mel->size()), mel_(mel), m_(m), a_(a), SphinxFlooring_(sphinxFlooring)
{}

const gsl_vector_float* LogFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector* melVec = mel_->next(frame_no_ + 1);
  increment_();

  unsigned err = 0;
  for (unsigned i = 0; i < nmel_; i++) {
    double val = gsl_vector_get(melVec, i);
    if (SphinxFlooring_) {

      static const double floor_ = 1.0E-05;
      if (val < floor_) {
        val = floor_;  err++;
      }
    } else {
      val += a_;
      if (val <= 0.0) {
        val = 1.0;  err++;
      }
    }

    gsl_vector_float_set(vector_, i, m_ * log10(val));
  }

  return vector_;
}

// ----- methods for class `CepstralFeature' -----
//
// type:
//   0 = Type 1 DCT ? 
//   1 = Type 2 DCT ? 
//   2 = Sphinx Legacy
CepstralFeature::CepstralFeature(const VectorFloatFeatureStreamPtr& mel,
				 unsigned ncep, int type, const String& nm) :
  VectorFloatFeatureStream(ncep, nm),
  cos_(gsl_matrix_float_calloc(ncep, mel->size())), mel_(mel)
{
  if (type == 0) {
    cout << "Using DCT Type 1." << endl;
    gsl_matrix_float_set_cosine(cos_, ncep, mel->size(), type);
  } else if (type == 1) {
    cout << "Using DCT Type 2." << endl;
    gsl_matrix_float_set_cosine(cos_, ncep, mel->size(), type);
  } else if (type == 2) {
    cout << "Using Sphinx legacy DCT." << endl;
    sphinxLegacy_();
  } else {
    throw jindex_error("Unknown DCT type\n");
  }
}

void CepstralFeature::sphinxLegacy_()
{
  for (unsigned cepstraX = 0; cepstraX < size(); cepstraX++) {
    double deltaF = M_PI * float(cepstraX) / mel_->size();
    for (unsigned filterX = 0; filterX < mel_->size(); filterX++) {
      double frequency = deltaF * (filterX + 0.5);
      double c	       = cos(frequency) / mel_->size();
      if (filterX == 0) c *= 0.5;
      gsl_matrix_float_set(cos_, cepstraX, filterX, c);
    }
  }
}

const gsl_vector_float* CepstralFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* logVec = mel_->next(frame_no_ + 1);
  increment_();

  gsl_blas_sgemv(CblasNoTrans, 1.0, cos_, logVec, 0.0, vector_);

  return vector_;
}

gsl_matrix* CepstralFeature::matrix() const
{
  cout << "Allocating DCT Matrix (" << size() << " x " << mel_->size() << ")" << endl;

  gsl_matrix* matrix = gsl_matrix_calloc(size(), mel_->size());

  for(unsigned i=0;i<matrix->size1;i++)
    for(unsigned j=0;j<matrix->size2;j++)
      gsl_matrix_set(matrix, i,j, gsl_matrix_float_get(cos_,i,j) );

  return matrix;
}

// ----- methods for class `FloatToDoubleConversionFeature' -----
//
const gsl_vector* FloatToDoubleConversionFeature::next(int frame_no) {
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* srcVec = src_->next(frame_no_ + 1);
  increment_();

  for (unsigned i=0; i<size(); i++)
    gsl_vector_set(vector_, i, gsl_vector_float_get(srcVec, i));

  return vector_;

}

// ----- methods for class `MeanSubtractionFeature' -----
//
const float     MeanSubtractionFeature::variance_floor_  = 0.0001;
const float	MeanSubtractionFeature::before_wgt_      = 0.98;
const float	MeanSubtractionFeature::after_wgt       = 0.995;
const unsigned	MeanSubtractionFeature::framesN2change_ = 500;

MeanSubtractionFeature::
MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight,
		       double devNormFactor, bool runon, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    src_(src), wgt_(weight),
    mean_(gsl_vector_float_calloc(size())), var_(gsl_vector_float_calloc(size())),
    devNormFactor_(devNormFactor), framesN_(0), runon_(runon), mean_var_found_(false)
{
  gsl_vector_float_set_zero(mean_);
  gsl_vector_float_set_all(var_, 1.0);
}

MeanSubtractionFeature::~MeanSubtractionFeature()
{
  gsl_vector_float_free(mean_);
  gsl_vector_float_free(var_);
}

void MeanSubtractionFeature::write(const String& fileName, bool variance) const
{
  if (frame_no_ <= 0)
    throw jio_error("Frame count must be > 0.\n", frame_no_);

  FILE* fp = btk_fopen(fileName, "w");

  const gsl_vector_float* vec = (variance) ? var_ : mean_;
  gsl_vector_float_fwrite(fp, vec);

  btk_fclose(fileName, fp);
}

const gsl_vector_float* MeanSubtractionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  if (runon_)
    return nextRunon_(frame_no);
  else
    return nextBatch_(frame_no);
}

const gsl_vector_float* MeanSubtractionFeature::nextRunon_(int frame_no)
{
  const gsl_vector_float* srcVec = src_->next(frame_no_ + 1);
  float 		  wgt    = 1.0;
  if (wgt_.is_null() == false) {
    const gsl_vector_float* wgtVec = wgt_->next(frame_no_ + 1);
     		            wgt    = gsl_vector_float_get(wgtVec, 0);
  }
  increment_();

  if (wgt > 0.0) {
    // update mean
    float wgt  = (framesN_ < framesN2change_) ? before_wgt_ : after_wgt;
    for (unsigned i = 0; i < size(); i++) {
      float f    = gsl_vector_float_get(srcVec, i);
      float m    = gsl_vector_float_get(mean_, i);

      float comp = wgt * m + (1.0 - wgt) * f;
      gsl_vector_float_set(mean_, i, comp);
    }

    // update square mean
    if (devNormFactor_ > 0.0) {
      for (unsigned i = 0; i < size(); i++) {
	float f    = gsl_vector_float_get(srcVec, i);
	float m    = gsl_vector_float_get(mean_, i);
	float diff = f - m;

	float sm   = gsl_vector_float_get(var_, i);

	float comp = wgt * sm + (1.0 - wgt) * (diff * diff);
	gsl_vector_float_set(var_, i, comp);
      }
    }

    framesN_++;
  }

  normalize_(srcVec);

  return vector_;
}

const gsl_vector_float* MeanSubtractionFeature::nextBatch_(int frame_no)
{
  if (mean_var_found_ == false)
    calcMeanVariance_();

  const gsl_vector_float* srcVec = src_->next(frame_no_ + 1);
  increment_();

  normalize_(srcVec);

  return vector_;
}

void MeanSubtractionFeature::calcMeanVariance_()
{
  int    frame_no = 0;
  double ttlWgt = 0.0;
  gsl_vector_float_set_zero(mean_);
  while (true) {
    try {
      const gsl_vector_float* srcVec = src_->next(frame_no);
      float                   wgt    = 1.0;
      if (wgt_.is_null() == false) {
        const gsl_vector_float* wgtVec = wgt_->next(frame_no);
        wgt    = gsl_vector_float_get(wgtVec, 0);
      }

      // printf("Frame %4d : Weight %6.2f\n", frame_no, wgt);

      // sum for mean
      for (unsigned i = 0; i < size(); i++) {
        float f = gsl_vector_float_get(srcVec, i);
        float m = gsl_vector_float_get(mean_, i);

        gsl_vector_float_set(mean_, i, m + wgt * f);
      }
      frame_no++;  ttlWgt += wgt;
    } catch (jiterator_error& e) {
      for (unsigned i = 0; i < size(); i++)
	gsl_vector_float_set(mean_, i, gsl_vector_float_get(mean_, i) / ttlWgt);
      break;
    } catch (j_error& e) {
      if (e.getCode() == JITERATOR) {
        for (unsigned i = 0; i < size(); i++)
          gsl_vector_float_set(mean_, i, gsl_vector_float_get(mean_, i) / ttlWgt);
        break;
      }
    }
  }

  frame_no = 0;  ttlWgt = 0.0;
  gsl_vector_float_set_zero(var_);
  while (true) {
    try {
      const gsl_vector_float* srcVec = src_->next(frame_no);
      float                   wgt    = 1.0;
      if (wgt_.is_null() == false) {
        const gsl_vector_float* wgtVec = wgt_->next(frame_no);
        wgt    = gsl_vector_float_get(wgtVec, 0);
      }

      // sum for covariance
      for (unsigned i = 0; i < size(); i++) {
        float f = gsl_vector_float_get(srcVec, i);
        float v = gsl_vector_float_get(var_, i);

        gsl_vector_float_set(var_, i, v + wgt * f * f);
      }
      frame_no++;  ttlWgt += wgt;
    } catch (jiterator_error& e) {
      for (unsigned i = 0; i < size(); i++) {
        float m = gsl_vector_float_get(mean_, i);
        gsl_vector_float_set(var_, i, (gsl_vector_float_get(var_, i) / ttlWgt) - (m * m));
      }
      break;
    } catch (j_error& e) {
      if (e.getCode() == JITERATOR) {
        for (unsigned i = 0; i < size(); i++) {
          float m = gsl_vector_float_get(mean_, i);
          gsl_vector_float_set(var_, i, (gsl_vector_float_get(var_, i) / ttlWgt) - (m * m));
        }
        break;
      }

    }
  }
  mean_var_found_ = true;
}

void MeanSubtractionFeature::normalize_(const gsl_vector_float* srcVec)
{
  // subtract mean
  for (unsigned i = 0; i < size(); i++) {
    float f     = gsl_vector_float_get(srcVec, i);
    float m     = gsl_vector_float_get(mean_, i);

    gsl_vector_float_set(vector_, i, f - m);
  }

  // normalize standard deviation
  if (devNormFactor_ > 0.0) {
    for (unsigned i = 0; i < size(); i++) {
      float f   = gsl_vector_float_get(vector_, i);
      float var = gsl_vector_float_get(var_, i);

      if (var < variance_floor_) var = variance_floor_;

      gsl_vector_float_set(vector_, i, f / (devNormFactor_ * sqrt(var)));
    }
  }
}

void MeanSubtractionFeature::reset()
{
  src_->reset();  VectorFloatFeatureStream::reset();  mean_var_found_ = false;

  if (wgt_.is_null() == false) wgt_->reset();
}

void MeanSubtractionFeature::next_speaker()
{
  framesN_ = 0;
  gsl_vector_float_set_zero(mean_);
  gsl_vector_float_set_all(var_, 1.0);
}


// ----- methods for class `FileMeanSubtractionFeature' -----
//
const float FileMeanSubtractionFeature::variance_floor_ = 0.0001;

FileMeanSubtractionFeature::
FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, double devNormFactor, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    src_(src), mean_(gsl_vector_float_calloc(size())), variance_(NULL), devNormFactor_(devNormFactor)
{
  gsl_vector_float_set_zero(mean_);
}

FileMeanSubtractionFeature::~FileMeanSubtractionFeature()
{
  gsl_vector_float_free(mean_);
  if (variance_ != NULL)
    gsl_vector_float_free(variance_);
}

const gsl_vector_float* FileMeanSubtractionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* srcVec = src_->next(frame_no_ + 1);
  increment_();

  // subtract mean
  for (unsigned i = 0; i < size(); i++) {
    float f = gsl_vector_float_get(srcVec, i);
    float m = gsl_vector_float_get(mean_, i);

    float comp = f - m;
    gsl_vector_float_set(vector_, i, comp);
  }

  // normalize standard deviation
  if (variance_ != NULL && devNormFactor_ > 0.0) {
    for (unsigned i = 0; i < size(); i++) {
      float f   = gsl_vector_float_get(vector_, i);
      float var = gsl_vector_float_get(variance_, i);

      if (var < variance_floor_) var = variance_floor_;

      gsl_vector_float_set(vector_, i, f / (devNormFactor_ * sqrt(var)));
    }
  }

  return vector_;
}

void FileMeanSubtractionFeature::reset()
{
  src_->reset();
  VectorFloatFeatureStream::reset();
}

void FileMeanSubtractionFeature::read(const String& fileName, bool variance)
{
  FILE* fp = btk_fopen(fileName, "r");
  if (variance == false) {

    printf("Loading mean from '%s'.\n", fileName.c_str());
    if (mean_->size != int(size())) {
      btk_fclose(fileName, fp);
      throw jdimension_error("Feature and mean do not have same size (%d vs. %d).\n", mean_->size, size());
    }
    gsl_vector_float_fread(fp, mean_);

  } else {

    if (variance_ == NULL)
      variance_ = gsl_vector_float_calloc(size());
    printf("Loading covariance from '%s'.\n", fileName.c_str());
    if (variance_->size != int(size())) {
      btk_fclose(fileName, fp);
      throw jdimension_error("Feature and covariance do not have same size (%d vs. %d).\n", variance_->size, size());
    }
    gsl_vector_float_fread(fp, variance_);

  }
  btk_fclose(fileName, fp);
}


// ----- methods for class `AdjacentFeature' -----
//
AdjacentFeature::
AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta,
		const String& nm) :
  VectorFloatFeatureStream((2 * delta + 1) * single->size(), nm), delta_(delta),
  single_(single), singleSize_(single->size()),
  plen_(2 * delta_ * singleSize_), framesPadded_(0)
{
}

void AdjacentFeature::buffer_next_frame_(int frame_no)
{
  if (frame_no == 0) {				// initialize the buffer
    const gsl_vector_float* singVec = single_->next(/* featX = */ 0);
    for (unsigned featX = 1; featX <= delta_ + 1; featX++) {
      unsigned offset = featX * singleSize_;
      for (unsigned sampX = 0; sampX < singleSize_; sampX++)
	gsl_vector_float_set(vector_, offset + sampX, gsl_vector_float_get(singVec, sampX));
    }

    for (unsigned featX = 1; featX < delta_; featX++) {
      singVec = single_->next(featX);
      unsigned offset = (featX + delta_ + 1) * singleSize_;
      for (unsigned sampX = 0; sampX < singleSize_; sampX++)
	gsl_vector_float_set(vector_, offset + sampX, gsl_vector_float_get(singVec, sampX));
    }
  }

  // slide down the old values
  for (unsigned sampX = 0; sampX < plen_; sampX++)
    gsl_vector_float_set(vector_, sampX, gsl_vector_float_get(vector_, singleSize_ + sampX));

  if (framesPadded_ == 0) { // normal processing
    try {
      const gsl_vector_float* singVec = single_->next(frame_no + delta_);
      unsigned offset = 2 * delta_ * singleSize_;
      for (unsigned sampX = 0; sampX < singleSize_; sampX++)
	gsl_vector_float_set(vector_, offset + sampX, gsl_vector_float_get(singVec, sampX));

    } catch (jiterator_error& e) {

      for (unsigned sampX = 0; sampX < singleSize_; sampX++)
	gsl_vector_float_set(vector_, plen_ + sampX, gsl_vector_float_get(vector_, (plen_ - singleSize_) + sampX));
      framesPadded_++;

    } catch (j_error& e) {

      if (e.getCode() != JITERATOR) {
	cout << e.what() << endl;
	throw;
      }
      for (unsigned sampX = 0; sampX < singleSize_; sampX++)
	gsl_vector_float_set(vector_, plen_ + sampX, gsl_vector_float_get(vector_, (plen_ - singleSize_) + sampX));
      framesPadded_++;

    }
  } else if (framesPadded_ < delta_) { // pad with zeros
    for (unsigned sampX = 0; sampX < singleSize_; sampX++)
      gsl_vector_float_set(vector_, plen_ + sampX, gsl_vector_float_get(vector_, (plen_ - singleSize_) + sampX));
    framesPadded_++;
  } else { // end of utterance
    throw jiterator_error("end of samples (FilterFeature)!");
  }
}

const gsl_vector_float* AdjacentFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  buffer_next_frame_(frame_no_ + 1);
  increment_();

  return vector_;
}

void AdjacentFeature::reset()
{
  single_->reset();

  framesPadded_ = 0;
  gsl_vector_float_set_zero(vector_);

  VectorFloatFeatureStream::reset();
}


// ----- methods for class `LinearTransformFeature' -----
//
LinearTransformFeature::
LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz, const String& nm) :
  // VectorFloatFeatureStream((sz == 0) ? mat->size1 : sz, nm),
  VectorFloatFeatureStream(sz, nm),
  src_(src),
  trans_(gsl_matrix_float_calloc(size(), src_->size()))
{}

const gsl_vector_float* LinearTransformFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* logVec = src_->next(frame_no_ + 1);
  increment_();

  gsl_blas_sgemv(CblasNoTrans, 1.0, trans_, logVec, 0.0, vector_);

  return vector_;
}

gsl_matrix_float* LinearTransformFeature::matrix() const
{
#if 0
  cout << "Allocating Transformation Matrix (" << size() << " x " << src_->size() << ")" << endl;

  gsl_matrix_float* matrix = gsl_matrix_float_calloc(size(), src_->size());

  gsl_matrix_float_memcpy(matrix, trans_);

  return matrix;
#else
  return trans_;
#endif
}

void LinearTransformFeature::load(const String& fileName, bool old)
{
  gsl_matrix_float_load(trans_, fileName, old);
  trans_ = gsl_matrix_float_resize(trans_, size(), src_->size());
}

void LinearTransformFeature::identity()
{
  if (size() != src_->size())
    throw jdimension_error("Cannot set an (%d x %d) matrix to identity.", size(), src_->size());

  gsl_matrix_float_set_zero(trans_);
  for (unsigned i = 0; i < size(); i++)
    gsl_matrix_float_set(trans_, i, i, 1.0);
}


// ----- methods for class `StorageFeature' -----
//
StorageFeature::StorageFeature(const VectorFloatFeatureStreamPtr& src,
			       const String& nm) :
  VectorFloatFeatureStream(src->size(), nm), src_(src), frames_(MaxFrames)
{
  for (int i = 0; i < MaxFrames; i++)
    frames_[i] = gsl_vector_float_calloc(size());
}

StorageFeature::~StorageFeature()
{
  for (int i = 0; i < MaxFrames; i++)
    gsl_vector_float_free(frames_[i]);
}

const int StorageFeature::MaxFrames = 100000;
const gsl_vector_float* StorageFeature::next(int frame_no)
{
  if (frame_no >= 0 && frame_no <= frame_no_) return frames_[frame_no];

  if (frame_no >= MaxFrames)
    throw jdimension_error("Frame %d is greater than maximum number %d.",
			   frame_no, MaxFrames);

  const gsl_vector_float* singVec = src_->next(frame_no_ + 1);
  increment_();

  gsl_vector_float_memcpy(frames_[frame_no_], singVec);

  return frames_[frame_no_];
}

void StorageFeature::write(const String& fileName, bool plainText) const
{
  if (frame_no_ <= 0)
    throw jio_error("Frame count must be > 0.\n", frame_no_);

  FILE* fp = btk_fopen(fileName, "w");
  if (plainText) {
    int sz = frames_[0]->size;
    fprintf(fp, "%d %d\n", frame_no_, sz);

    for (int i = 0; i <= frame_no_; i++) {
      for (int j = 0; j < sz; j++) {
	fprintf(fp, "%g", gsl_vector_float_get(frames_[i], j));
	if (j < sz - 1)
	  fprintf(fp, " ");
      }
      fprintf(fp, "\n");
    }

  } else {
    write_int(fp, frame_no_);
    write_int(fp, frames_[0]->size);

    for (int i = 0; i <= frame_no_; i++)
      gsl_vector_float_fwrite(fp, frames_[i]);
  }
  btk_fclose(fileName, fp);
}

void StorageFeature::read(const String& fileName)
{
  FILE* fp = btk_fopen(fileName, "r");
  frame_no_ = read_int(fp);
  int sz  = read_int(fp);

  if (sz != int(frames_[0]->size))
    throw jdimension_error("Feature dimensions (%d vs. %d) do not match.\n",
                           sz, frames_[0]->size);

  for (int i = 0; i < frame_no_; i++)
    gsl_vector_float_fread(fp, frames_[i]);
  btk_fclose(fileName, fp);
}

int StorageFeature::evaluate()
{
  reset();

  int frame_no = 0;
  try {
    while(true)
      next(frame_no++);
   } catch (jiterator_error& e) {
   } catch (j_error& e) {
     if (e.getCode() != JITERATOR) {
       throw;
     }
   }

  return frame_no_;
}

// ----- methods for class `StaticStorageFeature' -----
//
const int StaticStorageFeature::MaxFrames = 10000;

StaticStorageFeature::StaticStorageFeature(unsigned dim, const String& nm) :
  VectorFloatFeatureStream(dim, nm), frames_(MaxFrames)
{
  for (int i = 0; i < MaxFrames; i++)
    frames_[i] = gsl_vector_float_alloc(size());
}


StaticStorageFeature::~StaticStorageFeature()
{
  for (int i = 0; i < MaxFrames; i++)
    gsl_vector_float_free(frames_[i]);
}


const gsl_vector_float* StaticStorageFeature::next(int frame_no)
{
  if (frame_no >= 0 && frame_no <= frame_no_)
    return frames_[frame_no];

  increment_();

  if (frame_no_ >= framesN_) /* if (frame_no >= framesN_) */
    throw jiterator_error("end of samples!");

  return frames_[frame_no_];
}


void StaticStorageFeature::read(const String& fileName)
{
  FILE* fp = btk_fopen(fileName, "r");
  framesN_ = read_int(fp);
  int sz  = read_int(fp);

  if (sz != int(frames_[0]->size))
    throw jdimension_error("Feature dimensions (%d vs. %d) do not match.\n",
         sz, frames_[0]->size);

  for (int i = 0; i < framesN_; i++)
    gsl_vector_float_fread(fp, frames_[i]);
  btk_fclose(fileName, fp);

  //printf("StaticStorageFeature: read %i features\n", framesN_);
}


int StaticStorageFeature::evaluate()
{
  return framesN_;
  //return frame_no_;
}


// ----- methods for class `CircularStorageFeature' -----
//
CircularStorageFeature::CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN,
					       const String& nm) :
  VectorFloatFeatureStream(src->size(), nm), src_(src), framesN_(framesN), frames_(framesN_), pointerX_(framesN_-1)
{
  for (unsigned i = 0; i < framesN_; i++) {
    frames_[i] = gsl_vector_float_calloc(size());
    gsl_vector_float_set_zero(frames_[i]);
  }
}

CircularStorageFeature::~CircularStorageFeature()
{
  for (int i = 0; i < framesN_; i++)
    gsl_vector_float_free(frames_[i]);
}

unsigned CircularStorageFeature::get_index_(int frame_no) const
{
  int diff = frame_no_ - frame_no;
  if (diff >= framesN_)
    throw jconsistency_error("Difference (%d) is not in range [%d, %d)\n", frame_no, 0, framesN_);

  return (pointerX_ + framesN_ - diff) % framesN_;
}

const gsl_vector_float* CircularStorageFeature::next(int frame_no)
{
  if (frame_no >= 0 && frame_no <= frame_no_) return frames_[get_index_(frame_no)];

  if (frame_no >= 0 && frame_no != frame_no_ + 1)
    throw jconsistency_error("Requested frame %d\n", frame_no);

  const gsl_vector_float* block = src_->next(frame_no_ + 1);
  increment_();
  pointerX_ = (pointerX_ + 1) % framesN_;
  gsl_vector_float_memcpy(frames_[pointerX_], block);

  return frames_[pointerX_];
}

void CircularStorageFeature::reset()
{
  pointerX_ = framesN_ - 1;  src_->reset();  VectorFloatFeatureStream::reset();
}


// ----- methods for class `FilterFeature' -----
//
FilterFeature::
FilterFeature(const VectorFloatFeatureStreamPtr& src, gsl_vector* coeffA,
	      const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    src_(src),
    lenA_(coeffA->size),
    coeffA_(gsl_vector_calloc(lenA_)),
    offset_(int((lenA_ - 1) / 2)),
    buffer_(size(), lenA_),
    framesPadded_(0)
{
  if (lenA_ % 2 != 1)
    throw jdimension_error("Length of filter (%d) is not odd.", lenA_);

  gsl_vector_memcpy(coeffA_, coeffA);
}

FilterFeature::~FilterFeature()
{
  gsl_vector_free(coeffA_);
}

void FilterFeature::buffer_next_frame_(int frame_no)
{
  if (frame_no == 0) {				// initialize the buffer

    for (int i = 0; i < offset_; i++) {
      const gsl_vector_float* srcVec = src_->next(i);
      buffer_.nextSample(srcVec);
    }

  }

  if (framesPadded_ == 0) {			// normal processing

    try {
      const gsl_vector_float* srcVec = src_->next(frame_no + offset_);
      buffer_.nextSample(srcVec);
    } catch (jiterator_error& e) {
      buffer_.nextSample();
      framesPadded_++;
    } catch (j_error& e) {
      if (e.getCode() != JITERATOR) {
	cout << e.what() << endl;
	throw;
      }
      buffer_.nextSample();
      framesPadded_++;
    }


  } else if (framesPadded_ < offset_) {		// pad with zeros

    buffer_.nextSample();
    framesPadded_++;

  } else {					// end of utterance

    throw jiterator_error("end of samples (FilterFeature)!");

  }
}

const gsl_vector_float* FilterFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  buffer_next_frame_(frame_no_ + 1);
  increment_();

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {
    double sum = 0.0;

    for (int i = -offset_; i <= offset_; i++) {
      sum += gsl_vector_get(coeffA_, i + offset_) * buffer_.sample(-i, coeffX);
    }

    gsl_vector_float_set(vector_, coeffX, sum);
  }

  return vector_;
}

void FilterFeature::reset()
{
  src_->reset();

  framesPadded_ = 0;
  buffer_.zero();

  VectorFloatFeatureStream::reset();
}


// ----- methods for class `MergeFeature' -----
//
MergeFeature::
MergeFeature(VectorFloatFeatureStreamPtr& stat, VectorFloatFeatureStreamPtr& delta,
	     VectorFloatFeatureStreamPtr& deltaDelta, const String& nm)
  : VectorFloatFeatureStream(stat->size() + delta->size() + deltaDelta->size(), nm)
{
  flist_.push_back(stat);  flist_.push_back(delta);  flist_.push_back(deltaDelta);
}

const gsl_vector_float* MergeFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  unsigned dimX = 0;
  for (FeatureListIterator_ itr = flist_.begin(); itr != flist_.end(); itr++) {
    const gsl_vector_float* singVec = (*itr)->next(frame_no_ + 1);
    for (unsigned i = 0; i < singVec->size; i++)
      gsl_vector_float_set(vector_, dimX++, gsl_vector_float_get(singVec, i));
  }
  increment_();

  return vector_;
}

void MergeFeature::reset()
{
  for (FeatureListIterator_ itr = flist_.begin(); itr != flist_.end(); itr++)
    (*itr)->reset();

  VectorFloatFeatureStream::reset();
}

// ----- methods for class `MultiModalFeature' -----
 //
MultiModalFeature::MultiModalFeature( unsigned nModality, unsigned totalVecSize, const String& nm )
  : VectorFloatFeatureStream( totalVecSize, nm ),
    nModality_(nModality),curr_vecsize_(0)
{
  samplePeriods_ = new unsigned[nModality];
  minSamplePeriod_ = -1;
}

MultiModalFeature::~MultiModalFeature()
{
  delete [] samplePeriods_;
}

const gsl_vector_float* MultiModalFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;
  
  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);
  
  unsigned modalX = 0;
  unsigned dimX = 0;
  unsigned currFrameX = frame_no_ + 1;
  for (FeatureListIterator_ itr = flist_.begin(); itr != flist_.end(); itr++,modalX++) {
    if( ( currFrameX % (samplePeriods_[modalX]/minSamplePeriod_) ) == 0 || currFrameX==0 ){// update the feature vector
      const gsl_vector_float* singVec = (*itr)->next(currFrameX);
      for (unsigned i = 0; i < (*itr)->size(); i++)
	gsl_vector_float_set(vector_, dimX++, gsl_vector_float_get(singVec, i));
    }
    else{
      dimX += (*itr)->size();
    }
  }
  increment_();

  return vector_;
}

void MultiModalFeature::reset()
{
  for (FeatureListIterator_ itr = flist_.begin(); itr != flist_.end(); itr++)
    (*itr)->reset();

  VectorFloatFeatureStream::reset();
}

void MultiModalFeature::addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec )
{
  if( flist_.size() == nModality_ ){
    throw jdimension_error("The number of the modal features exceeds %d\n",nModality_);
  }
  flist_.push_back(feature);
  samplePeriods_[flist_.size()-1] = samplePeriodinNanoSec;
  if( samplePeriodinNanoSec < minSamplePeriod_ ){
    minSamplePeriod_ = samplePeriodinNanoSec;
  }
  curr_vecsize_ += feature->size();
  if( size() < curr_vecsize_ ){
    throw jdimension_error("The total vector size exceeds %d\n",size());
  }
}

void writeGSLMatrix(const String& fileName, const gsl_matrix* mat)
{
  FILE* fp = btk_fopen(fileName, "w");

  fprintf(fp, "%lu ", mat->size1);
  fprintf(fp, "%lu ", mat->size2);

  int ret = gsl_matrix_fwrite (fp, mat);
  if (ret != 0)
    throw jio_error("Could not write data to file %s.\n", fileName.c_str());
  btk_fclose(fileName, fp);
}

#ifdef JACK
// ----- methods for class `Jack' -----
//
Jack::Jack(const String& nm)
{
  can_capture = false;
  can_process = false;
  if ((client = jack_client_new (nm.c_str())) == 0)
    throw jio_error("Jack server not running?");
  jack_set_process_callback (client, _process_callback, this);
  jack_on_shutdown (client, _shutdown_callback, this);
  if (jack_activate (client)) {
    throw jio_error("cannot activate client");
  }
}

Jack::~Jack()
{
  jack_client_close(client);
  for (unsigned i = 0; i < channel.size(); i++) {
    jack_ringbuffer_free(channel[i]->buffer);
    delete channel[i];
  }
}

jack_channel_t* Jack::addPort(unsigned buffersize, const String& connection, const String& nm)
{
  jack_channel_t* ch = new (jack_channel_t);
  ch->buffersize = buffersize;
  if ((ch->port = jack_port_register (client, nm.c_str(), JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0)) == 0) {
    delete ch;
    ch = NULL;
    throw jio_error ("cannot register input port \"%s\"!", nm.c_str());
  }
  if (jack_connect (client, connection.c_str(), jack_port_name (ch->port))) {
    delete ch;
    ch = NULL;
    throw jio_error ("cannot connect input port %s to %s\n", nm.c_str(), connection.c_str());
  } 
  
  if (ch) {
    ch->buffer = jack_ringbuffer_create (sizeof(jack_default_audio_sample_t) * buffersize);
    ch->can_process = true;
    channel.push_back(ch);
  }
  can_process = true;		/* process() can start, now */

  return ch;
}

void Jack::shutdown_callback(void)
{
  throw j_error("JACK shutdown");
}

int Jack::process_callback(jack_nframes_t nframes)
{
	size_t bytes;
	jack_default_audio_sample_t *in;

	/* Do nothing until we're ready to begin. */
	if ((!can_process) || (!can_capture))
		return 0;

	for (unsigned i = 0; i < channel.size(); i++)
	  if (channel[i]->can_process) {
	    in = (jack_default_audio_sample_t *) jack_port_get_buffer (channel[i]->port, nframes);
	    bytes = jack_ringbuffer_write (channel[i]->buffer, (char *) in, sizeof(jack_default_audio_sample_t) * nframes);
	  }

	return 0;
}


// ----- methods for class `JackFeature' -----
//
JackFeature::JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
			 const String& connection, const String& nm) :
  VectorFloatFeatureStream(blockLen, nm), jack_(jack)

{
  channel = jack->addPort(buffersize, connection, nm);
}

const gsl_vector_float* JackFeature::next(int frame_no)
{
  unsigned i = 0;
  jack_default_audio_sample_t frame = 0;
  unsigned s = sizeof(jack_default_audio_sample_t)*size();

  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  while ((jack_ringbuffer_read_space(channel->buffer) < s) && (i < 10000)) { i++; usleep(0); }
  if (i >= 10000)
    throw jio_error("safe-loop overrun!");

  for (unsigned i = 0; i < size(); i++) {
    jack_ringbuffer_read (channel->buffer, (char*) &frame, sizeof(frame));
    gsl_vector_float_set(vector_, i, frame);
  }

  increment_();
  return vector_;
}

#endif


// ----- methods for class `ZeroCrossingHammingFeature' -----
// ----- calculates the Zero Crossing Rate with a Hamming window as weighting function
//
ZeroCrossingRateHammingFeature::ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(1, nm), samp_(samp), windowLen_(samp->size()),
    window_(new double[windowLen_])
{
  double temp = 2. * M_PI / (double)(windowLen_ - 1);
  for ( unsigned i = 0 ; i < windowLen_; i++ )
    window_[i] = 0.54 - 0.46*cos(temp*i);
}

const gsl_vector_float* ZeroCrossingRateHammingFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* block = samp_->next(frame_no_ + 1);
  increment_();

  float sum = 0;
  for (unsigned i = 0; i < windowLen_ - 1; i++) {
    int s_n = gsl_vector_float_get(block, i + 1) >= 0 ? 1 : -1;
    int s = gsl_vector_float_get(block, i) >= 0 ? 1 : -1;
    sum += abs(s_n - s) / 2 * window_[i];
  }
  sum /= windowLen_;
  gsl_vector_float_set(vector_, 0, sum);

  return vector_;
}


// ----- methods for class `YINPitchFeature' -----
// ----- source code adapted from aubio library ----
// ----- according to de Cheveigne and Kawahara "YIN, a fundamental frequency estimator for speech and music"
//
YINPitchFeature::YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate, float threshold, const String& nm)
  : VectorFloatFeatureStream(1, nm), samp_(samp), sr_(samplerate), tr_(threshold) { }

float YINPitchFeature::_getPitch(const gsl_vector_float *input, gsl_vector_float *yin, float tol)
{ 
  unsigned int j,  tau = 0;
  float tmp = 0., tmp2 = 0.;

  gsl_vector_float_set(yin, 0, 1.);
  for (tau = 1; tau < yin->size; tau++) {
    gsl_vector_float_set(yin, tau, 0.);	
    for (j = 0; j < yin->size; j++) {
      tmp = gsl_vector_float_get(input, j) - gsl_vector_float_get(input, j + tau);			
      gsl_vector_float_set(yin, tau, gsl_vector_float_get(yin, tau) + tmp * tmp);			
    }
    tmp2 += gsl_vector_float_get(yin, tau);				
    gsl_vector_float_set(yin, tau, gsl_vector_float_get(yin, tau) * tau/tmp2);

    if((gsl_vector_float_get(yin, tau) < tol) && 
       (gsl_vector_float_get(yin, tau-1) < gsl_vector_float_get(yin,tau))) {
      return tau-1;
    }
  }
  return 0;
}

const gsl_vector_float* YINPitchFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* input = samp_->next(frame_no_ + 1);
  increment_();
  gsl_vector_float* yin = gsl_vector_float_calloc(input->size / 2);

  float pitch =  _getPitch(input, yin, tr_);

  if (pitch>0) {
    pitch = sr_/(pitch+0.);
  } else {
     pitch = 0.;
  }

  gsl_vector_float_set(vector_, 0, pitch);

  gsl_vector_float_free(yin);

  return vector_;
}


// ----- methods for class `SpikeFilter' -----
//
SpikeFilter::SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), src_(src),
    adcN_(src->size()), queueN_((tapN-1)>>1), queue_(new float[queueN_]), windowN_(tapN),  window_(new float[windowN_])
{
  if (tapN < 3)
    throw jdimension_error("tapN should be at least 3.");

  if (adcN_ < tapN)
    throw jdimension_error("Cannot filter with adcN = %d and tapN = %d.", adcN_, tapN);
}

SpikeFilter::~SpikeFilter()
{
  delete[] queue_;
  delete[] window_;
}

const gsl_vector_float* SpikeFilter::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* adc = src_->next(frame_no_ + 1);
  increment_();

  // fill queue with initial values
  for (unsigned queueX = 0; queueX < queueN_; queueX++)
    queue_[queueX] = gsl_vector_float_get(adc, queueX);
  unsigned queuePnt = 0;

  // move filter window over the waveform array
  for (unsigned adcX = queueN_; adcX < adcN_ - queueN_; adcX++) {

    // copy samples into filter window and sort them
    for (unsigned windowX = 0; windowX < windowN_; windowX++) {
      window_[windowX] = gsl_vector_float_get(adc, adcX + windowX - queueN_);
      int i = windowX;
      int j = windowX-1;
      while ((j >= 0) && (window_[j] > window_[i])) {
        float swappy = window_[i];
        window_[i]   = window_[j];
        window_[j]   = swappy;
        i = j--;
      }
    }

    // take oldest sample out of the queue
    gsl_vector_float_set(vector_, adcX - queueN_, queue_[queuePnt]);

    // pick median and copy it into the queue
    queue_[queuePnt++] = window_[queueN_];
    queuePnt %= queueN_;
  }

  return vector_;
}


// ----- methods for class `SpikeFilter2' -----
//
SpikeFilter2::SpikeFilter2(VectorFloatFeatureStreamPtr& src, 
			   unsigned width, float maxslope, float startslope, float thresh, float alpha, unsigned verbose, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), src_(src),
    adcN_(src->size()), width_(width), maxslope_(maxslope), startslope_(startslope), thresh_(thresh), alpha_(alpha), beta_(1.0 - alpha), verbose_(verbose) { }

void SpikeFilter2::reset()
{
  src_->reset();  VectorFloatFeatureStream::reset();  meanslope_ = startslope_;  count_ = 0;
}

const gsl_vector_float* SpikeFilter2::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  const gsl_vector_float* adc = src_->next(frame_no_ + 1);
  increment_();

  for (unsigned i = 0; i < adcN_; i++)
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(adc, i));

  unsigned adcP      = 0;
  unsigned adcQ      = 1;
  while (adcQ < adcN_) {

    // check for very high slopes in the signal
    float slope = gsl_vector_float_get(vector_, adcQ) - gsl_vector_float_get(vector_, adcP);
    int signB, signE, spikeE;
    if (slope < 0.0) {
      slope *= -1;
      signB = -1;
    }
    else signB = 1;
    adcP = adcQ++;
    float max  = thresh_ * meanslope_;

    // check for spike
    if (slope > max && slope > maxslope_) {
      float oslope = slope;

      // determine width of actual spike
      unsigned spikeB = adcP-1;
      unsigned spikeN = 0;
      while ((adcQ < adcN_) && (spikeN < width_)) {
        slope = gsl_vector_float_get(vector_, adcQ) - gsl_vector_float_get(vector_, adcP);
        if (slope < 0) {
          slope = -1*slope;
          signE = -1;
        }
        else signE = 1;
        adcP = adcQ++;
        spikeN++;
        if (signB != signE && slope > max && slope > maxslope_) break;
      }
      spikeE = adcP;

      // filter out spike by linear interpolation
      for (int spikeX = spikeB+1; spikeX < spikeE; spikeX++) {
        float lambda = ((float) (spikeX - spikeB)) / (spikeE - spikeB);
        gsl_vector_float_set(vector_, spikeX, (1.0 - lambda) *  gsl_vector_float_get(vector_, spikeB) + lambda * gsl_vector_float_get(vector_, spikeE));
      }
      count_++;
      if (verbose_ > 1) printf("spike %d at %d..%d, slope = %f, max = %f\n",
                               count_, spikeB+1, spikeE-1, oslope, max);

    }
    else {
      meanslope_ = beta_ * meanslope_ + alpha_ * slope;
    }
  }
  if (verbose_ > 0 && count_ > 0) printf("%d spikes removed\n", count_);

  return vector_;
}


namespace sndfile {
// ----- methods for class `SoundFile' -----
//
SoundFile::SoundFile(const String& fn,
		     int mode,
		     int format,
		     int samplerate,
		     int channels,
		     bool normalize)
{
   memset(&sfinfo_, 0, sizeof(sfinfo_));
  sfinfo_.channels   = channels;
  sfinfo_.samplerate = samplerate;
  sfinfo_.format     = format;
  sndfile_ = sf_open(fn.c_str(), mode, &sfinfo_);
  cout << "Reading sound file " << fn.c_str() << endl;
  if (sndfile_ == NULL)
    throw jio_error("Could not open file %s.", fn.c_str());
  if (sf_error(sndfile_)) {
    sf_close(sndfile_);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile_));
  }
#ifdef DEBUG
  cout << "channels: "   << sfinfo_.channels   << endl;
  cout << "frames: "     << sfinfo_.frames     << endl;
  cout << "samplerate: " << sfinfo_.samplerate << endl;
#endif
  if (normalize)
    sf_command(sndfile_, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  else
    sf_command(sndfile_, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
}
}

// ----- methods for class `DirectSampleFeature' -----
//
DirectSampleFeature::DirectSampleFeature(const SoundFilePtr &sndfile, unsigned blockLen,
					 unsigned start, unsigned end, const String& nm)
  : VectorFloatFeatureStream(blockLen*sndfile->channels(), nm), sndfile_(sndfile),
    blockLen_(blockLen), start_(start), end_(end), cur_(0)
{
  if (end_ == (unsigned)-1) end_ = sndfile_->frames();
  sndfile_->seek(start_, SEEK_SET);
}

const gsl_vector_float* DirectSampleFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();

  if (cur_ >= end_)
    throw jiterator_error("end of samples!");

  unsigned readN;
  if (cur_ + blockLen_ >= end_) {
    gsl_vector_float_set_zero(vector_);
    readN = sndfile_->readf(gsl_vector_float_ptr(vector_, 0), end_-cur_);
    if (readN != end_-cur_)
      throw jio_error("Problem while reading from file. (%d != %d)", readN, end_-cur_);
  } else {
    readN = sndfile_->readf(gsl_vector_float_ptr(vector_, 0), blockLen_);
    if (readN != blockLen_)
      throw jio_error("Problem while reading from file. (%d != %d)", readN, blockLen_);
  }
  cur_ += readN;
  if (readN == 0)
    throw jiterator_error("end of samples!");

  return vector_;
}

// ----- methods for class `DirectSampleOutputFeature' -----
//
DirectSampleOutputFeature::DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
                                                     const SoundFilePtr &sndfile,
                                                     const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), src_(src), sndfile_(sndfile)
{
  blockLen_ = size() / sndfile_->channels();
  if ((size() % sndfile_->channels()) != 0)
    throw jconsistency_error("Block length (%d) is not a multiple of the number of channels (%d)\n", size(), sndfile_->channels());
  sndfile_->seek(0, SEEK_SET);
}

const gsl_vector_float* DirectSampleOutputFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();

  gsl_vector_float_memcpy(vector_, src_->next(frame_no_));
  unsigned n = sndfile_->writef(gsl_vector_float_ptr(vector_, 0), blockLen_);
  if (n != blockLen_)
    throw jio_error("Problem while writing to file. (%d != %d)", n, blockLen_);

  return vector_;
}

const gsl_vector_float* ChannelExtractionFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();

  const gsl_vector_float* allChannels = src_->next(frame_no_);
  gsl_vector_float_set_zero(vector_);
  for (unsigned i=0; i<size(); i++) {
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(allChannels, i * chN_ + chX_));
  }
  return vector_;
}


// ----- Methods for class 'SignalInterferenceFeature' -----
//
SignalInterferenceFeature::
SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal,
                          VectorFloatFeatureStreamPtr& interference,
                          double dBInterference, unsigned blockLen, const String& nm):
  VectorFloatFeatureStream(blockLen, nm),
  signal_(signal), interference_(interference), level_(pow(10.0, dBInterference / 20.0)) { }

const gsl_vector_float* SignalInterferenceFeature::next(int frame_no) {

  if (frame_no == frame_no_) return vector_;

  gsl_vector_float_memcpy(vector_, interference_->next(frame_no));

  gsl_vector_float_scale(vector_, level_);
  gsl_vector_float_add(vector_, signal_->next(frame_no));

  increment_();
  return vector_;
}


// ----- Methods for class 'AmplificationFeature' -----
//
const gsl_vector_float* AmplificationFeature::next(int frame_no)
{
  if (frame_no == frame_no_) return vector_;

  if (frame_no >= 0 && frame_no - 1 != frame_no_)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frame_no - 1, frame_no_);

  increment_();

  const gsl_vector_float* vector = src_->next(frame_no_);
  for (unsigned i=0; i<size(); i++) {
    gsl_vector_float_set(vector_, i, gsl_vector_float_get(vector, i) * amplify_);
  }
  return vector_;
}

 // ----- definition for class `WriteSoundFile' -----
//

WriteSoundFile::WriteSoundFile(const String& fn, int sampleRate, int nChan, int format)
{
  sfinfo_.samplerate = sampleRate;
  sfinfo_.channels = nChan;
  sfinfo_.format = format;
  sfinfo_.frames = 0;
  sfinfo_.sections = 0;
  sfinfo_.seekable = 0;

  sndfile_ = sndfile::sf_open(fn.c_str(), sndfile::SFM_WRITE, &sfinfo_);
  if (!sndfile_)
    throw jio_error("Error opening file %s.", fn.c_str());
}

WriteSoundFile::~WriteSoundFile()
{
  sf_close(sndfile_);
}

int WriteSoundFile::write( gsl_vector *vector )
{
  using namespace sndfile;
  int ret;

  if( sfinfo_.format & SF_FORMAT_FLOAT )
    ret = writeFloat( vector );
  else if(sfinfo_.format & SF_FORMAT_PCM_32 )
    ret = writeInt( vector );
  else
    ret = writeShort( vector );

  return ret;
}

int WriteSoundFile::writeInt( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  sfinfo_.channels * vector->size;
  int *buf = new int[frames];

  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_int( sndfile_, buf, frames);

  delete [] buf;
  return ret;
}

int WriteSoundFile::writeFloat( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  sfinfo_.channels * vector->size;
  float *buf = new float[frames];

  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_float( sndfile_, buf, frames);

  delete [] buf;
  return ret;
}

int WriteSoundFile::writeShort( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  sfinfo_.channels * vector->size;
  short *buf = new short[frames];

  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_short( sndfile_, buf, frames);

  delete [] buf;
  return ret;
}

