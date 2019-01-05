/**
 * @file localization.h
 * @brief source localization
 * @author Ulrich Klee
 */

#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fit.h>

#include "common/jexception.h"

gsl_vector* getSrpPhat(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeX, gsl_vector *searchRangeY, gsl_matrix *arrgeom, int zPos);

void calcDelays(int x, int y, int z, const gsl_matrix* mpos, gsl_vector* delays);

void calcDelaysOfLinearMicrophoneArray(float azimuth, const gsl_matrix* mpos, gsl_vector* delays);

void calcDelaysOfCircularMicrophoneArray(float azimuth, float elevation, const gsl_matrix* mpos, gsl_vector* delays);

gsl_vector* getDelays(double delta_f, gsl_matrix_complex* mFramePerChannel, gsl_vector* searchRange);

double getAzimuth(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_matrix *arrgeom, gsl_vector *delays);

double getPlaneWaveSrp(double delta_f, gsl_matrix_complex *mFramePerChannel, gsl_vector *searchRangeY, gsl_matrix *arrgeom);

//void halfComplexPack(double* tgt, const gsl_vector_complex* src);

void getGCCRaw(const gsl_matrix_complex* spectralSample, double sampleRate, gsl_vector* gcc);

const gsl_vector* getGCC(gsl_matrix_complex *spectralSample, double sampleRate);
const gsl_vector* getWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCratio(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCdirect(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getWindowedGCCabs(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay);

const gsl_vector* getDynWindowedGCC(gsl_matrix_complex *spectralSample, double sampleRate, double minDelay, double maxDelay, double wMinDelay, double wMaxDelay, double threshold);

double getInterpolation(gsl_matrix *crossResult, int delayPos);

gsl_vector* get3DPosition(gsl_vector *yCoord, gsl_vector *azimuth1, gsl_vector *azimuth2, double xPos, double zPos);

gsl_vector* get3DPosition_T_shape(gsl_matrix *arrgeom1, int arrayNr1, gsl_matrix *arrgeom2, int arrayNr2,  gsl_matrix *arrgeom3, double azimuth1, double azimuth2, double azimuth3);

gsl_vector* getGCC_old(gsl_matrix_complex *spectralSample, double delta_f, gsl_vector *delays);

gsl_matrix* getLowerTriangMatrix(gsl_matrix* fullMatrix);

gsl_vector* getXi(gsl_matrix* D1_2);

class PhaseTransform {
public:
  PhaseTransform(unsigned sz);
  ~PhaseTransform();
private:
  gsl_vector_complex*					_crossSpectrum;
  double*						_crossCorrelation;
};

class NoisePowerSpectrum
{
 public:
  NoisePowerSpectrum(double alpha = 0.95);
  ~NoisePowerSpectrum() {
    if (powerSpectrum != NULL)
      gsl_vector_free(powerSpectrum);
  }
  void setAlpha(double alpha) {
    this->alpha = alpha;
    alpha1 = 1 - alpha;
  }
  double getAlpha() { return alpha; }
  void add(const gsl_vector_complex *noiseSpectrum, double timestamp);
  const gsl_vector *get() {
    return powerSpectrum;
  }
 private:
  gsl_vector *powerSpectrum;
  double alpha, alpha1;
  double timestamp;
};

class NoiseCrossSpectrum
{
 public:
  NoiseCrossSpectrum(double alpha = 0.95);
  ~NoiseCrossSpectrum() {
    if (crossSpectrum != NULL)
      gsl_vector_complex_free(crossSpectrum);
  }
  void setAlpha(double alpha) {
    this->alpha = alpha;
    alpha1 = 1 - alpha;
  }
  double getAlpha() { return alpha; }
  const gsl_vector_complex *get() {
    return crossSpectrum;
  }
  void add(const gsl_vector_complex *noiseSpectrum1, const gsl_vector_complex *noiseSpectrum2);
 private:
  gsl_vector_complex *crossSpectrum;
  double alpha, alpha1;
};

class GCC
{
 public:
  GCC(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true);
  ~GCC();
  void calculate(const gsl_vector_complex *spectralSample1, unsigned chan1, const gsl_vector_complex *spectralSample2, unsigned chan2, unsigned pair, double timestamp, bool sad = false, bool smooth = true);
  const gsl_vector* findMaximum(double minDelay = -HUGE_VAL, double maxDelay = HUGE_VAL);
  double getPeakDelay() { return delay; }
  double getPeakCorr() { return maxCorr; }
  double getRatio() { return ratio; }
  const gsl_vector* getNoisePowerSpectrum(unsigned chan) { return noisePowerSpectrum[chan].get(); }
  const gsl_vector_complex* getNoiseCrossSpectrum(unsigned pair) { return noiseCrossSpectrum[pair].get(); }
  const gsl_vector_complex* getCrossSpectrum() { return crossSpectrum; }
  const gsl_vector* getCrossCorrelation() { return crossCorrelation; }
  void setAlpha(double alpha) {
    for(unsigned i=0; i < pairs; i++)
      noiseCrossSpectrum[i].setAlpha(alpha);
    for(unsigned i=0; i < nChan; i++)
      noisePowerSpectrum[i].setAlpha(alpha);
  }
  double getAlpha() { return noiseCrossSpectrum[0].getAlpha(); }
 protected:
  virtual gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i) { throw jdimension_error("Not implemented!!!\n"); };

  //auxiliary variables  
  gsl_complex x1, x2, Gx1x2, Gs1s2, G;
  double W1, W2, X1, X12, X2, X22;

  double sampleRate;
  unsigned fftLen;
  unsigned fftLen2;
  unsigned len;
  unsigned nChan;
  unsigned pairs;
  unsigned delayPos;
  double q, q1, q2, beta, beta1;
  double ratio;
  double delay;
  double maxCorr;
  double maxCorr2;
  gsl_vector *crossCorrelation;
  gsl_matrix *interpolValues;
  gsl_vector* retValues;
  gsl_vector *powerSpectrum1, *powerSpectrum2;
  gsl_vector_complex *cSpectrum;
  gsl_vector_complex *crossSpectrum;
  NoisePowerSpectrum* noisePowerSpectrum;
  NoiseCrossSpectrum* noiseCrossSpectrum;
  bool noisereduction;
  bool sad;
  bool interpolate;
};

class GCCRaw : public GCC
{
 public:
  GCCRaw(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

class GCCGnnSub : public GCC
{
 public:
  GCCGnnSub(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

class GCCPhat : public GCC
{
 public:
  GCCPhat(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

class GCCGnnSubPhat : public GCC
{
 public:
  GCCGnnSubPhat(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

class GCCMLRRaw : public GCC
{
 public:
  GCCMLRRaw(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

class GCCMLRGnnSub : public GCC
{
 public:
  GCCMLRGnnSub(double sampleRate = 44100.0 , unsigned fftLen = 2048, unsigned nChan = 16, unsigned pairs = 6, double alpha = 0.95, double beta = 0.5, double q = 0.3, bool interpolate = true, bool noisereduction = true) : GCC(sampleRate, fftLen, nChan, pairs, alpha, beta, q, interpolate, noisereduction) {};
 protected:
  gsl_complex calcCrossSpectrumValue(gsl_complex x1, gsl_complex x2, const gsl_vector_complex* Gn1n2, const gsl_vector* N1, const gsl_vector* N2, unsigned i);
};

#endif

