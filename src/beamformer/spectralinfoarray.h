#ifndef SPECTRALINFOARRAY_H
#define SPECTRALINFOARRAY_H

// ----- definition for class `SnapShotArray' -----
// 
class SnapShotArray {
 public:
  SnapShotArray(unsigned fftLn, unsigned nChn);
  virtual ~SnapShotArray();

  const gsl_vector_complex* snapshot(unsigned fbinX) const {
    assert (fbinX < fftLen_);
    return snapshots_[fbinX];
  }

  void set_samples(const gsl_vector_complex* samp, unsigned chanX);
  void set_snapshots(const gsl_vector_complex* snapshots, unsigned fbinX);

  unsigned fftLen() const { return fftLen_; }
  unsigned nChan()  const { return nChan_;  }

  virtual void update();
  virtual void zero();

#ifdef ENABLE_LEGACY_BTK_API
  const gsl_vector_complex* getSnapShot(unsigned fbinX){ return snapshot(fbinX); }
  void newSample(const gsl_vector_complex* samp, unsigned chanX){ set_samples(samp, chanX); }
#endif

 protected:
  const unsigned	fftLen_;
  const unsigned	nChan_;

  mutable gsl_vector_complex**	samples_;
  mutable gsl_vector_complex**	snapshots_;
};

typedef refcount_ptr<SnapShotArray> 	SnapShotArrayPtr;


// ----- definition for class `SpectralMatrixArray' -----
// 
class SpectralMatrixArray : public SnapShotArray {
 public:
  SpectralMatrixArray(unsigned fftLn, unsigned nChn, float forgetFact = 0.95);
  virtual ~SpectralMatrixArray();

  gsl_matrix_complex* matrix_f(unsigned idx) const {
    assert (idx < fftLen_);
    return matrices_[idx];
  }

  virtual void update();
  virtual void zero();

#ifdef ENABLE_LEGACY_BTK_API
  gsl_matrix_complex* getSpecMatrix(unsigned idx){ return matrix_f(idx); }
#endif

 protected:
  const gsl_complex	mu_; // forgetting factor
  gsl_matrix_complex**	matrices_; // multi-channel spectrums [fftLen_][nChan_][nChan_]
};

typedef refcount_ptr<SpectralMatrixArray> 	SpectralMatrixArrayPtr;

// ----- definition for class `FBSpectralMatrixArray' -----
// 
class FBSpectralMatrixArray : public SpectralMatrixArray {
 public:
  FBSpectralMatrixArray(unsigned fftLn, unsigned nChn, float forgetFact = 0.95);
  virtual ~FBSpectralMatrixArray();

  virtual void update();
};

#endif
