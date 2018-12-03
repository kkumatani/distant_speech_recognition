/**
 * @file videofeature.h
 * @brief Audio-visual speech recognition front end.
 * @author Munir Georges, John McDonough, Friedrich Faubel
 */
#include "btk.h"

#ifdef AVFORMAT

#ifdef OPENCV

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sort_vector.h>

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif

#undef HAVE_CONFIG_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define HAVE_CONFIG_H

#include "stream/stream.h"
#include "common/mlist.h"

#ifndef VIDEOFEATURE_H
#define VIDEOFEATURE_H


// ----- definition for class VideoFeature -----
//
class VideoFeature : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  VideoFeature(int mode, unsigned width, unsigned height, const String& nm = "VideoFeature");

  ~VideoFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void read(const String& fileName, int from, int _to);

  int width() const { return _width; }

  int height() const { return _height; }

  int frameNumber() const { return _frameNumber;}
  int Frames();

 private:
  CvCapture*					_capture;
  IplImage*					_frame;
  IplImage*					_gray;
  IplImage*					_hsv;

  unsigned					_width;
  unsigned					_height;

  gsl_vector_float*				_R;
  gsl_vector_float*				_G;
  gsl_vector_float*				_B;

  String					_filename;
  unsigned					_from;
  unsigned					_to;
  int						_mode;

  bool						_endOfSamples;
  string					_fileName;
  int						_frameNumber;
};

typedef Inherit<VideoFeature, VectorFloatFeatureStreamPtr> VideoFeaturePtr;


// ----- definition for class 'ImageROI' -----
//
class ImageROI : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageROI(const VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, unsigned x=0, unsigned y=0, unsigned w=2, unsigned h=3, const String& nm = "ImageROI");
  
  ~ImageROI();
  void setROI(int x,int y);

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
  unsigned width() const { return _width; }

  unsigned height() const { return _height; }

  unsigned x_pos() const { return _x_pos; }

  unsigned y_pos() const { return _y_pos; }

  unsigned w() const { return _w; }

  unsigned h() const { return _h; }

 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  const unsigned				_width;
  const unsigned				_height;
  gsl_vector_float*				_srcVec;
  const unsigned				_w;
  const unsigned				_h;
  unsigned					_x_pos;
  unsigned					_y_pos;
};

typedef Inherit<ImageROI, VectorFloatFeatureStreamPtr> ImageROIPtr;


// ----- definition for class ImageSmooth ----- 
//
class ImageSmooth : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageSmooth(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int smoothtype, int param1, int param2, const String& nm = "ImageSmooth");
  
  ~ImageSmooth();
  
  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  unsigned					_width;
  unsigned					_height;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  IplImage*					_imgf;
  int						_smoothtype;
  int						_param1;
  int						_param2;
};

typedef Inherit<ImageSmooth, VectorFloatFeatureStreamPtr> ImageSmoothPtr;


// ----- definition for class ImageMorphology ----- 
//
class ImageMorphology : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageMorphology(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int type, int param, const String& nm = "ImageMorphology");
  
  ~ImageMorphology();
  
  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  unsigned					_width;
  unsigned					_height;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  IplImage*					_imgf;
  int						_type;
  int						_param;
};

typedef Inherit<ImageMorphology, VectorFloatFeatureStreamPtr> ImageMorphologyPtr;


// ----- definition for class ImageMorphologyEx ----- 
//
class ImageMorphologyEx : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageMorphologyEx(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int type, int param, const String& nm = "ImageMorphologyEx");
  
  ~ImageMorphologyEx();
  
  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  unsigned					_width;
  unsigned					_height;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  IplImage*					_imgf;
  IplImage*					_temp;
  int						_type;
  int						_param;
};

typedef Inherit<ImageMorphologyEx, VectorFloatFeatureStreamPtr> ImageMorphologyExPtr;


// ----- definition for class Canny ----- 
//
class Canny : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  Canny(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int param0, int param1, int param2, const String& nm = "Canny");

  ~Canny();

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  unsigned					_width;
  unsigned					_height;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  IplImage*					_imgf;
  int						_param0;
  int						_param1;
  int						_param2;
};

typedef Inherit<Canny, VectorFloatFeatureStreamPtr> CannyPtr;


// ----- definition for class ImageThreshold ----- 
//
class ImageThreshold : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageThreshold(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, float param0, int param1, int param2, const String& nm = "Canny");

  ~ImageThreshold();

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_endOfSamples;
  unsigned					_width;
  unsigned					_height;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  IplImage*					_imgf;
  int						_param0;
  int						_param1;
  int						_param2;
};

typedef Inherit<ImageThreshold, VectorFloatFeatureStreamPtr> ImageThresholdPtr;


/*Do not use this funktion within python - there is a problem with the window-handle or threading....*/
// ----- definition for class imageshow ----- 
//
class ImageShow : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageShow(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, const String& nm = "ImageShow");

  ~ImageShow();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			_src;
  unsigned					_width;
  unsigned					_height;
  IplImage*					_img;
  gsl_vector_float*				_srcVec;
  bool						_endOfSamples;
};

typedef Inherit<ImageShow, VectorFloatFeatureStreamPtr> ImageShowPtr;


// ----- definition for class 'SaveImage' ----- 
//
class SaveImage {
 public:
  // sz := number of elements in _vector
  SaveImage(unsigned width, unsigned height);

  ~SaveImage();

  void save(const gsl_vector_float* V, const String& filename);
  void savedouble(const gsl_vector* V, const String& filename);

 private:  
  unsigned					_width;
  unsigned					_height;
  IplImage*					_img;
  const String					_filename;
};

typedef refcount_ptr<SaveImage> SaveImagePtr;


// ----- definition for class 'ImageDetection' ----- 
//
class ImageDetection : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageDetection(VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, unsigned w, unsigned h,
		 const String& filename, double scale_factor, int min_neighbors, int flags, int min_sizeX, int min_sizeY,
		 const String& nm = "ImageDetection");
  
  ~ImageDetection();

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  double _linearnext(gsl_vector * v);

  VectorFloatFeatureStreamPtr			_src;
  const unsigned				_width;
  const unsigned				_height;
  unsigned					_w;
  unsigned					_h;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;

  const String					_filename;
  CvHaarClassifierCascade*			_pCascade;
  CvMemStorage*					_pStorage;
  CvSeq*					_pRectSeq;
  double					_scale_factor;
  int						_min_neighbors;
  int						_flags;
  CvSize					_min_size;

  int						_x_pos;
  int						_y_pos;

  const int					_his;
  gsl_vector*					_x_his;
  gsl_vector*					_y_his;
  int						_tmp_x;
  int						_tmp_y;
  bool						_endOfSamples;
};

typedef Inherit<ImageDetection, VectorFloatFeatureStreamPtr> ImageDetectionPtr;


// ----- definition for class FaceDetection ----- 
//
class FaceDetection : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  FaceDetection(VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, int region,
		const String& filename_eye, double scale_factor_eye, int min_neighbors_eye, int flags_eye, int min_sizeX_eye, int min_sizeY_eye,
		const String& filename_nose, double scale_factor_nose, int min_neighbors_nose, int flags_nose, int min_sizeX_nose, int min_sizeY_nose,
		const String& filename_mouth, double scale_factor_mouth, int min_neighbors_mouth, int flags_mouth, int min_sizeX_mouth, int min_sizeY_mouth,
		const String& nm = "FaceDetection");

  ~FaceDetection();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  bool						_endOfSamples;
  VectorFloatFeatureStreamPtr			_src;
  gsl_vector_float*				_srcVec;
  unsigned					_width;
  unsigned					_height;
};

typedef Inherit<FaceDetection, VectorFloatFeatureStreamPtr> FaceDetectionPtr;


// ----- definition for class ImageCentering ----- 
//
class ImageCentering : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  ImageCentering(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height,	const String& nm = "ImageCentering");
  
  ~ImageCentering();
  
  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  bool                                          _endOfSamples;
  VectorFloatFeatureStreamPtr	                _src;
  gsl_vector_float*                             _srcVec;
  IplImage*					_img;
  IplImage*					_gmi;
  IplImage*					_out;
  unsigned                                      _width;
  unsigned                                      _height;
};

typedef Inherit<ImageCentering, VectorFloatFeatureStreamPtr> ImageCenteringPtr;

// ----- definition for class LinearInterpolation ----- 
//
class LinearInterpolation : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  LinearInterpolation(const VectorFloatFeatureStreamPtr& src, double fps_src, double fps_dest,
		      const String& nm = "LinearInterpolation");
  ~LinearInterpolation();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			_src;
  const double					_DeltaTs;
  const double					_DeltaTd;

  int						_sourceFrameX;
  gsl_vector_float*				_x_n;
  bool						_endOfSamples;
};

typedef Inherit<LinearInterpolation, VectorFloatFeatureStreamPtr> LinearInterpolationPtr;


// ----- definition for class OpticalFlowFeature ----- 
//
class OpticalFlowFeature : public VectorFloatFeatureStream {
 public:
  OpticalFlowFeature(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, const char* filename, const String& nm = "OpticalFlowFeature");
  
  ~OpticalFlowFeature();

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  float StringToFloat(const std::string& str_input);
  int StringToInt(const std::string& str_input);

  void disalloc_matrix(float **matrix, long nx, long ny);
  void alloc_matrix(float ***matrix, long nx, long ny);
  float ** _gray;
  float ** _ux;
  float ** _uy;
  typedef float fptype;
  typedef int itype;
  fptype **f1_c1;         /* in     : 1st image, channel 1*/
  fptype **f2_c1;         /* in     : 2nd image, channel 1*/
  fptype **f1_c2;         /* in     : 1st image, channel 2*/
  fptype **f2_c2;         /* in     : 2nd image, channel 2*/
  fptype **f1_c3;         /* in     : 1st image, channel 3*/
  fptype **f2_c3;         /* in     : 2nd image, channel 3*/
  fptype **u;             /* in+out : u component of flow field */
  fptype **v;             /* in+out : v component of flow field */
  itype  _nx;             /* in     : size in x-direction on current grid */
  itype  _ny;             /* in     : size in y-direction on current grid */
  itype  _bx;             /* in     : boundary size in x-direction */
  itype  _by;             /* in     : boundary size in y-direction */
  fptype hx;              /* in     : grid size in x-dir. on current grid */
  fptype hy;              /* in     : grid size in y-dir. on current grid */
  itype  m_type_d;        /* in     : type of data term */
  itype  m_type_s;        /* in     : type of smoothness term */
  fptype m_gamma_ofc1;    /* in     : weight of grey value constancy */
  fptype m_gamma_ofc2;    /* in     : weight of gradient constancy */
  fptype m_gamma_ofc3;    /* in     : weight of Hessian constancy */
  fptype m_gamma_gradnorm;/* in     : weight of gradient norm constancy */
  fptype m_gamma_laplace; /* in     : weight of Laplacian constancy */
  fptype m_gamma_hessdet; /* in     : weight of Hessian det. constancy */
  itype  m_function_d;    /* in     : type of robust function in data term */
  itype  m_function_s;    /* in     : type of robust function in smoothness term */
  fptype m_epsilon_d;     /* in     : parameter data term */
  fptype m_epsilon_s;     /* in     : parameter smoothness term */
  fptype m_power_d;       /* in     : exponent data term */
  fptype m_power_s;       /* in     : exponent smoothness term */
  fptype m_rhox;          /* in     : integration scale in x-direction*/
  fptype m_rhoy;          /* in     : integration scale in y-direction*/
  fptype m_rhox_F;        /* in     : FLOW integration scale in x-direction*/
  fptype m_rhoy_F;        /* in     : FLOW integration scale in y-direction*/
  fptype m_sigma_F;       /* in     : FLOW presmoothing scale */
  fptype m_alpha;         /* in     : smoothness weight */
  itype  m_color;         /* in     : color flag */
  itype  n_solver;        /* in     : general solver */
  itype  n_max_rec_depth; /* in     : max rec depth */
  itype  n_mg_res_prob;   /* in     : resample problem */
  itype  n_theta;         /* in     : discretisation parameter */
  itype  n_mg_cycles;     /* in     : number of cycles */
  itype  n_mg_solver;     /* in     : type of multigrid solver */
  itype  n_mg_pre_relax;  /* in     : number of pre relaxation steps */
  itype  n_mg_post_relax; /* in     : number of post relaxation steps */
  itype  n_mg_rec_calls;  /* in     : number of recursive calls */
  itype  n_mg_centered;   /* in     : intergrid transfer: 0-cell/1-vertex */
  itype  n_mg_mat_order_r;/* in     : matrix multigrid restriction order */
  itype  n_mg_cas_order_r;/* in     : cascadic multigrid restriction order */
  itype  n_mg_cas_order_p;/* in     : cascadic multigrid prolongation order */
  itype  n_mg_cor_order_r;/* in     : correcting multigrid restriction order */
  itype  n_mg_cor_order_p;/* in     : correcting multigrid prolongation order */
  itype  n_iter_out;      /* in     : outer iterations */
  itype  n_iter_in;       /* in     : inner iterations */
  fptype n_omega;         /* in     : overrelaxation parameter */
  fptype n_tau;           /* in     : time step size */
  fptype n_warp_eta;      /* in     : reduction factor */
  itype  n_warp_steps;    /* in     : warping steps per level */
  itype  n_warp_max_rec_depth; /*in : max warp rect depth */
  itype  n_warp_scaling;  /* in     : flag for scaling  */
  itype  n_warp_rem;      /* in     : flag for removing out of bounds estimates */
  itype  n_warp_centered; /* in     : warping intergrid transfer: 0-cell/1-vertex */
  itype  n_warp_order_r;  /* in     : correcting warping restriction order */
  itype  n_warp_order_p;  /* in     : correcting warping prolongation order */
  fptype n_tensor_eps;    /* in     : motion tensor normalisation factor */
  fptype **utruth;        /* in     : x-component of numerical ground truth */
  fptype **vtruth;        /* in     : y-component of numerical ground truth */
  fptype **uref;          /* in     : x-component of problem ground truth */
  fptype **vref;          /* in     : y-component of problem ground truth */
  fptype **error_mag;     /* in     : error magnitude map */
  itype  i_rel_res_flag;  /* in     : flag for relative residual computation */
  fptype *i_rel_res;      /* out    : relative residual */
  itype  i_rel_err_flag;  /* in     : flag for relative error computation */
  fptype *i_rel_err;      /* out    : relative error */
  itype  i_time_flag;     /* in     : flag for runtime computation */
  itype  *i_time_sec;     /* out    : consumed seconds */
  itype  *i_time_usec;    /* out    : consumed microseconds */
  itype  info_flag;       /* in     : flag for information */
  itype  info_step;       /* in     : stepsize for information */
  itype  write_flag;      /* in     : flag for writing out */
  itype  write_step;      /* in     : stepsize for writing out */
  itype  frame_nr;        /* in     : current frame number */
 

  bool						_endOfSamples;
  VectorFloatFeatureStreamPtr			_src;
  gsl_vector_float*				_srcVec;
  unsigned					_width;
  unsigned					_height;
};

typedef Inherit<OpticalFlowFeature, VectorFloatFeatureStreamPtr> OpticalFlowFeaturePtr;

// ----- definition for class SnakeImage ----- 
//
class SnakeImage : public VectorFloatFeatureStream {
 public:
  SnakeImage(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, float alpha, float beta, float gamma, const String& nm = "SnakeImage");
  
  ~SnakeImage();

  gsl_vector_float* data();
  virtual const gsl_vector_float* next(int frameX = -5);
 
  virtual void reset();
 
 private:
  bool                          		_endOfSamples;
  VectorFloatFeatureStreamPtr			_src;
  gsl_vector_float*				_srcVec;
  IplImage*					_img;
  unsigned                      		_width;
  unsigned                      		_height;
  float                       		        _alpha;
  float                       		        _beta ;
  float                       		        _gamma;
  
};

typedef Inherit<SnakeImage, VectorFloatFeatureStreamPtr> SnakeImagePtr;


// ----- definition for class PCAFeature ----- 
//
class PCAFeature : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  PCAFeature(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, const char *filename, const char *filename_mean, int n, int k, const String& nm = "PCAFeature");

  ~PCAFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  void _floatToDouble(gsl_vector_float *src, gsl_vector * dest);
  void _doubleToFloat(gsl_vector *src, gsl_vector_float * dest);
  void _normalizeRows();

  VectorFloatFeatureStreamPtr			_src;

  const unsigned				_width;
  const unsigned				_height;
  const unsigned				_N;
  const unsigned				_M;

  gsl_matrix*					_evec;
  gsl_vector*					_mean;
  gsl_vector_float*				_srcVec;

  bool						_endOfSamples;
};

typedef Inherit<PCAFeature, VectorFloatFeatureStreamPtr> PCAFeaturePtr;


// ----- definition for class IPCAFeature ----- 
//
class IPCAFeature : public VectorFloatFeatureStream {
 public:
  // sz := number of elements in _vector
  IPCAFeature(const VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, const String& filename,  const String& filename_mean, int n, int k, const String& nm = "IPCAFeature");
  ~IPCAFeature();

  virtual const gsl_vector_float* next(int frameX = -5);
  const gsl_vector* get_mean() const { return _mean; }
  const gsl_matrix* get_vec(int i) const { return _evec; }

  virtual void reset();

 private:
  void _floatToDouble(gsl_vector_float* src, gsl_vector* dest);
  void _doubleToFloat(gsl_vector* src, gsl_vector_float* dest);
  void _normalizeRows();

  VectorFloatFeatureStreamPtr			_src;

  const unsigned				_width;
  const unsigned				_height;
  const unsigned				_N;
  const unsigned				_M;

  gsl_matrix*					_evec;
  gsl_vector*					_mean;
  gsl_vector_float*				_srcVec;

  bool						_endOfSamples;
};

typedef Inherit<IPCAFeature, VectorFloatFeatureStreamPtr> IPCAFeaturePtr;


// ----- definition for class PCAEstimator ----- 
//
class PCAEstimator{
 public:
  // sz := number of elements in _vector
  PCAEstimator(unsigned n,unsigned m);

  ~PCAEstimator();

  void clearaccu();
  void accumulate(gsl_vector_float *V);
  void estimate();
  void save(const String& filename);

  const gsl_vector* get_mean()   const { return _mean; }
  const gsl_matrix* get_vec()    const { return _evec; }
  const gsl_vector* get_eigval() const { return _eval; }

 private:
  void     _floatToDouble(gsl_vector_float* src, gsl_vector* dest);
  void     _doubleToFloat(gsl_vector* src, gsl_vector_float* dest);
  unsigned					_N;
  unsigned					_M;

  unsigned					_in_data;
  gsl_vector*					_sum;    //sum per row
  gsl_vector*					_mean;

  gsl_matrix*					_data;   // original data, one image in each row
  gsl_matrix*					_mdata;  // data without row mean
  gsl_matrix*					_mdatat; // data'

  gsl_matrix*					_cov;    // cov = 1/M * (mdata' * data)
  gsl_vector*					_eval;   //eigen values
  gsl_matrix*					_evec;   //eigen vector
};

typedef refcount_ptr<PCAEstimator> PCAEstimatorPtr;


// ----- definition for class PCAModEstimator ----- 
//
class PCAModEstimator {
 public:
  // sz := number of elements in _vector
  PCAModEstimator(unsigned n, unsigned m);

  ~PCAModEstimator();

  const gsl_vector* get_mean()    const { return _mean; } 
  const gsl_matrix* get_data()    const { return _data; }
  const gsl_matrix* get_mdata()   const { return _mdata; }
  const gsl_vector* get_eigval()  const { return _eval; }
  const gsl_matrix* get_eigface() const { return _eigface; }
  const gsl_matrix* get_evecc()   const { return _evecc; }
  const gsl_matrix* get_evec()    const { return _evec; }
  const gsl_matrix* get_cov()     const { return _cov; }

  void clearaccu();
  void accumulate(gsl_vector_float* V);
  void estimate();
  void save(const char* filename1, const char* filename2);

 private:
  void _floatToDouble(gsl_vector_float* src, gsl_vector* dest);
  void _doubleToFloat(gsl_vector* src, gsl_vector_float* dest);
  void _normalizeRows();
  void _normalizeColumn();

  const unsigned				_N;
  const unsigned				_M;

  unsigned					_indata;
  gsl_matrix*					_data;   // original data, one image in each row
  gsl_matrix*					_mdata;  // data without row mean
  gsl_vector*					_sum;    // sum per row
  gsl_vector*					_mean;   // sum per row
  gsl_matrix*					_mdatat; // data'

  gsl_matrix*					_cov;    // cov = 1/M * (mdata' * data)
  gsl_vector*					_eval;   // eigen values
  gsl_matrix*					_evec;   // eigen vector
  gsl_matrix*					_evecc;  // eigen vector
  gsl_matrix*					_evecct; // eigen vector
  gsl_matrix*					_eigface;// eigen vector
};

typedef refcount_ptr<PCAModEstimator> PCAModEstimatorPtr;

#endif

#endif

#endif
