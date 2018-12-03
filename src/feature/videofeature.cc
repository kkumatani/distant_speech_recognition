/*
 * @file videofeature.cc
 * @brief Audio-visual speech recognition front end.
 * @author Munir Georges, John McDonough, Friedrich Faubel
 */
#ifdef AVFORMAT

#ifdef OPENCV

#include "feature/videofeature.h"
#include "matrix/gslmatrix.h"
#include "common/mach_ind_io.h"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>


// ----- methods for class `VideoFeature' -----
//
VideoFeature::VideoFeature(int mode, unsigned width, unsigned height, const String& nm) :
  VectorFloatFeatureStream(mode*width*height, nm),
  _capture(NULL), _frame(NULL), _gray(NULL), _hsv(NULL),
  _width(width), _height(height),
  _R(gsl_vector_float_calloc(_height*_width)),
  _G(gsl_vector_float_calloc(_height*_width)),
  _B(gsl_vector_float_calloc(_height*_width)),
  _fileName(""), _from(0), _to(0), _mode(mode),
  _endOfSamples(false),
  _frameNumber(0) { }

//Destructor
VideoFeature::~VideoFeature()
{
  gsl_vector_float_free(_R);
  gsl_vector_float_free(_G);
  gsl_vector_float_free(_B);
  cvReleaseCapture(&_capture);
  cvReleaseImage(&_frame);
  cvReleaseImage(&_gray);
  cvReleaseImage(&_hsv);
}

const gsl_vector_float* VideoFeature::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);
  _frame = cvQueryFrame( _capture );
  if (_frame == NULL) {
    cout << "ERROR: _frame = NULL with frameX: " << frameX << endl; 
    throw jiterator_error("end of samples!");
  }
  if (_mode == 1) {
    cvConvertImage(_frame,_gray,0);
    //cvCvtColor(_frame,_gray,CV_RGB2GRAY);
    //cvEqualizeHist(_gray, _gray );
    //cvCvtColor(_frame,_hsv,CV_RGB2HSV);
    //cvCvtColor(_frame,_gray,CV_HSV2GRAY) 
    int k = 0;
    for (unsigned i = 0; i < _width; i++) {
      for (unsigned j = 0; j < _height; j++) {
	gsl_vector_float_set(_vector, k, ((uchar*)(_gray->imageData + _gray->widthStep*j))[i]);
	k++;
      }
    }
  } else { 
    //TODO - Linear-Color-Representation?
  }
  _increment();    // _frameX++
  if (_frameX == _frameNumber) _endOfSamples = true;
  // cout << _frameX  << endl;
  return _vector;
}

void VideoFeature::reset()
{
  // cout << "Reseting in VideoFeature" << endl;

  _endOfSamples = false;
  VectorFloatFeatureStream::reset();
}

void VideoFeature::read(const String& filename, int from, int to)            // you should test this function!!
{ 
  //reset();
  if (_fileName != filename) {
    //reset();
    if (_capture != NULL)  cvReleaseCapture(&_capture);
    _capture = cvCreateFileCapture(filename);
    if(_capture == NULL)
      throw jio_error("Could not open file %s.", filename.c_str());

    _frame  = cvQueryFrame( _capture ); //read first frame!
    if (_width != (int)cvGetCaptureProperty( _capture, CV_CAP_PROP_FRAME_WIDTH)) {
      throw jindex_error("Problem in Feature %s: %d != %d\n",
			 name().c_str(), _width,(int)cvGetCaptureProperty( _capture, CV_CAP_PROP_FRAME_WIDTH) );      
    }
    if (_height != (int)cvGetCaptureProperty( _capture, CV_CAP_PROP_FRAME_HEIGHT)){
      throw jindex_error("Problem in Feature %s: %d != %d\n",
			 name().c_str(), _height,(int)cvGetCaptureProperty( _capture, CV_CAP_PROP_FRAME_HEIGHT) );      
    }
    if (_gray) cvReleaseImage(&_gray); 
    _gray   = cvCreateImage(cvSize(_width,_height),_frame->depth,_mode);
  }
  if (_from != from) {
    for (int i = _from; i < from; i++) {
      _frame = cvQueryFrame( _capture );
      _from++;
    }
  }
  _filename    = filename;
  _from        = from;
  _to          = to;
  _frameNumber = _to - _from + 1;
}

int VideoFeature::Frames()
{
  int  nFrames;
  char tempSize[4];

  // Trying to open the video file
  ifstream  videoFile( _filename, ios::in | ios::binary);

  // Checking the availablity of the file
  if (videoFile == NULL)
    throw jio_error("Could not open file %s.", _filename.c_str());

  // get the number of frames
  videoFile.seekg( 0x30 , ios::beg );
  videoFile.read( tempSize , 4 );
  nFrames = (unsigned char ) tempSize[0] + 0x100*(unsigned char ) tempSize[1] + 0x10000*(unsigned char ) tempSize[2] +    0x1000000*(unsigned char ) tempSize[3];
  videoFile.close(  );
  return nFrames;
}


// ----- methods for class `ImageROI' -----
//
ImageROI::ImageROI(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height,unsigned x,unsigned y,unsigned w,unsigned h, const String& nm)
  : VectorFloatFeatureStream(h*w,nm), _src(src), _endOfSamples(false), 
    _width(width), _height(height), _w(w), _h(h), _srcVec(gsl_vector_float_calloc(_width*_height)) 
{
  _x_pos = x;
  _y_pos = y;
  cout << "ImageROI" << endl;  
}

ImageROI::~ImageROI()
{
  gsl_vector_float_free(_srcVec);
}

void ImageROI::setROI(int x, int y)
{
  _x_pos = x;
  _y_pos = y;
}

const gsl_vector_float* ImageROI::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));
  _increment();  // _frameX++

  int i,j,k,l;
  k=0;  
  l=0;
  for (i=0; i <_width ; i++){
      for (j=0; j <_height; j++){
	if (i >= _x_pos && i < _x_pos+_w){
	  if (j >= _y_pos && j < _y_pos+_h ){
	    gsl_vector_float_set(_vector,l,gsl_vector_float_get(_srcVec,k));
	    l++;
	  }
	}
	k++;
      }
      if ( i >= _x_pos && _y_pos+_h >= _height) l = l + (_y_pos+_h-_height);
  }
  
  //  gsl_vector_float_free(srcVec);
  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* ImageROI::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void ImageROI::reset()
{
  // cout << "Reseting in ImageROI" << endl;

  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `ImageSmooth' -----
//
ImageSmooth::ImageSmooth(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int smoothtype, int param1, int param2, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _endOfSamples(false),
  _width(width), _height(height),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _imgf(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _smoothtype(smoothtype), _param1(param1), _param2(param2)
{
  cout << "ImageSmooth" << endl;
}

ImageSmooth::~ImageSmooth()
{
  cvReleaseImage(&_img);
  cvReleaseImage(&_imgf);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* ImageSmooth::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }

  cvSmooth(_img, _imgf, _smoothtype, _param1, _param2);                  //Filter
                   // 0 = CV_BLUR_NO_SCALE (simple blur with no scaling)
                   // 1 = CV_BLUR (simple blur)
                   // 2 = CV_GAUSSIAN (gaussian blur)
                   // 3 = CV_MEDIAN (median blur)
                   // 4 = CV_BILATERAL (bilateral filter)
                   // see the cv.h file in /opencv/include/opencv/
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_imgf->imageData + _imgf->widthStep*j))[i]);
      l++;
    }

  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* ImageSmooth::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void ImageSmooth::reset()
{
  // cout << "Reseting in ImageSmooth" << endl;

  _src->reset();
  VectorFloatFeatureStream::reset();
}

// ----- methods for class `ImageMorphology' -----
//
ImageMorphology::ImageMorphology(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int type, int param, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _endOfSamples(false),
  _width(width), _height(height),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _imgf(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _type(type),_param(param)
{
  cout << "ImageMorphology" << endl;
}

ImageMorphology::~ImageMorphology()
{
  cvReleaseImage(&_img);
  cvReleaseImage(&_imgf);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* ImageMorphology::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }

  if (_type == 0){
    cvErode(_img,_imgf,NULL,_param);
  } else {
    cvDilate(_img,_imgf,NULL,_param);
  }

  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_imgf->imageData + _imgf->widthStep*j))[i]);
      l++;
    }

  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* ImageMorphology::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void ImageMorphology::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `ImageMorphologyEx' -----
//
ImageMorphologyEx::ImageMorphologyEx(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int type, int param, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _endOfSamples(false),
  _width(width), _height(height),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _imgf(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _type(type),_param(param)
{
  cout << "ImageMorphologyEx" << endl;
  _temp = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1);
}

ImageMorphologyEx::~ImageMorphologyEx()
{
  cvReleaseImage(&_img);
  cvReleaseImage(&_imgf);
  cvReleaseImage(&_temp);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* ImageMorphologyEx::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }

  cvMorphologyEx(_img,_imgf,_temp,NULL,_type,_param);
  //_type = CV_MOP_OPEN
  //_type = CV_MOP_CLOSE
  //_type = CV_MOP_GRADIENT
  //_type = CV_MOP_TOPHAT
  //_type = CV_MOP_BLACKHAT

  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_imgf->imageData + _imgf->widthStep*j))[i]);
      l++;
    }

  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* ImageMorphologyEx::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void ImageMorphologyEx::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}

// ----- methods for class `Canny' -----
//
Canny::Canny(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int param0, int param1, int param2, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _endOfSamples(false),
  _width(width), _height(height),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _imgf(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _param0(param0), _param1(param1), _param2(param2)
{
  cout << "Canny" << endl;
}

Canny::~Canny()
{
  cvReleaseImage(&_img);
  cvReleaseImage(&_imgf);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* Canny::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }
  
  cvCanny(_img,_imgf, _param0, _param1, _param2);

  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_imgf->imageData + _imgf->widthStep*j))[i]);
      l++;
    }

  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* Canny::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}


void Canny::reset()
{
  // cout << "Reseting in Canny" << endl;

  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `ImageThreshold' -----
//
ImageThreshold::ImageThreshold(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, float param0, int param1, int param2, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _endOfSamples(false),
  _width(width), _height(height),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _imgf(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _param0(param0), _param1(param1), _param2(param2)
{
  cout << "ImageThreshold" << endl;
}

ImageThreshold::~ImageThreshold()
{
  cvReleaseImage(&_img);
  cvReleaseImage(&_imgf);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* ImageThreshold::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }
  
  cvThreshold(_img,_imgf, _param0, _param1, _param2);
  //CV_THRESH_BINARY
  //CV_THRESH_BINARY_INV
  //CV_THRESH_TRUNC
  //CV_THRESH_TOZERO_INV
  //CV_THRESH_TOZERO

  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_imgf->imageData + _imgf->widthStep*j))[i]);
      l++;
    }

  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* ImageThreshold::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}


void ImageThreshold::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `ImageShow' -----
//
/*Do not use this funktion within python - there is a problem with the window-handle or threading.... */
ImageShow::ImageShow(const VectorFloatFeatureStreamPtr& src,unsigned width, unsigned height, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src),
  _width(width), _height(height),
  _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1)),
  _srcVec(gsl_vector_float_calloc(_width*_height)),
  _endOfSamples(false)
{
  cout << "ImageShow" << endl;
}

ImageShow::~ImageShow()
{
  cvReleaseImage(&_img);
  gsl_vector_float_free(_srcVec);
}

const gsl_vector_float* ImageShow::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  // const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++
  

  unsigned l = 0;
  for (unsigned i = 0; i < _width ; i++) {
    for (unsigned j = 0; j < _height; j++) {
      gsl_vector_float_set(_vector, l, gsl_vector_float_get(_srcVec, l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i] = gsl_vector_float_get(_srcVec,l);
      l++;
    }
  }
  /*char buffer [50];
  sprintf (buffer, "/home/mgeorges/Desktop/test%d.png", _frameX);
  if( !cvSaveImage(buffer, img) )  // funktioniert nicht - warum??
  {
    fprintf(stderr, "failed to write image file\n");
  }*/

  cvNamedWindow("video", CV_WINDOW_AUTOSIZE);
  cvShowImage("video", _img);
  cvWaitKey( 10 )& 0xFF;                               //??? FUNKTIONIERT NICHT - Problem mit Python?
  cvDestroyWindow("video");

  //  gsl_vector_float_free(srcVec);
  // cout << _frameX  << endl;
  return _vector;
}

void ImageShow::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `SaveImage' -----
//
SaveImage::SaveImage(unsigned width, unsigned height)
  : _width(width), _height(height),
    _img(cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1))
{
  cout << "SaveImage" << endl;
}

//Destructor
SaveImage::~SaveImage()
{
  cvReleaseImage(&_img);
}

void SaveImage::save(const gsl_vector_float* V,const String& filename)
{
  cout << "bin da" << endl;
  unsigned l = 0;
  for (unsigned i = 0; i <_width ; i++) {
    for (unsigned j = 0; j <_height; j++) {
      ((uchar*)(_img->imageData + _img->widthStep*j))[i] = gsl_vector_float_get(V, l);
      l++;
    }
  }
  if( !cvSaveImage(filename, _img) )  // funktioniert nicht - warum??
  {
    fprintf(stderr, "failed to write image file\n");
  }
  /*
  if(!cvSaveImage(_filename, _img))  // funktioniert nicht - warum??
  {
    fprintf(stderr, "failed to write image file\n");
  }
  */
}
 
void SaveImage::savedouble(const gsl_vector* V,const String& filename)
{
  int l=0;
  for (unsigned i = 0; i < _width ; i++) {
    for (unsigned j = 0; j < _height; j++) {
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_get(V, l);
      l++;
    }
  }
  //char buffer [100];
  //sprintf (buffer, "%s%s", _filename,filename);
  if( !cvSaveImage(filename, _img) )  // funktioniert nicht - warum??
  {
    fprintf(stderr, "failed to write image file\n");
  }
  /*
  if(!cvSaveImage(_filename, _img))  // funktioniert nicht - warum??
  {
    fprintf(stderr, "failed to write image file\n");
  }
  */
}


// ----- methods for class `ImageDetection' -----
//
ImageDetection::ImageDetection(VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, unsigned w, unsigned h,
			       const String& filename, double scale_factor, int min_neighbors, int flags, int min_sizeX, int min_sizeY, const String& nm) :
  VectorFloatFeatureStream(w*h,nm), _src(src),
  _width(width), _height(height), _w(w), _h(h),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _img(cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, 1)),
  _filename(filename),
  _pCascade((CvHaarClassifierCascade *) cvLoad(_filename.c_str(), 0, 0, 0 )),
  _pStorage(cvCreateMemStorage(0)), _pRectSeq(NULL),
  _scale_factor(scale_factor), _min_neighbors(min_neighbors),
  _flags(flags), _min_size(cvSize(min_sizeX,min_sizeY)), 
  _x_pos(0), _y_pos(0), 
  _his(50), _x_his(gsl_vector_calloc(_his)), _y_his(gsl_vector_calloc(_his)),
  _endOfSamples(false)
{
  cout << "ImageDetection" << endl;

  if (_pStorage == NULL || _pCascade == NULL) {
    cout << "ERROR: detection, pStorage is " << _pStorage << " and pCascade is " << _pCascade << " !!" << endl;
    throw jio_error("Could not open file %s.");
  }
}

ImageDetection::~ImageDetection()
{
  if(_pCascade) cvReleaseHaarClassifierCascade(&_pCascade);
  if(_pStorage) cvReleaseMemStorage(&_pStorage);
  cvReleaseImage(&_img);
  gsl_vector_float_free(_srcVec);

  gsl_vector_free(_x_his);
  gsl_vector_free(_y_his);
}

gsl_vector_float* ImageDetection::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}


const gsl_vector_float* ImageDetection::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  // const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  unsigned l = 0;
  for (unsigned i = 0; i < _width ; i++) {
    for (unsigned j = 0; j < _height; j++) {
      //gsl_vector_float_set(_vector,l,gsl_vector_float_get(srcVec,l));
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec, l);
      l++;
    }
  }
 
  cvEqualizeHist(_img, _img);
  cvClearMemStorage(_pStorage);
  _pRectSeq = cvHaarDetectObjects(_img, _pCascade, _pStorage, _scale_factor, _min_neighbors, _flags, _min_size);

  CvPoint pt;
  CvRect *rr;
  int x=0;
  int y=0;
  unsigned number = (_pRectSeq ? _pRectSeq->total : 0);
  //cout << "number: " << number << " ";
  for(unsigned i = 0; i < number; i++) {
    rr = (CvRect*)cvGetSeqElem(_pRectSeq,i); 
    //Average over all possible mouth-regions
    x = x + (rr->x+(rr->width/2 ));
    y = y + (rr->y+(rr->height/2));
    pt.x = (rr->x+(rr->width/2 ));
    pt.y = (rr->y+(rr->height/2));
    cvRectangle(_img, pt, pt, CV_RGB(255,0,0), 3, 4, 0); // RED-DOT in the middle of the mouth!
    /*  //Take the lowest possition in the frame!! 
    if (y < (rr->y+(rr->height/2))){
	  x = (rr->x+(rr->width/2 ));
	  y = (rr->y+(rr->height/2));
	  pt.x = x;
	  pt.y = y;
	  cvRectangle(img,pt,pt,CV_RGB(255,0,0),3,4,0); // RED-DOT in the middle of the mouth!
	} 
    */
   }
  int x_pos;
  int y_pos;
  if (number==0){
    x_pos = _x_pos;
    y_pos = _y_pos;
  } else {
    x = x/number;
    y = y/number;
    x_pos = x -(_w/2);
    y_pos = y -(_h/2);
  }
    
  for(unsigned i = 1; i < _his; i++){
    gsl_vector_swap_elements(_x_his, i-1, i);
    gsl_vector_swap_elements(_y_his, i-1, i);
  }

  gsl_vector_set(_x_his, _his-1, x_pos);
  gsl_vector_set(_y_his, _his-1, y_pos);

  /*
  gsl_permutation *tv = gsl_permutation_calloc(_his);
  gsl_sort_vector_index(tv,x_his);
  x_pos = gsl_vector_get(x_his,gsl_permutation_get(tv,_his/2));
  gsl_vector_set(x_his,_his-1,x_pos);
  gsl_sort_vector_index(tv,y_his);
  y_pos = gsl_vector_get(y_his,gsl_permutation_get(tv,_his/2));
  gsl_vector_set(y_his,_his-1,y_pos); 
  */

  _tmp_x = 0;
  _tmp_y = 0;
  for (unsigned i = 0; i < _his; i++) {
    _tmp_x = _tmp_x + gsl_vector_get(_x_his, i);
    _tmp_y = _tmp_y + gsl_vector_get(_y_his, i);
  }

  _tmp_x =  _tmp_x/_his;
  _tmp_y =  _tmp_y/_his;

  //Box Smoothing over history
  //x_pos = _tmp_x;
  //y_pos = _tmp_y;

  //Linear Fitting over the history and return the value at possition 3/2 of the history 
  //x_pos = _linearnext(x_his);
  //y_pos = _linearnext(y_his);

  //Exponential Smoothing over all past values
  _x_pos = 0.9*_x_pos+(1.0-0.9)*x_pos;
  _y_pos = 0.9*_y_pos+(1.0-0.9)*y_pos; 
  x_pos = _x_pos;
  y_pos = _y_pos;

  _x_pos = x_pos;
  _y_pos = y_pos;

  /*
  cout << " x: " << _x_pos << " x-Var: " << gsl_vector_max(x_his) - gsl_vector_min(x_his)<< " meanX: " << _tmp_x 
       << " y: " << _y_pos << " y-Var: " << gsl_vector_max(y_his) - gsl_vector_min(y_his)<< " meanY: " << _tmp_y << endl;
   //cout << "x_pos: " << x_pos << " y_pos: " << y_pos << endl;
  */
 
 gsl_vector_float_set_zero(_vector);

 l=0;
 for (unsigned i = 0; i < _w; i++) {
   for (unsigned j = 0; j < _h; j++) {
     x = i+x_pos;
     y = j+y_pos;
     if (x>=0 && y >= 0){
       if (x < _width && y < _height){
	 gsl_vector_float_set(_vector,l,((uchar*)(_img->imageData + _img->widthStep*y))[x]);
       }
     }
     l++;
   }
 }
 return _vector;
}

double ImageDetection::_linearnext(gsl_vector * v){
  int i;
  int n = v->size;
  double xq = 0.0;
  for(i=0;i<n;i++){
    xq = xq + (i+1);
  }
  xq = xq/n;

  double yq=0.0;
  for(i=0;i<n;i++){
    yq = yq + gsl_vector_get(v,i);
  }
  yq = yq/n;

  double b1=0.0;
  for(i=0;i<n;i++){
    b1 = b1 + (gsl_vector_get(v,i)-yq)*((i+1)-xq);
  }

  double b2=0.0;
  for(i=0;i<n;i++){
    b2 = b2 + ((i+1)-xq)*((i+1)-xq);
  }
  double b;
  b = b1/b2;

  double a;
  a = yq - b*xq;

  return (a + b*((n+1)*2/3));
  
}



void ImageDetection::reset()
{
  // cout << "Reseting in ImageDetection" << endl;

  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `FaceDetection' -----
//
FaceDetection::FaceDetection(VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, int region,
			     const String& filename_eye  , double scale_factor_eye  , int min_neighbors_eye  , int flags_eye  , int min_sizeX_eye  , int min_sizeY_eye  ,
			     const String& filename_nose , double scale_factor_nose , int min_neighbors_nose , int flags_nose , int min_sizeX_nose , int min_sizeY_nose ,
			     const String& filename_mouth, double scale_factor_mouth, int min_neighbors_mouth, int flags_mouth, int min_sizeX_mouth, int min_sizeY_mouth,
			     const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src)
{
}

//Destructor
FaceDetection::~FaceDetection()
{
}

const gsl_vector_float* FaceDetection::next(int frameX)
{
#if 0
fftw_complex* img1 = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * 360 * 240);
#endif
}

void FaceDetection::reset()
{

}

// ----- methods for class `ImageCentering' -----
//
ImageCentering::ImageCentering(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src)
{
  cout << "ImageCentering" << endl;
  _width  = width;
  _height = height;
  _endOfSamples = false;
  _srcVec = gsl_vector_float_calloc(_width*_height);
  _img = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1);
  _gmi = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U,1);
  _out = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_64F,1);
}

//Destructor
ImageCentering::~ImageCentering()
{
  gsl_vector_float_free(_srcVec);
  cvReleaseImage(&_img);
  cvReleaseImage(&_gmi);  
  cvReleaseImage(&_out);
}

const gsl_vector_float* ImageCentering::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  double tmp;
  int step     = _img->widthStep;
  int fft_size = _width * _height;
  fftw_complex *img1 = (fftw_complex*)fftw_malloc( sizeof( fftw_complex ) * _width * _height);
  fftw_complex *img2 = (fftw_complex*)fftw_malloc( sizeof( fftw_complex ) * _width * _height);
  fftw_complex *res  = (fftw_complex*)fftw_malloc( sizeof( fftw_complex ) * _width * _height);    
  
  fftw_plan fft_img1 = fftw_plan_dft_2d(_width, _height, img1, img1, FFTW_FORWARD,  FFTW_ESTIMATE );
  fftw_plan fft_img2 = fftw_plan_dft_2d(_width, _height, img2, img2, FFTW_FORWARD,  FFTW_ESTIMATE );
  fftw_plan ifft_res = fftw_plan_dft_2d(_width, _height, res,  res,  FFTW_BACKWARD, FFTW_ESTIMATE );
  
  unsigned l = 0;
  int m = _width - 1;
  for (unsigned i = 0, k = 0; i < _width; i++) {
    for (unsigned j = 0; j < _height; j++, k++) {
      img1[k][0] = (double)gsl_vector_float_get(_srcVec,l);
      img1[k][1] = 0.0;
      img2[k][0] = (double)gsl_vector_float_get(_srcVec,m*_height+j);
      img2[k][1] = 0.0;
      l++;
    }
    m--;
  }
  fftw_execute( fft_img1 );
  fftw_execute( fft_img2 );
  /* obtain the cross power spectrum */
  for (unsigned i = 0; i < fft_size; i++) {
    res[i][0] = img1[i][0] * img2[i][0];
    res[i][1] = img1[i][1] * (-img2[i][1]);
    tmp = sqrt( pow( res[i][0], 2.0 ) + pow( res[i][1], 2.0 ) );
    res[i][0] /= tmp;
    res[i][1] /= tmp;
    //    res[i][0] = ( img2[i][0] * img1[i][0] ) - ( img2[i][1] * ( -img1[i][1] ) );
    //    res[i][1] = ( img2[i][0] * ( -img1[i][1] ) ) + ( img2[i][1] * img1[i][0] );
    //    tmp = sqrt( pow( res[i][0], 2.0 ) + pow( res[i][1], 2.0 ) );
    //    res[i][0] /= tmp;
    //    res[i][1] /= tmp;
  }
  fftw_execute(ifft_res);
  
  for (unsigned i = 0 ; i < fft_size ; i++ ) {
    //  _out->imageData[i] = res[i][0] / (double)fft_size;
    gsl_vector_float_set(_vector, i, img1[i][1]); /// (double)fft_size);
    //gsl_vector_float_set(_vector,i,img1[i][0]);     
  }
  /* deallocate FFTW arrays and plans */
  fftw_destroy_plan(fft_img1);
  fftw_destroy_plan(fft_img2);
  fftw_destroy_plan(ifft_res);
  fftw_free(img1);
  fftw_free(img2);
  fftw_free(res);

  cout << _frameX  << endl;
  return _vector; 
}

/*

  gsl_vector_float *h = gsl_vector_float_calloc(_height);
  gsl_vector_float *w = gsl_vector_float_calloc(_height);
  gsl_vector_float *sumh = gsl_vector_float_calloc(_width);
  gsl_vector_float *sumw = gsl_vector_float_calloc(_width);
  m = _width-1;
  for (i=0; i <_width ; i++){
    gsl_matrix_float_get_col(h,mat,i);  
    gsl_matrix_float_get_col(w,mat,m);
    for(j=0; j < _height; j++){
      gsl_vector_float_set(sumh,i,gsl_vector_float_get(sumh,i)+gsl_vector_float_get(h, j));
      gsl_vector_float_set(sumw,i,gsl_vector_float_get(sumw,i)+gsl_vector_float_get(w, j));
    }
    //gsl_matrix_float_set_row(cat,i,h);
    m--;
  }

  gsl_vector_float *b = gsl_vector_float_calloc(2*_height);
  float center = 20; 
 //center = 20;
  for (i=0; i <_width ; i++){
    int value = i+(int)center-1;
    if (value >= _width) value = value - _width ;
    if (value >= _width) value = _width -1 ;
    // cout << i << " and "<< value << endl;
    //gsl_matrix_float_get_col(h,mat,i);  
    
    gsl_matrix_float_swap_columns(mat,value,i); 
  }


  l=0;
  for (i=0; i <_width ; i++){
    for (j=0; j <_height; j++){
      gsl_vector_float_set(wrkVec,l,gsl_matrix_float_get(mat,j,i));
      //gsl_vector_float_set(wrkVec,l,gsl_vector_float_get(sumh,i)*0.02);
      //gsl_vector_float_set(wrkVec,l,abs(gsl_vector_float_get(srcVec,l)-gsl_vector_float_get(wrkVec,l)));
      l++;
    }
  }
  
   
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      //gsl_vector_float_set(_vector,l,((uchar*)(imgf->imageData + imgf->widthStep*j))[i]);
      gsl_vector_float_set(_vector,l,gsl_vector_float_get(wrkVec,l));
      l++;
    }
 */

gsl_vector_float* ImageCentering::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void ImageCentering::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `LinearInterpolation' -----
//
LinearInterpolation::LinearInterpolation(const VectorFloatFeatureStreamPtr& src, double fps_src, double fps_dest, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), _src(src),
    _DeltaTs(1.0 / fps_src), _DeltaTd(1.0 / fps_dest), 
    _sourceFrameX(0), _x_n(gsl_vector_float_calloc(_size)),
    _endOfSamples(false)
{
  cout << "LinearInterpolation" << endl;
}

LinearInterpolation::~LinearInterpolation()
{
  gsl_vector_float_free(_x_n);
}

const gsl_vector_float* LinearInterpolation::next(int frameX)
{

  if (_endOfSamples) {

    // cout << "Read " << _frameX << " blocks of interpolated video data." << endl;

    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  double destTime = (_frameX + 1) * _DeltaTd;
  double srcTime  = _sourceFrameX * _DeltaTs;

  // Time for a new source frame?
  if (destTime >= srcTime + _DeltaTs || _sourceFrameX == 0) {

    // cout << "Incrementing source vector at time " << srcTime << endl;

    gsl_vector_float_memcpy(_x_n, _src->next(_sourceFrameX));
    _sourceFrameX++;  srcTime += _DeltaTs;
  }
  const gsl_vector_float* _x_n1 = _src->next(_sourceFrameX);

  // Perform interpolation
  double interpolationFactor = (destTime - srcTime) / _DeltaTs;
  for (unsigned dimX = 0; dimX < size(); dimX++) {
    double xn			= gsl_vector_float_get(_x_n, dimX);
    double xn1			= gsl_vector_float_get(_x_n1, dimX);
    double interpolationValue	= interpolationFactor * (xn1 - xn);

    gsl_vector_float_set(_vector, dimX, interpolationValue);
  }

  _increment();  // _frameX++

  return _vector;
}

void LinearInterpolation::reset()
{
  _src->reset();
  _sourceFrameX = 0;
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `OpticalFlowFeature' -----
//
OpticalFlowFeature::OpticalFlowFeature(const VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, const char* filename, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src)
{
  cout << "OpticalFlowFeature" << endl;
  _width  = width;
  _height = height;
  _srcVec = gsl_vector_float_calloc(_width*_height);  // combin of _ux and _uy!!2*
  _nx = _width;
  _ny = _height;
  _bx = 1;
  _by = 1;
  alloc_matrix(&_gray, _nx+2*_bx, _ny+2*_by);
  alloc_matrix(&_ux  , _nx+2*_bx, _ny+2*_by);
  alloc_matrix(&_uy  , _nx+2*_bx, _ny+2*_by);
  //read file
  fstream f;
  string str;
  string name;
  string value;
  f.open(filename, ios::in);
  size_t pos;
  while (!f.eof())
    {
      getline(f,str);
      pos = str.find_first_of("=");
      if (pos < str.length()){
	name = str.substr(0,pos);
	if (name.substr(0,1).compare("/")!=0){
	  value = str.substr(pos+1);
	  pos = value.find_first_of(";");
	  value = value.substr(0,pos);
	  //cout << name << " = " << value << endl;
	       if (name.compare("nx") == 0)                   _nx                  = StringToInt(value);
	  else if (name.compare("ny") == 0)                   _ny                  = StringToInt(value);
	  else if (name.compare("bx") == 0)                   _bx                  = StringToInt(value);
	  else if (name.compare("by") == 0)                   _by                  = StringToInt(value);
	  else if (name.compare("hx") == 0)                   hy                   = StringToFloat(value);
	  else if (name.compare("hy") == 0)                   hy                   = StringToFloat(value);
	  else if (name.compare("m_type_d") == 0)             m_type_d             = StringToInt(value);
	  else if (name.compare("m_type_s") == 0)             m_type_s             = StringToInt(value);
	  else if (name.compare("m_gamma_ofc1") == 0)         m_gamma_ofc1         = StringToFloat(value);
	  else if (name.compare("m_gamma_ofc2") == 0)         m_gamma_ofc2         = StringToFloat(value);
	  else if (name.compare("m_gamma_ofc3") == 0)         m_gamma_ofc3         = StringToFloat(value);
	  else if (name.compare("m_gamma_gradnorm") == 0)     m_gamma_gradnorm     = StringToFloat(value);
	  else if (name.compare("m_gamma_laplace") == 0)      m_gamma_laplace      = StringToFloat(value);
	  else if (name.compare("m_gamma_hessdet") == 0)      m_gamma_hessdet      = StringToFloat(value);
	  else if (name.compare("m_function_d") == 0)         m_function_d         = StringToInt(value);
	  else if (name.compare("m_function_s") == 0)         m_function_s         = StringToInt(value);
	  else if (name.compare("m_epsilon_d") == 0)          m_epsilon_d          = StringToFloat(value);
	  else if (name.compare("m_epsilon_s") == 0)          m_epsilon_s          = StringToFloat(value);
	  else if (name.compare("m_power_d") == 0)            m_power_d            = StringToFloat(value);
	  else if (name.compare("m_power_s") == 0)            m_power_s            = StringToFloat(value);
	  else if (name.compare("m_rhox") == 0)               m_rhox               = StringToFloat(value);
	  else if (name.compare("m_rhoy") == 0)               m_rhoy               = StringToFloat(value);
	  else if (name.compare("m_rhox_F") == 0)             m_rhox_F             = StringToFloat(value);
	  else if (name.compare("m_rhoy_F") == 0)             m_rhoy_F             = StringToFloat(value);
	  else if (name.compare("m_sigma_F") == 0)            m_sigma_F            = StringToFloat(value);
	  else if (name.compare("m_alpha") == 0)              m_alpha              = StringToFloat(value);
	  else if (name.compare("m_color") == 0)              m_color              = StringToInt(value);
	  else if (name.compare("n_solver") == 0)             n_solver             = StringToInt(value);
	  else if (name.compare("n_max_rec_depth") == 0)      n_max_rec_depth      = StringToInt(value);
	  else if (name.compare("n_mg_res_prob") == 0)        n_mg_res_prob        = StringToInt(value);
	  else if (name.compare("n_theta") == 0)              n_theta              = StringToInt(value);
	  else if (name.compare("n_mg_cycles") == 0)          n_mg_cycles          = StringToInt(value);
	  else if (name.compare("n_mg_solver") == 0)          n_mg_solver          = StringToInt(value);
	  else if (name.compare("n_mg_pre_relax") == 0)       n_mg_pre_relax       = StringToInt(value);
	  else if (name.compare("n_mg_post_relax") == 0)      n_mg_post_relax      = StringToInt(value);
	  else if (name.compare("n_mg_rec_calls") == 0)       n_mg_rec_calls       = StringToInt(value);
	  else if (name.compare("n_mg_centered") == 0)        n_mg_centered        = StringToInt(value);
	  else if (name.compare("n_mg_mat_order_r") == 0)     n_mg_mat_order_r     = StringToInt(value);
	  else if (name.compare("n_mg_cas_order_r") == 0)     n_mg_cas_order_r     = StringToInt(value);
	  else if (name.compare("n_mg_cas_order_p") == 0)     n_mg_cas_order_p     = StringToInt(value);
	  else if (name.compare("n_mg_cor_order_r") == 0)     n_mg_cor_order_r     = StringToInt(value);
	  else if (name.compare("n_mg_cor_order_p") == 0)     n_mg_cor_order_p     = StringToInt(value);
	  else if (name.compare("n_iter_out") == 0)           n_iter_out           = StringToInt(value);
	  else if (name.compare("n_iter_in") == 0)            n_iter_in            = StringToInt(value);
	  else if (name.compare("n_omega") == 0)              n_omega              = StringToFloat(value);
	  else if (name.compare("n_tau") == 0)                n_tau                = StringToFloat(value);
	  else if (name.compare("n_warp_eta") == 0)           n_warp_eta           = StringToFloat(value);
	  else if (name.compare("n_warp_steps") == 0)         n_warp_steps         = StringToInt(value);
	  else if (name.compare("n_warp_max_rec_depth") == 0) n_warp_max_rec_depth = StringToInt(value);
	  else if (name.compare("n_warp_scaling") == 0)       n_warp_scaling       = StringToInt(value);
	  else if (name.compare("n_warp_rem") == 0)           n_warp_rem           = StringToInt(value);
	  else if (name.compare("n_warp_centered") == 0)      n_warp_centered      = StringToInt(value);
	  else if (name.compare("n_warp_order_r") == 0)       n_warp_order_r       = StringToInt(value);
	  else if (name.compare("n_warp_order_p") == 0)       n_warp_order_p       = StringToInt(value);
	  else if (name.compare("n_tensor_eps") == 0)         n_tensor_eps         = StringToFloat(value);
	  else if (name.compare("i_rel_res_flag") == 0)       i_rel_res_flag       = StringToInt(value);
	  else if (name.compare("i_rel_err_flag") == 0)       i_rel_err_flag       = StringToInt(value);
	  else if (name.compare("i_time_flag") == 0)          i_time_flag          = StringToInt(value);
	  else if (name.compare("info_flag") == 0)            info_flag            = StringToInt(value);
	  else if (name.compare("info_step") == 0)            info_step            = StringToInt(value);
	  else if (name.compare("write_flag") == 0)           write_flag           = StringToInt(value);
	  else if (name.compare("write_step") == 0)           write_step           = StringToInt(value);
	  else if (name.compare("frame_nr") == 0)             frame_nr             = StringToInt(value);
	}
      }
    }
  f.close();
}

//Destructor
OpticalFlowFeature::~OpticalFlowFeature()
{
  gsl_vector_float_free(_srcVec);
  disalloc_matrix(_gray, _nx+2*_bx, _ny+2*_by);
  disalloc_matrix(_ux  , _nx+2*_bx, _ny+2*_by);
  disalloc_matrix(_uy  , _nx+2*_bx, _ny+2*_by);
}

int OpticalFlowFeature::StringToInt(const std::string& str_input){
  const int default_value = 0;
  std::stringstream s_str(str_input);
  int value = default_value;
  if (!(s_str>>value))
  { //error during conversion -> return given default value
    return default_value;
  }
  return value;
}

float OpticalFlowFeature::StringToFloat(const std::string& str_input){
  const float default_value = 0.0;
  std::stringstream s_str(str_input);
  float value = default_value;
  if (!(s_str>>value))
  { //error during conversion -> return given default value
    return default_value;
  }
  return value;
}

const gsl_vector_float* OpticalFlowFeature::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++

  unsigned l = 0;
  for (unsigned j = 1; j <= _ny; j++) {
    for (unsigned i = 1; i <= _nx; i++) {
      _gray[i][j] = gsl_vector_float_get(_srcVec,l);
      l++;
    }
  }

  // Combine _ux and _uy and write it in _vector
  l = 0;
  for (unsigned j = 1; j <= _ny; j++) {
    for (unsigned i = 1; i <= _nx; i++) {
      gsl_vector_float_set(_srcVec,l,_gray[i][j]);
      l++;
    }
  }

  l=0;
  for (unsigned i = 0; i <_width; i++) {
    for (unsigned j = 0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,gsl_vector_float_get(_srcVec,l));
      l++;
    }
  }
  
  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* OpticalFlowFeature::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}

void OpticalFlowFeature::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}

void OpticalFlowFeature::alloc_matrix(float ***matrix, long nx, long ny)
{
  long i;
  *matrix = (float **) malloc (nx * sizeof(float *));
  if (*matrix == NULL)
    {
      cout << "not enough memory: x" << endl;	 
    }
  for (i=0; i<nx; i++){
    (*matrix)[i] = (float *) malloc (ny * sizeof(float));
    if ((*matrix)[i] == NULL)
      cout << "not enough memory: y" << endl;
       }
  return;
}

void OpticalFlowFeature::disalloc_matrix(float **matrix, long nx, long ny)
{
  long i;
  for (i=0; i<nx; i++)
    free(matrix[i]);
  free(matrix);
  return;
}


// ----- methods for class `SnakeImage' -----
//
SnakeImage::SnakeImage(const VectorFloatFeatureStreamPtr& src,unsigned width,unsigned height, float alpha, float beta, float gamma, const String& nm) :
  VectorFloatFeatureStream(width*height,nm), _src(src)
{
  cout << "SnakeImage" << endl;
  _width  = width;
  _height = height;
  _img    = cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, 1);
  _srcVec = gsl_vector_float_calloc(_width*_height);  // combin of _ux and _uy!!2*
  _alpha     = alpha;
  _beta      = beta;
  _gamma     = gamma;
}

//Destructor
SnakeImage::~SnakeImage()
{
  gsl_vector_float_free(_srcVec);
}


const gsl_vector_float* SnakeImage::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1));  
  _increment();  // _frameX++


  int i,j,l;
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      ((uchar*)(_img->imageData + _img->widthStep*j))[i]= gsl_vector_float_get(_srcVec,l);
      l++;
    }

  int length       = 10;
  CvPoint * points = (CvPoint *)malloc(length * sizeof(CvPoint));
  cout << "in: ";
  for(i=0;i<length;i++){
    points[i].x = (i+1)*5+25;
    points[i].y = 25;
    cout << points[i].x << " " << points[i].y << ".";
  }

  CvSize win;
  win.width        = 17;
  win.height       = 17;
  CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,20,0.3);
  //criteria.maxCount = 100;
  //criteria.epsilon  = 1;
  //criteria.type = CV_TERMCRIT_EPS | CV_TERMCRIT_ITER;
  cvSnakeImage(_img,points,length,&_alpha,&_beta,&_gamma,CV_VALUE,win,criteria,1);
  cout << "out: ";
  for(i=0;i<length;i++){
    cout << points[i].x << " " << points[i].y << ".";
    cvRectangle(_img,points[i],points[i],CV_RGB(0,0,0),3,4,0);
    //cvRectangle(_img,points[i],points[i],CV_RGB(255,255,255),3,4,0);
  }
  cout << endl;

  cvPolyLine( _img, &points, 
	      &length, 
	      2,
	      0, 
	      CV_RGB(255,255,255),1,8,0);  
  

  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++) {
      gsl_vector_float_set(_vector,l,((uchar*)(_img->imageData + _img->widthStep*j))[i]);
      l++;
    }
  /*
  l=0;
  for (i=0; i <_width ; i++)
    for (j=0; j <_height; j++){
      gsl_vector_float_set(_vector,l,gsl_vector_float_get(_srcVec,l));
      l++;
    }
  */
  // cout << _frameX  << endl;
  return _vector;
}

gsl_vector_float* SnakeImage::data()
{
  gsl_vector_float* vec = gsl_vector_float_calloc(_vector->size);
  gsl_vector_float_memcpy(vec, _vector);
  return vec;
}


void SnakeImage::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}


// ----- methods for class `PCAFeature' -----
//
PCAFeature::PCAFeature(const VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, const char *filename, const char *filename_mean, int n, int k, const String& nm) :
  VectorFloatFeatureStream(k,nm), _src(src),
  _width(width), _height(height), _N(k), _M(_width * _height),
  _evec(gsl_matrix_calloc(_M, k)), _mean(gsl_vector_calloc(_M)),
  _srcVec(gsl_vector_float_calloc(_width * _height)),
  _endOfSamples(false)
{
  cout << "PCAFeature" << endl;
  FILE * f = fopen (filename, "r");
  if(f == NULL)
    throw jio_error("Could not open file %s.", filename);

  gsl_matrix* evc = gsl_matrix_calloc(_M, n);
  gsl_vector* v = gsl_vector_calloc(_M);

  gsl_matrix_fread (f,  evc);
  fclose (f);

  int c = 0;
  for (int i = n - k;i < n; i++) {
    gsl_matrix_get_col(v,evc,i);
    gsl_matrix_set_col(_evec,c,v);
    c++;
  }

  gsl_matrix_free(evc);
  gsl_vector_free(v);
  
  f = fopen (filename_mean, "r");
  if(f == NULL)
    throw jio_error("Could not open file %s.", filename_mean);

  gsl_vector_fread (f,  _mean);
  fclose (f);
}

PCAFeature::~PCAFeature()
{
  gsl_matrix_free(_evec);
  gsl_vector_free(_mean);
  gsl_vector_float_free(_srcVec);
}

void PCAFeature::_normalizeRows(){
  for (unsigned rowX = 0; rowX < _N; rowX++) {
    double mag2 = 0.0;
    for (unsigned colX = 0; colX < _M; colX++) {
      double val = gsl_matrix_get(_evec, rowX, colX);
      mag2 += val * val;
    }
    double magnitude = sqrt(mag2);
    if (magnitude != 1.0) {
      cout << "evec is not normalized - try normalization" <<  endl;
      for (unsigned colX = 0; colX < _M; colX++) {   // here should be a M or?
	double val = gsl_matrix_get(_evec, rowX, colX);
	gsl_matrix_set(_evec, rowX, colX, val / magnitude);
      }
    }
  }
}

const gsl_vector_float* PCAFeature::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1)); 
  _increment();  // _frameX++  
    
  gsl_vector* temp = gsl_vector_calloc(_width * _height);
  _floatToDouble(_srcVec, temp);

  gsl_vector_sub(temp, _mean);

  gsl_vector* outT = gsl_vector_calloc(_N);
  gsl_blas_dgemv(CblasTrans, 1.0, _evec, temp, 0.0, outT); 

  _doubleToFloat(outT, _vector);
    
  gsl_vector_free(temp);
  gsl_vector_free(outT);
  
  // cout << _frameX  << endl;
  return _vector;
}

void PCAFeature::reset()
{
  // cout << "Resetting in PCAFeature" << endl;

  _src->reset();
  VectorFloatFeatureStream::reset();
}

void PCAFeature::_floatToDouble(gsl_vector_float *src, gsl_vector *dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

  for(unsigned i = 0; i < src->size; i++)
    gsl_vector_set(dest, i, gsl_vector_float_get(src, i));
}

void PCAFeature::_doubleToFloat(gsl_vector* src, gsl_vector_float* dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

  for(unsigned i = 0; i < src->size; i++)
    gsl_vector_float_set(dest, i, gsl_vector_get(src, i));
}


// ----- methods for class `IPCAFeature' -----
//
IPCAFeature::IPCAFeature(const VectorFloatFeatureStreamPtr& src, unsigned width, unsigned height, const String& filename,  const String& filename_mean, int n, int k, const String& nm)
  : VectorFloatFeatureStream(width * height, nm), _src(src),
    _width(width), _height(height), _N(k), _M(_width * _height),
    _evec(gsl_matrix_calloc(_M, k)), _mean(gsl_vector_calloc(_M)),
    _srcVec(gsl_vector_float_calloc(_width * _height)),
    _endOfSamples(false)
{
  cout << "IPCAFeature" << endl;
  FILE* f = fopen (filename, "r");
  if(f == NULL)
    throw jio_error("Could not open file %s.", filename.c_str());

  gsl_matrix* evc = gsl_matrix_calloc(_M, n);
  gsl_vector* v = gsl_vector_calloc(_M);

  gsl_matrix_fread (f,  evc);
  fclose (f);

  int c = 0;
  for (int i = n - k; i < n; i++) {
    gsl_matrix_get_col(v, evc, i);
    gsl_matrix_set_col(_evec, c, v);
    c++;
  }

  gsl_matrix_free(evc);
  gsl_vector_free(v);
  
  f = fopen(filename_mean, "r");
  if(f == NULL)
    throw jio_error("Could not open file %s.", filename_mean.c_str());

  gsl_vector_fread(f,  _mean);
  fclose (f);

  _srcVec = gsl_vector_float_calloc(k);
}

//Destructor
IPCAFeature::~IPCAFeature()
{
  gsl_matrix_free(_evec);
  gsl_vector_free(_mean);
  gsl_vector_float_free(_srcVec);
}

void IPCAFeature::_normalizeRows()
{
  for (unsigned rowX = 0; rowX < _N; rowX++) {
    double mag2 = 0.0;
    for (unsigned colX = 0; colX < _M; colX++) {
      double val = gsl_matrix_get(_evec, rowX, colX);
      mag2 += val * val;
    }
    double magnitude = sqrt(mag2);
    if (magnitude != 1.0) {
      cout << "evec is not normalized - try normalization" <<  endl;
      cout << "Row " << rowX << " has length " << magnitude << "." << endl;
      for (unsigned colX = 0; colX < _M; colX++) {
	double val = gsl_matrix_get(_evec, rowX, colX);
	gsl_matrix_set(_evec, rowX, colX, val / magnitude);
      }
    }
  }
}

const gsl_vector_float* IPCAFeature::next(int frameX)
{
  if (_endOfSamples) {
    throw jiterator_error("end of samples!");
  }
  if (frameX == _frameX) return _vector;
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  gsl_vector_float_memcpy(_srcVec, _src->next(_frameX + 1)); 
  _increment();  // _frameX++  
    
  gsl_vector* temp = gsl_vector_calloc(_N);
  _floatToDouble(_srcVec,temp);

  gsl_vector* outT = gsl_vector_calloc(_M);
  gsl_blas_dgemv(CblasNoTrans, 1.0, _evec, temp, 0.0, outT); 

  gsl_vector_add(outT, _mean);

  _doubleToFloat(outT, _vector);
  
  gsl_vector_free(temp);
  gsl_vector_free(outT);

  // cout << _frameX  << endl;
  return _vector;
}

void IPCAFeature::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}

void IPCAFeature::_floatToDouble(gsl_vector_float* src, gsl_vector* dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

    for (unsigned i = 0; i < src->size; i++)
      gsl_vector_set(dest, i, gsl_vector_float_get(src, i));
}

void IPCAFeature::_doubleToFloat(gsl_vector* src, gsl_vector_float* dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

  for (unsigned i = 0; i < src->size; i++)
    gsl_vector_float_set(dest, i, gsl_vector_get(src,i));
}


// ----- methods for class `PCAEstimator' -----
//
PCAEstimator::PCAEstimator(unsigned n,unsigned m)
  :   _N(n), _M(m), _in_data(0),
      _sum(gsl_vector_calloc(_M)), _mean(gsl_vector_calloc(_M)),
      _data(gsl_matrix_calloc(_N, _M)), _mdata(gsl_matrix_calloc(_N, _M)),
      _mdatat(gsl_matrix_calloc(_M, _N)), _cov(gsl_matrix_calloc(_M, _M)),
      _eval(gsl_vector_calloc(_M)), _evec(gsl_matrix_calloc(_M, _M))
{
  cout << "PCAEstimator" << endl;
}

PCAEstimator::~PCAEstimator()
{
  gsl_matrix_free(_data);
  gsl_matrix_free(_mdata);
  gsl_matrix_free(_mdatat);
  gsl_vector_free(_sum);
  gsl_vector_free(_mean);
  gsl_matrix_free(_cov);

  gsl_vector_free(_eval);
  gsl_matrix_free(_evec);
}

void PCAEstimator::clearaccu()
{
  gsl_matrix_set_zero(_data);
  gsl_matrix_set_zero(_mdata);
  gsl_matrix_set_zero(_mdatat);
  gsl_vector_set_zero(_sum);
  gsl_vector_set_zero(_mean);
  gsl_matrix_set_zero(_cov);
  gsl_vector_set_zero(_eval);
  gsl_matrix_set_zero(_evec);
  _in_data = 0;
}

void PCAEstimator::accumulate(gsl_vector_float* V)
{
  gsl_vector* temp = gsl_vector_calloc(V->size);
  _floatToDouble(V, temp);
  gsl_matrix_set_row(_data, _in_data, temp);
  gsl_vector_add(_sum, temp);
  _in_data++;
}

void PCAEstimator::estimate()
{
  cout << "calculate: PCA" << endl;  
  cout << "calculate: mean = sum / N" << endl;
  //  mean  = gsl_vector_calloc(M);
  gsl_vector_add(_mean, _sum);
  //const float c = 1.0/(N);
  gsl_vector_scale(_mean, 1.0 / _N);
  
  cout << "calculate: mdata = data - mean" << endl;
  gsl_vector* h = gsl_vector_calloc(_M);
  for(unsigned i = 0; i < _N; i++) {
    gsl_matrix_get_row(h, _data,i);
    gsl_vector_sub(h, _mean);                //h = h - mean
    gsl_matrix_set_row(_mdata, i, h);
  }
  gsl_vector_free(h);
  
  cout << "calculate: mdatat = transpose(mdata) " << endl;
  gsl_matrix_transpose_memcpy(_mdatat, _mdata);
  
  cout << "calculate: cov = 1/N (mdatat * mdata) " << endl;
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _mdatat, _mdata, 0.0, _cov);
  gsl_matrix_scale(_cov, 1.0 / _M);              // cov = cov * 1/M

  cout << "calculate: [eval, evec] = eig(cov)" << endl;
  gsl_matrix * ccov;
  ccov = gsl_matrix_calloc(_M, _M);
  gsl_matrix_memcpy(ccov, _cov);
  gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(_M);
  gsl_eigen_symmv(ccov, _eval, _evec, w);
  gsl_eigen_symmv_free(w);  
  gsl_matrix_free(ccov);
  
  cout << "calculate: [eval, evec] = sort(eval, evec) " << endl;
  gsl_eigen_symmv_sort (_eval, _evec, GSL_EIGEN_SORT_ABS_ASC);
}

void PCAEstimator::save(const String& filename) // save the eigenvectors as doubles in a file
{
  cout << "size1: " << _evec->size1 << " size2: " << _evec->size2 << endl;
  FILE* f = fopen (filename, "wb");
  gsl_matrix_fwrite (f,  _evec);
  fclose(f);
}

void PCAEstimator::_floatToDouble(gsl_vector_float* src, gsl_vector* dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

  for(unsigned i = 0; i < src->size; i++)
    gsl_vector_set(dest,i,(double)gsl_vector_float_get(src,i));
}

void PCAEstimator::_doubleToFloat(gsl_vector* src, gsl_vector_float* dest)
{
  if (src->size != dest->size)
    throw jdimension_error("Source size (%d) is not equal to the destination size (%d).\n", src->size, dest->size);

  for(unsigned i = 0; i < src->size; i++)
    gsl_vector_float_set(dest, i, gsl_vector_get(src, i));
}


// ----- methods for class `PCAModEstimator' -----
//
PCAModEstimator::PCAModEstimator(unsigned n,unsigned m)
  : _N(n), _M(m), _indata(0),
    _data(gsl_matrix_calloc(_M, _N)),
    _mdata(gsl_matrix_calloc(_M, _N)),
    _sum(gsl_vector_calloc(_M)),
    _mean(gsl_vector_calloc(_M)),
    _cov(gsl_matrix_calloc(_N, _N)),
    _eval(gsl_vector_calloc(_N)),
    _evec(gsl_matrix_calloc(_N, _N)),
    _evecc(gsl_matrix_calloc(_M, _N)),
    _evecct(gsl_matrix_calloc(_M, _N)),
    _eigface(gsl_matrix_calloc(_M, _N))

{
  cout << "PCAModEstimator" << endl;
}

PCAModEstimator::~PCAModEstimator()
{
  gsl_matrix_free(_data);
  gsl_matrix_free(_mdata);
  gsl_vector_free(_sum);
  gsl_vector_free(_mean);
  gsl_matrix_free(_cov);

  gsl_vector_free(_eval);
  gsl_matrix_free(_evec);
  gsl_matrix_free(_evecc);
  gsl_matrix_free(_evecct);
  gsl_matrix_free(_eigface);
}

void PCAModEstimator::clearaccu()
{
  gsl_matrix_set_zero(_data);
  gsl_matrix_set_zero(_mdata);
  gsl_vector_set_zero(_sum);
  gsl_vector_set_zero(_mean);
  gsl_matrix_set_zero(_cov);
  gsl_vector_set_zero(_eval);
  gsl_matrix_set_zero(_evec);
  gsl_matrix_set_zero(_evecc);   // - mod. no evecc in normal PCA
  gsl_matrix_set_zero(_evecct);
  gsl_matrix_set_zero(_eigface); // - mod. no evecc in normal PCA
  _indata = 0;
}

void PCAModEstimator::accumulate(gsl_vector_float* V)
{
  gsl_vector*  temp = gsl_vector_calloc(V->size);
  _floatToDouble(V, temp);
  gsl_matrix_set_col(_data, _indata, temp);
  gsl_vector_add(_sum, temp);
  _indata++;
}

void PCAModEstimator::_normalizeRows()
{
  for (unsigned rowX = 0; rowX < _N; rowX++) {
    double mag2 = 0.0;
    for (unsigned colX = 0; colX < _M; colX++) {
      double val = gsl_matrix_get(_evecct, rowX, colX);
      mag2 += val * val;
    }

    double magnitude = sqrt(mag2);
    //    cout << "Row " << rowX << " has length " << magnitude << "." << endl;
    for (unsigned colX = 0; colX < _M; colX++) {   // here should be a M or?
      double val = gsl_matrix_get(_evecct, rowX, colX);
      gsl_matrix_set(_evecct, rowX, colX, val / magnitude);
    }
  }
}

void PCAModEstimator::_normalizeColumn()
{
  for (unsigned colX = 0; colX < _N; colX++) {
    //   if (gsl_vector_get(_eval,colX)>0.0000001){
      double mag2 = 0.0;
      for (unsigned rowX = 0; rowX < _M; rowX++) {
	double val = gsl_matrix_get(_evecc, rowX, colX);
	mag2 += val * val;
      }
      double magnitude = sqrt(mag2);
      //    cout << "Row " << rowX << " has length " << magnitude << "." << endl;
      if (magnitude > 0.0000001){
	for (unsigned rowX = 0; rowX < _M; rowX++) {   // here should be a M or?
	  double val = gsl_matrix_get(_evecc, rowX, colX);
	  gsl_matrix_set(_evecc, rowX, colX, val / magnitude);
	}
      } else {
	for (unsigned rowX = 0; rowX < _M; rowX++) {
	  gsl_matrix_set(_evecc, rowX, colX,0.0);
	}
      }
    //} else {
    //      for (unsigned rowX = 0; rowX < _M; rowX++) {
    //	gsl_matrix_set(_evecc, rowX, colX,0.0);
    //      }      
    //    } 
  }
}

void PCAModEstimator::estimate()
{
  cout << "calculate: modify PCA" << endl;  
  cout << "calculate: mean = sum / N" << endl;
  gsl_vector_add(_mean, _sum);
  gsl_vector_scale(_mean, 1.0/_N);

  cout << "calculate: mdata = data - mean" << endl;
  gsl_vector *h = gsl_vector_calloc(_M);
  for(unsigned i = 0; i < _N; i++){
    gsl_matrix_get_col(h, _data, i);
    gsl_vector_sub(h, _mean);                                                  //h = h - mean
    gsl_matrix_set_col(_mdata, i, h);
  }
  gsl_vector_free(h);
  
  cout << "calculate: cov = 1/N (mdatat * mdata) " << endl;                    //Trick to reduce the dimension of the cov-matrix
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, _mdata, _mdata, 0.0, _cov); 
  gsl_matrix_scale(_cov, 1.0/_N);                                              // - mod. normal PCA would be ...1.0/M

  cout << "calculate: [eval, evec] = eig(cov)" << endl;
  gsl_matrix * ccov = gsl_matrix_calloc(_N, _N);                               // - mod. normal PCA would be M x M
  gsl_matrix_memcpy(ccov, _cov);
  gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(_N);                   // - mod. normal PCA would be ...alloc(M)
  gsl_eigen_symmv(ccov, _eval, _evec, w);
  gsl_eigen_symmv_free (w);  
  gsl_matrix_free(ccov);
  
  cout << "calculate: [eval, evec] = sort(eval, evec) " << endl;
  gsl_eigen_symmv_sort (_eval, _evec, GSL_EIGEN_SORT_VAL_ASC);

  cout << "calculate: evecc = mdata * evect " << endl;                         // - mod. this is not a part of the normal PCA
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, _mdata, _evec, 0.0, _evecc); // _evec(N,N) _mdata(N,M) _evecc(N,M)
  gsl_matrix_memcpy(_eigface,_evecc);

  cout << "calculate: normalization of evecc " << endl;
  _normalizeColumn();
}

void PCAModEstimator::save(const char *filename1, const char *filename2)       // save the eigenvectors as doubles in a file
{
  cout << "Save evecc: " << "size1: " << _evecc->size1 << " size2: " << _evecc->size2 << endl;
  FILE * f = fopen (filename1, "wb");
  gsl_matrix_fwrite (f,  _evecc);                                              // - mod. the normal PCA would save the evec
  fclose (f);
  cout << "Save mean" << "size: " << _mean->size << endl;
  f = fopen (filename2, "wb");
  gsl_vector_fwrite (f,  _mean);                                               // - mod. the normal PCA would save the evec
  fclose (f);
}

void PCAModEstimator::_floatToDouble(gsl_vector_float* src, gsl_vector* dest)
{
  if (src->size == dest->size){
    int i;
    for(i=0;i<src->size;i++){
      gsl_vector_set(dest,i,(double)gsl_vector_float_get(src,i));
    }
  }
}

void PCAModEstimator::_doubleToFloat(gsl_vector* src, gsl_vector_float* dest)
{
  if (src->size == dest->size){
    int i;
    for(i=0;i<src->size;i++){
      gsl_vector_float_set(dest,i,(float)gsl_vector_get(src,i));
    }
  }
}

#endif

#endif
