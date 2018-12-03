AM_CPPFLAGS = $(PYTHON_CPPFLAGS)
lib_LTLIBRARIES = libbtk_convolution.la
libbtk_convolution_la_SOURCES = \
	convolution.cc 
libbtk_convolution_la_LDFLAGS = $(GSL_LIBS) $(top_srcdir)/stream/libbtk_stream.la

btk_convolution_includedir = $(pkgincludedir)/convolution
btk_convolution_include_HEADERS = \
	convolution.h

pkgpython_PYTHON = convolution.py
pkgpyexec_LTLIBRARIES = _convolution.la

_convolution_la_SOURCES = convolution_wrap.cxx
_convolution_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_convolution_la_LIBADD = libbtk_convolution.la
_convolution_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = convolution_wrap.cxx convolution.py
EXTRA_DIST = convolution.i

SWIG_FILES = convolution_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
