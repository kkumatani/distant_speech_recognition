AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_postfilter.la
libbtk_postfilter_la_SOURCES = binauralprocessing.cc postfilter.cc spectralsubtraction.cc $(top_srcdir)/beamformer/beamformer.cc
libbtk_postfilter_la_LIBADD  = \
$(top_srcdir)/stream/libbtk_stream.la \
$(top_srcdir)/matrix/libbtk_matrix.la \
$(top_srcdir)/modulated/libbtk_modulated.la 
libbtk_postfilter_la_LDFLAGS = $(GSL_LIBS) 

btk_postfilter_includedir = $(pkgincludedir)/postfilter
btk_postfilter_include_HEADERS = postfilter.h spectralsubtraction.h binauralprocessing.h

pkgpython_PYTHON = postfilter.py
pkgpyexec_LTLIBRARIES = _postfilter.la

_postfilter_la_SOURCES = postfilter_wrap.cxx
_postfilter_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_postfilter_la_LIBADD = libbtk_postfilter.la  
_postfilter_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = postfilter_wrap.cxx postfilter.py
EXTRA_DIST = postfilter.i

SWIG_FILES = postfilter_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
