AM_CPPFLAGS = $(PYTHON_CPPFLAGS)
lib_LTLIBRARIES = libbtk_lms.la
libbtk_lms_la_SOURCES = lms.cc
libbtk_lms_la_LIBADD = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/matrix/libbtk_matrix.la $(top_srcdir)/convolution/libbtk_convolution.la
libbtk_lms_la_LDFLAGS = $(GSL_LIBS)

btk_lms_includedir = $(pkgincludedir)/lms
btk_lms_include_HEADERS = lms.h

pkgpython_PYTHON = lms.py
pkgpyexec_LTLIBRARIES = _lms.la

_lms_la_SOURCES = lms_wrap.cxx
_lms_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_lms_la_LIBADD = libbtk_lms.la
_lms_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = lms_wrap.cxx lms.py
EXTRA_DIST = lms.i

SWIG_FILES = lms_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
