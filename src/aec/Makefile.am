AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_aec.la
libbtk_aec_la_SOURCES = \
	aec.cc
libbtk_aec_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la
libbtk_aec_la_LDFLAGS = $(GSL_LIBS)

btk_aec_includedir = $(pkgincludedir)/aec
btk_aec_include_HEADERS = \
	aec.h

pkgpython_PYTHON = aec.py
pkgpyexec_LTLIBRARIES = _aec.la

_aec_la_SOURCES = aec_wrap.cxx
_aec_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_aec_la_LIBADD = libbtk_aec.la
_aec_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = aec_wrap.cxx aec.py
EXTRA_DIST = aec.i

SWIG_FILES = aec_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include

include $(SWIG_RULES)
