FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib
AM_CPPFLAGS = $(CPPFLAGS) $(FEATURE_FLAGS)
lib_LTLIBRARIES = libbtk_tde.la
libbtk_tde_la_SOURCES = tde.cc 
libbtk_tde_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la $(top_srcdir)/modulated/libbtk_modulated.la
libbtk_tde_la_LDFLAGS = $(GSL_LIBS)

btk_tde_includedir = $(pkgincludedir)/tde
btk_tde_include_HEADERS = tde.h

pkgpython_PYTHON = tde.py tdoa.py
pkgpyexec_LTLIBRARIES = _tde.la

_tde_la_SOURCES = tde_wrap.cxx
_tde_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS) 
_tde_la_LIBADD = libbtk_tde.la
_tde_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = tde_wrap.cxx tde.py
EXTRA_DIST = tde.i

SWIG_FILES = tde_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(FEATURE_FLAGS)

include $(SWIG_RULES)
