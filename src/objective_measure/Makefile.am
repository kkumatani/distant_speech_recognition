FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib
AM_CPPFLAGS = $(CPPFLAGS) $(FEATURE_FLAGS)
lib_LTLIBRARIES = libbtk_objective_measure.la
libbtk_objective_measure_la_SOURCES = objective_measure.cc 
libbtk_objective_measure_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la $(top_srcdir)/modulated/libbtk_modulated.la
libbtk_objective_measure_la_LDFLAGS = $(GSL_LIBS)

btk_objective_measure_includedir = $(pkgincludedir)/objective_measure
btk_objective_measure_include_HEADERS = objective_measure.h

pkgpython_PYTHON = objective_measure.py
pkgpyexec_LTLIBRARIES = _objective_measure.la

_objective_measure_la_SOURCES = objective_measure_wrap.cxx
_objective_measure_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS) 
_objective_measure_la_LIBADD = libbtk_objective_measure.la
_objective_measure_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = objective_measure_wrap.cxx objective_measure.py
EXTRA_DIST = objective_measure.i

SWIG_FILES = objective_measure_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(FEATURE_FLAGS)

include $(SWIG_RULES)
