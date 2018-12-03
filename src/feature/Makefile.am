AM_CPPFLAGS = \
    $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib

lib_LTLIBRARIES = libbtk_feature.la
libbtk_feature_la_SOURCES = feature.cc lpc.cc spectralestimator.cc
libbtk_feature_la_LIBADD = \
    $(top_srcdir)/matrix/libbtk_matrix.la \
    $(top_srcdir)/stream/libbtk_stream.la \
    $(top_srcdir)/gnulib/libgnu.la
libbtk_feature_la_LDFLAGS = \
    $(GSL_LIBS) \
    $(SNDFILE_LIBS) \
    $(SAMPLERATE_LIBS) \
    $(JACK_LIBS) \
    $(AVCODEC_LIBS) \
    $(AVFORMAT_LIBS) \
    $(OPENCV_LIBS) -lpthread

btkfeature_includedir = $(pkgincludedir)/feature
btkfeature_include_HEADERS = feature.h lpc.h spectralestimator.h feature.i

pkgpython_PYTHON = feature.py
pkgpyexec_LTLIBRARIES = _feature.la

_feature_la_SOURCES = feature_wrap.cxx
_feature_la_LIBADD = libbtk_feature.la
_feature_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_feature_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS) $(PYTHON_LSPEC)

CLEANFILES = feature_wrap.cxx feature.py

SWIG_FILES = feature_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
