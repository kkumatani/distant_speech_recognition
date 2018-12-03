AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_beamformer.la
libbtk_beamformer_la_SOURCES = beamformer.cc taylorseries.cc modal_beamformer.cc tracker.cc
libbtk_beamformer_la_LIBADD  = \
$(top_srcdir)/stream/libbtk_stream.la \
$(top_srcdir)/matrix/libbtk_matrix.la \
$(top_srcdir)/feature/libbtk_feature.la \
$(top_srcdir)/modulated/libbtk_modulated.la \
$(top_srcdir)/postfilter/libbtk_postfilter.la
libbtk_beamformer_la_LDFLAGS = $(GSL_LIBS)

btk_beamformer_includedir = $(pkgincludedir)/beamformer
btk_beamformer_include_HEADERS = beamformer.h spectralinfoarray.h taylorseries.h modal_beamformer.h tracker.h

pkgpython_PYTHON = beamformer.py
pkgpyexec_LTLIBRARIES = _beamformer.la

_beamformer_la_SOURCES = beamformer_wrap.cxx
_beamformer_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_beamformer_la_LIBADD = libbtk_beamformer.la
_beamformer_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = beamformer_wrap.cxx beamformer.py
EXTRA_DIST = beamformer.i

SWIG_FILES = beamformer_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
