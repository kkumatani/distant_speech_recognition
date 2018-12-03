FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib
AM_CPPFLAGS = \
	$(CPPFLAGS) $(FEATURE_FLAGS) 
lib_LTLIBRARIES = libbtk_localization.la 
libbtk_localization_la_SOURCES = localization.cc MCCLocalizer.cc
libbtk_localization_la_LIBADD = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la $(top_srcdir)/squareRoot/libbtk_squareroot.la
libbtk_localization_la_LDFLAGS = $(GSL_LIBS)

libbtklocalization_la_includedir = $(pkgincludedir)/localization
libbtklocalization_la_include_HEADERS = localization.h MCCLocalizer.h

pkgpython_PYTHON = localization.py
pkgpyexec_LTLIBRARIES = _localization.la

_localization_la_SOURCES = localization_wrap.cxx
_localization_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_localization_la_LIBADD = libbtk_localization.la
_localization_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = localization_wrap.cxx localization.py
EXTRA_DIST = localization.i

SWIG_FILES = localization_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(FEATURE_FLAGS)

include $(SWIG_RULES)
