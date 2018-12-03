AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_modulated.la
libbtk_modulated_la_SOURCES = \
	modulated.cc pc_lattice.c prototype_design.cc
libbtk_modulated_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la
libbtk_modulated_la_LDFLAGS = $(GSL_LIBS)

btk_modulated_includedir = $(pkgincludedir)/modulated
btk_modulated_include_HEADERS = \
	modulated.h pc_lattice.h prototype_design.h

pkgpython_PYTHON = modulated.py
pkgpyexec_LTLIBRARIES = _modulated.la

_modulated_la_SOURCES = modulated_wrap.cxx
_modulated_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_modulated_la_LIBADD = libbtk_modulated.la
_modulated_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = modulated_wrap.cxx modulated.py
EXTRA_DIST = modulated.i

SWIG_FILES = modulated_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
