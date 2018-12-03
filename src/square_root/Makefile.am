AM_CPPFLAGS = $(CPPFLAGS)

lib_LTLIBRARIES = libbtk_squareroot.la
libbtk_squareroot_la_SOURCES = square_root.cc
libbtk_squareroot_la_LDFLAGS = $(GSL_LIBS)

btksquareroot_includedir = $(pkgincludedir)/square_root
btksquareroot_include_HEADERS = square_root.h

pkgpython_PYTHON = square_root.py kalman.py
pkgpyexec_LTLIBRARIES = _square_root.la

_square_root_la_SOURCES = square_root_wrap.cxx
_square_root_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_square_root_la_LIBADD = \
$(top_srcdir)/common/libbtk_common.la
_square_root_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = square_root_wrap.cxx square_root.py
EXTRA_DIST = square_root.i

SWIG_FILES = square_root_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include

include $(SWIG_RULES)
