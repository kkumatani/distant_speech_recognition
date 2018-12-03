lib_LTLIBRARIES = libbtk_utils.la
libbtk_utils_la_SOURCES = utils.cc
libbtk_utils_la_LIBADD = $(top_srcdir)/common/libbtk_common.la

btkutils_includedir = $(pkgincludedir)/utils
btkutils_include_HEADERS = utils.h

pkgpython_PYTHON = utils.py
pkgpyexec_LTLIBRARIES = _utils.la

_utils_la_SOURCES = utils_wrap.cxx
_utils_la_CPPFLAGS = $(PYTHON_CPPFLAGS)
_utils_la_LIBADD = libbtk_utils.la
_utils_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = utils_wrap.cxx utils.py
EXTRA_DIST = utils.i

SWIG_FILES = utils_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
