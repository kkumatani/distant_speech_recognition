AM_CPPFLAGS = $(CPPFLAGS)

lib_LTLIBRARIES = libbtk_matrix.la
libbtk_matrix_la_SOURCES = gslmatrix.cc blas1_c.cc linpack_c.cc
libbtk_matrix_la_LIBADD = $(top_srcdir)/common/libbtk_common.la $(GSL_LIBS)

btkmatrix_includedir = $(pkgincludedir)/matrix
btkmatrix_include_HEADERS = \
    gslmatrix.h blas1_c.h linpack_c.h

noinst_HEADERS =

pkgpython_PYTHON = matrix.py
pkgpyexec_LTLIBRARIES = _matrix.la

_matrix_la_SOURCES = matrix_wrap.cxx
_matrix_la_LIBADD = libbtk_matrix.la
_matrix_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_matrix_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = matrix_wrap.cxx matrix.py
EXTRA_DIST = matrix.i

SWIG_FILES = matrix_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
