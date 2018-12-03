AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_common.la
libbtk_common_la_SOURCES = \
    common.cc error.c jexception.cc \
    mach_ind_io.cc memory_manager.cc refcount.cc \
    jpython_error.cc
libbtk_common_la_CPPFLAGS = $(PYTHON_CPPFLAGS)

btkcommon_includedir = $(pkgincludedir)/common
btkcommon_include_HEADERS = \
    common.h error.h jexception.h \
    mach_ind_io.h refcount.h memory_manager.h \
    mlist.h jpython_error.h

pkgpython_PYTHON = common.py
pkgpyexec_LTLIBRARIES = _common.la

_common_la_SOURCES = common_wrap.cxx
_common_la_CPPFLAGS = $(PYTHON_CPPFLAGS)
_common_la_LIBADD = libbtk_common.la
_common_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = common_wrap.cxx common.py
EXTRA_DIST = common.i

SWIG_FILES = common_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
