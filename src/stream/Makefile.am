AM_CPPFLAGS = $(GSL_CFLAGS) $(CPPFLAGS)

lib_LTLIBRARIES = libbtk_stream.la
libbtk_stream_la_SOURCES = stream.cc file_stream.cc
libbtk_stream_la_LIBADD = $(top_srcdir)/common/libbtk_common.la $(GSL_LIBS)

btkstream_includedir = $(pkgincludedir)/stream
btkstream_include_HEADERS = stream.h pyStream.h file_stream.h stream.i

pkgpython_PYTHON = stream.py
pkgpyexec_LTLIBRARIES = _stream.la

_stream_la_SOURCES = stream_wrap.cxx 
_stream_la_LIBADD = libbtk_stream.la
_stream_la_CPPFLAGS =  $(PYTHON_CPPFLAGS) $(GSL_CFLAGS)
_stream_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = stream_wrap.cxx stream.py

SWIG_FILES = stream_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(GSL_CFLAGS) 

include $(SWIG_RULES)
