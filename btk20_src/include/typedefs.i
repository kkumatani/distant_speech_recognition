//  Module:  bt20.typedefs
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs

//  we use PyString_{As,From}StringAndSize so that we can use null-embedded
//  bytes in strings since both python strings and C++ strings handle them ok

//  Old SWIG versions don't know Py_ssize_t. So f.f. added this.
//  This is just a guess of when it was introduced. If you find out the exact
//  inflationary version number, change it.
#if SWIG_VERSION < 0x010325
#define Py_ssize_t int
#endif

// need to include "common/refcount.h"
%include <std_string.i>
%include <typemaps.i>
using namespace std;

// convert from python to C++ string
%typemap(in) String {
        char * temps; Py_ssize_t templ;
        if (PyString_AsStringAndSize($input, &temps, &templ)) return NULL;
        $1 = $1_ltype (temps);
 }

%typemap(in) const String& (String tempstr) {
        char * temps; Py_ssize_t templ;
        if (PyString_AsStringAndSize($input, &temps, &templ)) return NULL;
        tempstr = String(temps);
        $1 = &tempstr;
 }

// this is for setting string structure members:
%typemap(in) String* *INPUT($*1_ltype tempstr) {
        char * temps; Py_ssize_t templ;
        if (PyString_AsStringAndSize($input, &temps, &templ)) return NULL;
        tempstr = $*1_ltype(temps);
        $1 = &tempstr;
 }

// convert from C++ to python string
%typemap(out) const string&, const String& {
  $result = PyString_FromStringAndSize($1->data(), $1->length());
 }

// this is for getting string structure members:
%typemap(out) string*, String*, StringPtr, stringPtr {
  $result = PyString_FromStringAndSize($1->data(), $1->length());
 }

%typemap(varin) String {
        char *temps; Py_ssize_t templ;
        if (PyString_AsStringAndSize($input, &temps, &templ)) return NULL;
        $1 = $1_ltype(temps);
 }

%typemap(varin) string {
        char *temps; Py_ssize_t templ;
        if (PyString_AsStringAndSize($input, &temps, &templ)) return NULL;
        $1 = $1_ltype(temps, templ);
 }

%typemap(varout) string, String {
  $result = PyString_FromStringAndSize($1.data(), $1.length());
 }


//  Typemaps for list<string>
%typemap(in) const list<string>& (list<string> tempList) {

  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }

  for (int i=0; i<PySequence_Length($input); i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyString_Check(o)) {
      string* s = new string(PyString_AsString(o));
      tempList.push_back(*s);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be strings");
      return NULL;
    }
  }
        $1 = &tempList;
};


// Typemaps for FILE*

%typemap(in) FILE *
{
	if(PyFile_Check($input)) {
		$1 = PyFile_AsFile($input);
	} else {
		PyErr_SetString(PyExc_ValueError,"Object must be a FILE!");
		return NULL;
	}
}

