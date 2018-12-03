//  Module:  btk20.jexception
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs

%{
#include "common/jexception.h"
#include "common/jpython_error.h"

using namespace std;
%}

%init %{
#ifdef THREADING
  PyEval_InitThreads();
#endif
%}

%exception {
  //jpython_error *pe = NULL;
  try {
    $action
  } catch(j_error& e) {
    switch (e.getCode()) {
    case JITERATOR:
      PyErr_SetString(PyExc_StopIteration, "");
      return NULL;
    case JIO:
      PyErr_SetString(PyExc_IOError, e.what());
      return NULL;
    case JPYTHON:
      //pe = static_cast<jpython_error*>(&e);
      //PyErr_Restore(pe->getType(), pe->getValue(), pe->getTrace());
      return NULL;
    default:
      break;
    }
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_Exception, "unknown error");
    return NULL;
  };
}
