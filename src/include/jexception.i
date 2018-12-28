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

%include "exception.i"

%exception {
  //jpython_error *pe = NULL;
  try {
    $action
  }
  catch(jiterator_error& e) {
    //PyErr_SetNone(PyExc_StopIteration);
    PyErr_SetString(PyExc_StopIteration, "stop iteration");
    return NULL;
  }
  catch(jallocation_error& e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
    return NULL;
  }
  catch(jarithmetic_error& e) {
    PyErr_SetString(PyExc_ArithmeticError, e.what());
    return NULL;
  }
  catch(jnumeric_error& e) {
    PyErr_SetString(PyExc_FloatingPointError, e.what());
    return NULL;
  }
  catch(jindex_error& e) {
    PyErr_SetString(PyExc_IndexError, e.what());
    return NULL;
  }
  catch(jio_error& e) {
    PyErr_SetString(PyExc_IOError, e.what());
    return NULL;
  }
  catch(jkey_error& e) {
    PyErr_SetString(PyExc_KeyError, e.what());
    return NULL;
  }
  catch(jparameter_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return NULL;
  }
  catch(jparse_error& e) {
    PyErr_SetString(PyExc_SyntaxError, e.what());
    return NULL;
  }
  catch(jtype_error& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }
  catch(jconsistency_error& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
  catch(jinitialization_error& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
  catch(jdimension_error& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
    }
  catch(j_error& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
  catch (...) {
    PyErr_SetString(PyExc_Exception, "Unknown error");
    return NULL;
  };
}
