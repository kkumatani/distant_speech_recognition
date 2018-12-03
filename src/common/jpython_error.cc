#include "common/jpython_error.h"

jpython_error::jpython_error() {
  _what = "###Python Error###";
  code = JPYTHON;
};
