/**
 * @file file_stream.h
 * @brief Common operations for files.
 * @author Kenichi Kumatani
 */

#ifndef FILE_STREAM_H
#define FILE_STREAM_H

#include <stdio.h>
#include "common/mach_ind_io.h"
#include "common/mlist.h"
#include "common/refcount.h"

class FileHandler {
 public:
  FileHandler(const String &filename, const String &mode);
  ~FileHandler();
  int read_int();
  String read_string();
  void write_int(int val);
  void write_string(String val);
#ifdef ENABLE_LEGACY_BTK_API
  int readInt(){ return read_int(); }
  String readString(){ return read_string(); }
  void writeInt(int val){ write_int(val); }
  void writeString(String val){ write_string(val); }
#endif

 private:
  FILE* fp_;
  String filename_;
  char buf_[FILENAME_MAX];
};

typedef refcount_ptr<FileHandler>  FileHandlerPtr;

#endif /* #ifndef FILE_STREAM_H */

