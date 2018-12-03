#include "file_stream.h"

FileHandler::FileHandler(const String &filename, const String &mode):
  fp_(NULL), filename_(filename)
{
  fp_ = btk_fopen(filename.chars(), mode.chars());
  if( NULL == fp_ ){
    throw jio_error("could not open %s\n", filename.chars());
  }
}

FileHandler::~FileHandler()
{
  btk_fclose(filename_.chars(), fp_);
}

int FileHandler::read_int()
{
  return ::read_int(fp_);
}

String FileHandler::read_string()
{
  ::read_string(fp_, buf_);

  return String(buf_);
}

void FileHandler::write_int(int val)
{
  ::write_int(fp_, val);
}

void FileHandler::write_string(String val)
{
  ::write_string(fp_, val.chars());
}

