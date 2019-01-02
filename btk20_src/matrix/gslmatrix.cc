#include <math.h>
#include "gslmatrix.h"
#include "common/mlist.h"
#include "common/mach_ind_io.h"

gsl_matrix_float* gsl_matrix_float_resize(gsl_matrix_float* m, size_t size1, size_t size2)
{
  assert(m && size1 > 0 && size2 > 0);
  if (m->size1 == size1 && m->size2 == size2) return m;

  if (m->size1 < size1)
    throw jdimension_error("Cannot resize from %d to %d", m->size1, size1);
  if (m->size2 < size2)
    throw jdimension_error("Cannot resize from %d to %d", m->size2, size2);

  gsl_matrix_float* new_m = gsl_matrix_float_calloc(size1, size2);

  for (unsigned rowX = 0; rowX < size1; rowX++)
    for (unsigned colX = 0; colX < size2; colX++)
      gsl_matrix_float_set(new_m, rowX, colX, gsl_matrix_float_get(m, rowX, colX));

  gsl_matrix_float_free(m);

  return new_m;
}

static gsl_matrix_float* gsl_matrix_float_load_(gsl_matrix_float* m, const char* fileName)
{
  FILE* mf = btk_fopen(fileName, "r");
  if (gsl_matrix_float_fread(mf, m) < 0) {
    fclose(mf);
    throw jio_error("Could not read matrix from %s", fileName);
  }

  btk_fclose(fileName, mf);
  return m;
}

static gsl_matrix_float* gsl_matrix_float_load_old_(gsl_matrix_float* m, const char* filename)
{
  int pos, left, total;
  int size1, size2;
  long total_size;
  char fmagic[5];

  FILE* mf = btk_fopen(filename, "r");

  fmagic[4] = '\0';
  if ((fread(fmagic, sizeof(char), 4, mf) != 4)||(strncmp("FMAT", fmagic, 4))) {
    fclose(mf);
    throw jio_error("Couldn't find magic number in file\n");
  }

  size1 = read_int(mf);
  size2 = read_int(mf);
  read_float(mf); // A->count, what is it in gsl???

  pos = ftell(mf);
  fseek(mf, 0, SEEK_END);
  left = ftell(mf) - pos;
  fseek(mf, pos, SEEK_SET);
  if (size2 && (size1 < 0)) size1 = left/(sizeof(float)*size2); /* number of rows wasn't set */
  total = size1 * size2;
  total_size = total*sizeof(float);
  if (!left) {
    fclose(mf);
    throw jio_error("File empty, matrix unchanged!\n");
  } else if (left < 0 || left != total_size) {
    fclose(mf);
    throw jio_error ("Number of bytes in file = don't match matrix dimension:\n");
  }

  if (m == NULL) {
    m = gsl_matrix_float_calloc(size1, size2);
  } else if ((m = gsl_matrix_float_resize(m, size1, size2)) == NULL) {
    fclose(mf);
    throw jdimension_error("Could not resize matrix to %d x %d.", size1, size2);
  }

  if (read_floats(mf, m->data, total) != total) {
    fclose(mf);
    throw jio_error("Could not read %d floats from %s.", total, filename);
  }

  btk_fclose(filename, mf);
  return m;
}

gsl_matrix_float* gsl_matrix_float_load(gsl_matrix_float* m, const char* filename, bool old)
{
  if (old)
    return gsl_matrix_float_load_old_(m, filename);

  return gsl_matrix_float_load_(m, filename);
}


/**
 * @brief Cosine Transform
 * @details
 *  type 0  is the original cosine transform (IDFT for a symmetric power
 *          spectrum) taking N/2+1 power coefficients from a N-point DFT.
 *    (as in NIST's PLP calculation)
 *  type 1  is used to transform log mel-frequency coefficients.
 *          (as in mfcc_lib.c by Aki Ohshima aki@speech1.cs.cmu.edu)
 */
void gsl_matrix_float_set_cosine( gsl_matrix_float* m, size_t i, size_t j, int type)
{
  int  k,l;

  assert(m && i>=0 && j>=0);
  m = gsl_matrix_float_resize(m, i, j);

  if (type == 0) {
    for (k=0; k<m->size1; k++) {
      double fac = k * M_PI / (double)(m->size2 - 1);
      float* ptr = gsl_matrix_float_ptr(m, k, 0);
      if (j) *ptr++ = 1.0;
      for (l=1; l<(m->size2-1); l++)
        *ptr++ = 2.0 * cos(fac*l);
      if (j) *ptr = cos(k * M_PI);
    }
  } else if (type == 1) {
    for (k=0; k<m->size1; k++) {
      double fac = k * M_PI / (double)m->size2;
      float* ptr = gsl_matrix_float_ptr(m, k, 0);
      for (l=0; l<m->size2; l++) *ptr++ = cos(fac*(l+0.5));
    }
  } else
    throw j_error("cosine: type must be 0 or 1\n");
}


static int readMagic4(FILE* fp, const char* magic)
{
  static char *fmagic = NULL;
  long        current = ftell(fp);
  size_t      t = 4;

  fseek(fp,current,SEEK_SET);

  if (fmagic == NULL) fmagic = (char*)malloc(5 * sizeof(char));
  fmagic[4] = '\0';
  t = fread (fmagic, sizeof(char), 4, fp);

  if (  t != 4 || strncmp(magic,fmagic,4)) {
    fseek(fp,current,SEEK_SET);
    throw jio_error("Couldn't find magic number in file\n");
  }

  return 1;
}

static gsl_vector_float* _gsl_vector_float_load(gsl_vector_float* A, const char *filename)
{
  FILE* mf = fopen(filename,"r");
  if (gsl_vector_float_fread(mf, A) < 0) {
    fclose(mf);
    throw jio_error("Could not read matrix from %s", filename);
  }

  fclose(mf);
  return A;
}

static long bytesLeft(FILE* fp)
{
   fpos_t position, end;
   
   if (fgetpos(fp, &position) != 0) return -1;
   fseek(fp, (long)0, SEEK_END);
   if (fgetpos(fp, &end) != 0) return -2;
   if (fsetpos(fp, &position) != 0) return -3;

#if defined __GLIBC__ && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 2
   return (end.__pos - position.__pos);
#else
   return (end - position);
#endif
}

static gsl_vector_float* gsl_vector_float_resize(gsl_vector_float* m, size_t size)
{
  if (m->size == size) return m;

  if (m->size < size)
    throw jdimension_error("Cannot resize from %d to %d", m->size, size);

  gsl_vector_float* new_m = gsl_vector_float_calloc(size);

  for (unsigned colX = 0; colX < size; colX++)
    gsl_vector_float_set(new_m, colX, gsl_vector_float_get(m, colX));

  gsl_vector_float_free(m);

  return new_m;
}

static gsl_vector_float* _gsl_vector_float_load_old(gsl_vector_float* A, const char *filename)
{
  FILE    *fp;
  int     n,total;
  long    left;
  long    total_size;
  gsl_vector_float* ret = A;
  assert(A);

  if ((fp = btk_fopen(filename, "r")) == NULL) return NULL;
  if (readMagic4(fp,"FVEC")) {

    /* -- read dimensions -- */
    n = read_int(fp);
    /* count= */
    read_float(fp);

    /* -- check with file length -- */
    left = bytesLeft(fp);
    total = n;
    total_size = total*sizeof(float);

    if (left < 0L || left != total_size)
      throw j_error("Number of bytes in file = don't match vector dimension:\n");

    /* -- load data -- */
    else {
      A = gsl_vector_float_resize(A, n);
      if (read_floats(fp,A->data,total) != total) ret = NULL;
    }
  }
  else ret = NULL;

  btk_fclose(filename, fp);
  return ret;
}

gsl_vector_float* gsl_vector_float_load(gsl_vector_float* m, const char* filename, bool old)
{
  if (old)
    return _gsl_vector_float_load_old(m, filename);

  return _gsl_vector_float_load(m, filename);
}
