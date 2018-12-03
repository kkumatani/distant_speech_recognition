import struct
import Numeric
#from matrix import FMatrix

numArrayType = type(Numeric.identity(1))
intSize = struct.calcsize(">i")
floatSize = struct.calcsize(">f")

#def _fmatrixWrite_fm(fmatrix, file):
#    m = fmatrix.m
#    n = fmatrix.n
#
#    file.write(struct.pack(">ii", m, n))
#
#    for i in range(m):
#        for j in range(n):
#            file.write(struct.pack(">f", fmatrix.getItem(i,j)))

def _fmatrixWrite_na(matrix, file):
    if len(matrix.shape) != 2:
        raise Exception("Matrix expected!")
    (m,n) = matrix.shape
    print matrix.shape

    file.write(struct.pack(">4s", "FMAT"))
    file.write(struct.pack(">iif", m, n, 0.0))

    for i in range(m):
        for j in range(n):
            file.write(struct.pack(">f", matrix[i,j]))
    
def fmatrixWrite(matrix, file):
    """
    Writes a FMatrix, NumPy Array or Python list to disc using the
    binary fmatrix format of JANUS
    """
    if (isinstance(matrix, numArrayType)):
        _fmatrixWrite_na(matrix, file)
 #   elif (isinstance(matrix, FMatrix)):
 #       _fmatrixWrite_fm(matrix, file)
    elif (isinstance(matrix, list)):
        _fmatrixWrite_na(Numeric.array(matrix), file)
    
def fmatrixRead(file):
    """
    Reads a JANUS fmatrix file and returns it as a NumPy array
    """
    s,m,n,f = struct.unpack(">4siif", file.read(struct.calcsize(">4siif")))
    matrix = Numeric.zeros((m,n), Numeric.Float)
    for i in range(m):
        for j in range(n):
            matrix[i,j] = struct.unpack(">f", file.read(floatSize))[0]
    return matrix

if __name__ == "__main__":
#    fm = FMatrix(13,13)
#    fm.initDiag(10.5)
#    fmatrixWrite(fm, open("janus2.bin", "w"))

    na = Numeric.identity(13)*10.5
    fmatrixWrite(na, open("janus3.bin", "w"))

    print fmatrixRead(open("janus2.bin", "r"))
