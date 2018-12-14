import os.path
import pickle
import string
import getopt, sys
import re

inputFile  = ''
outputFile = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
except getopt.GetoptError:
    # print help information and exit:
    sys.exit(2)
 
for o, a in opts:
    if o in ("-h", "--help"):
        sys.exit()
    elif o in ("-i", "--input"):
        inputFile  = a
    elif o in ("-o", "--output"):
        outputFile = a

if outputFile == '':
    outputFile  = os.path.dirname( inputFile )
    
baseName = os.path.splitext( os.path.basename( inputFile ) )
baseName = baseName[0]
baseName = baseName.replace('-',' ').split()

analysisFileName  = '%s/h' %(outputFile)
synthesisFileName = '%s/g' %(outputFile)

for entry in baseName:
    par = entry.replace('=',' ').split()
    print par
    if par[0] == 'M':
        M = int(par[1])
        analysisFileName =  analysisFileName +  '-M=%d' %(M)
        synthesisFileName = synthesisFileName + '-M=%d' %(M) 
    elif par[0] == 'm':
        m = int(par[1])
        analysisFileName =  analysisFileName +  '-m=%d' %(m)
        synthesisFileName = synthesisFileName + '-m=%d' %(m) 
    elif par[0] == 'r':
        r = int(par[1])
        analysisFileName =  analysisFileName +  '-r=%d' %(r)
        synthesisFileName = synthesisFileName + '-r=%d' %(r) 
    elif par[0] == 'v':
        v = float(par[1])
        analysisFileName =  analysisFileName +  '-v=%0.6f' %(v)
        synthesisFileName = synthesisFileName + '-v=%0.6f' %(v) 
    elif par[0] == 'w':
        w = float(par[1])
        analysisFileName =  analysisFileName +  '-w=%0.6f' %(w)
        synthesisFileName = synthesisFileName + '-w=%0.6f' %(w)

analysisFileName =  analysisFileName +  '.txt'
synthesisFileName = synthesisFileName + '.txt'
print '%s to \n%s and\n%s \n' %(inputFile,analysisFileName,synthesisFileName)

fp = open(inputFile, 'r')

count = 0
for line in fp:
    fbv = string.split(string.strip(line))
    if count == 0:
        h = []
        for entries in fbv:
            h.append( float(entries) )
        print 'analysis: %d' %(len(h))
    else:
        g = []
        for entries in fbv:
            g.append( float(entries) )
        print 'synthesis: %d' %(len(g))
    count += 1

fp.close()

fp = open(analysisFileName, 'w')
pickle.dump(h, fp, True)
fp.close()

fp = open(synthesisFileName, 'w')
pickle.dump(g, fp, True)
fp.close()
