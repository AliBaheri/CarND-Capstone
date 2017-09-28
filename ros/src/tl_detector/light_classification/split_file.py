import os
import sys
from functools import partial
import argparse

chunksize = 40960000

def splitfile(filename, directory, chunksize=chunksize):
    chunknum = 0
    filename = os.path.abspath(filename)
    with open(filename, 'rb') as infile:
        #for chunk in iter(partial(infile.read, chunksize), ''):
        for chunk in iter(lambda: infile.read(chunksize), b''):
            ofilename = os.path.join(directory, ('chunk%04d'%(chunknum)))
            outfile = open(ofilename, 'wb')
            outfile.write(chunk)
            outfile.close()
            chunknum += 1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="path to file which requires splitting")
ap.add_argument("-o", "--output", required=True, help="sub-directory for file chunks (must exist)")
args = vars(ap.parse_args())

splitfile(args["file"], args["output"])
print("File is split into output folder")