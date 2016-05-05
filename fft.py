# -*- coding: utf-8 -*-
"""
Created on Mon May 02 16:02:36 2016

@author: abc
"""

import os
import glob
import sys
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from utils import GENRE_DIR, CHART_DIR, GENRE_LIST

def write_fft(fft_features, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)
    print "Written ", data_fn


def create_fft(fn):
    """
        Creates the MFCC features. 
    """    
    sample_rate, X = scipy.io.wavfile.read(fn)
    
    X = X[:,0]
    print X.shape
    X[X==0]=1
    fft_features = abs(scipy.fft(X))
    print fft_features.shape,"sahpe"
    write_fft(fft_features, fn)


def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        for fn in file_list:
            fft_features = np.load(fn)
            X.append(fft_features[:1000])
            y.append(label)
    return np.array(X), np.array(y)


def create_ceps_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    sample_rate, X = scipy.io.wavfile.read(fn)
    
    X[X==0]=1
    np.nan_to_num(X)
    ceps, mspec, spec = mfcc(X)
    base_fn, ext = os.path.splitext(fn)
    print base_fn
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written ", data_fn
    return data_fn


def read_ceps_test(test_file):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    ceps = np.load(test_file)
    num_ceps = len(ceps)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    print "Working with these genres --> ", traverse
    print "Starting ceps generation"     
    for subdir, dirs, files in os.walk(GENRE_DIR):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                
                tmp = subdir[subdir.rfind('/',0)+1:]
                print tmp
                if tmp in traverse:
                    
                    create_fft(path)
                    
    stop = timeit.default_timer()
    print "Total ceps generation and feature writing time (s) = ", (stop - start) 
