#!/usr/bin/env python

import argparse
from crf import LinearChainCRF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file for testing input")
    parser.add_argument("modelfile", help="the model file name.")

    args = parser.parse_args()

    crf = LinearChainCRF()
    crf.load(args.modelfile)
    crf.test(args.datafile)
