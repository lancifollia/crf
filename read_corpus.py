#!/usr/bin/env python
# -*- coding: utf-8 -*-


from utils.exceptions import FileFormatError


def read_conll_corpus(filename):
    """
    Read a corpus file with a format used in CoNLL.
    """
    data = list()
    data_string_list = list(open(filename))

    element_size = 0
    X = list()
    Y = list()
    for data_string in data_string_list:
        words = data_string.strip().split()
        if len(words) is 0:
            data.append((X, Y))
            X = list()
            Y = list()
        else:
            if element_size is 0:
                element_size = len(words)
            elif element_size is not len(words):
                raise FileFormatError
            X.append(words[:-1])
            Y.append(words[-1])
    if len(X) > 0:
        data.append((X, Y))

    return data
