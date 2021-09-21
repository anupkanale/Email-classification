import sys
import os
import json
import random


def percclassify(datapath):
    test_filepaths = []
    for root, d_names, f_names in os.walk(datapath):
        for f in f_names:
            if ".txt" in f:
                test_filepaths.append(os.path.join(root, f))
    test_filepaths.sort()

    # Fetch learned model
    fmodel = open('percmodel.txt', 'r', encoding='latin1')
    bias = float(fmodel.readline())
    weights = json.load(fmodel)

    # Classify test data
    fout = open('percoutput.txt', 'w', encoding='latin1')
    for fpath in test_filepaths:
        fHandle = open(fpath, "r", encoding="latin1")
        lines = fHandle.readlines()

        # create feature vector for this file
        wordCount = {}
        for line in lines:
            words = line.split()
            for token in words:
                if token in wordCount:
                    wordCount[token] += 1
                else:
                    wordCount[token] = 1

        # calculate activation for this file
        alph = bias
        for line in lines:
            words = line.split()
            for token in words:
                if token in weights:
                    alph += weights[token] * wordCount[token]

        # classify and write to file
        if alph>0:
            fout.write("spam %s\n" % fpath)
        else:
            fout.write("ham %s\n" % fpath)

    fout.close()


if __name__=="__main__":
    data_path = sys.argv[1]
    percclassify(data_path)
