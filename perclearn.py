import os
import sys
import time
import json
import random


def read_input(path):
    filepaths = []
    for root, dirs, filenames in os.walk(path):
        for file in filenames:
            if ".txt" in file:
                filepaths.append(os.path.join(root, file))

    # sort in alphabetical order
    filepaths.sort()
    return filepaths


def build_weights(all_files):
    """
    find all words in all documents, and store in dictionary

    :param all_files:
    :return weights: dictionary with Key=word and value=0
    """
    weights_dict = {}
    for fpath in all_files:
        fHandle = open(fpath, "r", encoding="latin1")
        lines = fHandle.readlines()
        for line in lines:
            words = line.split()
            for token in words:
                if token not in weights_dict:
                    weights_dict[token] = 0

    return weights_dict


def get_feature_vectors(file_paths):
    """
    Store word counds in feature vector for each file, along with true label
    """
    xy_train = {}
    for file in file_paths:
        xy_train[file] = [{}, 0]
        fHandle = open(file, "r", encoding="latin1")
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

        xy_train[file][0] = wordCount
        if 'spam' in file:
            xy_train[file][1] = 1
        else:
            xy_train[file][1] = -1

    return xy_train


def train(xy_train, w, maxIter=100):
    """
    Train perceptron and write weights and bias into file.
    """
    u = w.copy()
    beta = 0
    count = 1
    bias = 0
    keyList = list(xy_train.keys())
    for ii in range(maxIter):
        random.shuffle(keyList)
        # iterStart = time.time()
        for file in keyList:
            # get feature vector corresponding ot this email
            wordCount = xy_train[file][0]

            # calculate activation for this email
            alph = 0  # initialize for THIS EMAIL
            for token in wordCount:
                if token in w:
                    alph += w[token] * wordCount[token]
            alph += bias

            # If misclassified, update weights and bias
            if xy_train[file][1]*alph<=0:
                bias += xy_train[file][1]
                for token in wordCount:
                    if token in w:
                        w[token] += wordCount[token]*xy_train[file][1]
                        u[token] += count*wordCount[token] * xy_train[file][1]
            count += 1

        # print("iter: %d, time: %f seconds" % (ii, time.time() - iterStart))

    # Update averaged weights
    for key in u:
        u[key] = w[key] - 1/count * u[key]
    beta = bias - 1/count * beta

    # Write averaged model to file
    fmodel = open('percmodel.txt', 'w', encoding='latin1')
    fmodel.write(str(beta) + "\n")
    json.dump(u, fmodel)
    fmodel.close()

    # Write UN-averaged model to file
    fmodel = open('percmodel_noAvg.txt', 'w', encoding='latin1')
    fmodel.write(str(bias) + "\n")
    json.dump(w, fmodel)
    fmodel.close()


if __name__=="__main__":
    maxIter = 100
    train_path = sys.argv[1]

    # 1. Get all files paths for training
    file_paths = read_input(train_path)

    # 2. build dictionary of weights
    # startTime = time.time()
    weights = build_weights(file_paths)
    # print("Built weights dictionary: %d seconds" % (time.time() - startTime))

    # 3. Get feature vectors
    # startTime = time.time()
    xy_train = get_feature_vectors(file_paths)
    # print("Built feature vectors: %d seconds" % (time.time() - startTime))

    # 3. Train perceptron and write model to file
    # startTime = time.time()
    train(xy_train, weights, maxIter)
    # print("Training done: %f" % ((time.time() - startTime) / 60))
