import numpy as np
import sys
import os


def nbclassify(datapath):
    test_filepaths = []
    for root, d_names, f_names in os.walk(datapath):
        for f in f_names:
            if ".txt" in f:
                test_filepaths.append(os.path.join(root, f))
    test_filepaths.sort()

    # Fetch learned model
    f = open('nbmodel.txt', 'r', encoding='latin1')
    p_ham = float(f.readline())  # P(ham)
    p_spam = float(f.readline())  # P(spam)

    p_token_ham = {}  # P(token|ham)
    p_token_spam = {}  # P(token|spam)
    for line in f:
        tag, key, val = line.split()
        if tag == "tokenham":
            p_token_ham[key] = float(val)
        elif tag == "tokenspam":
            p_token_spam[key] = float(val)

    # Evaluate test data
    fOutput = open('nboutput.txt', 'w', encoding='latin1')
    for fpath in test_filepaths:
        fHandle = open(fpath, "r", encoding="latin1")
        lines = fHandle.readlines()
        ham_score = np.log(p_ham)
        spam_score = np.log(p_spam)
        for line in lines:
            words = line.split()
            for token in words:
                # if token seen in HAM, then update ham score
                if token in p_token_ham.keys():
                    ham_score += np.log(p_token_ham[token])

                # if token seen in SPAM, then update spam score
                if token in p_token_spam.keys():
                    spam_score += np.log(p_token_spam[token])

                # if token not seen during training, ignore it

        if ham_score>=spam_score:
            fOutput.write("ham %s \n" % fpath)
        else:
            fOutput.write("spam %s\n" % fpath)

    fOutput.close()

    return 0


if __name__=="__main__":
    path = sys.argv[1]
    nbclassify(path)
