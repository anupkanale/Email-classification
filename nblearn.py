import os
import sys


def read_input(path):
    fpaths_ham = []
    fpaths_spam = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            if ".txt" in f and "ham" in f:
                fpaths_ham.append(os.path.join(root, f))
            elif ".txt" in f and "spam" in f:
                fpaths_spam.append(os.path.join(root, f))

    return fpaths_ham, fpaths_spam


def get_prior_probs(fpaths_ham, fpaths_spam):
    """
        Calculate prior probabilities by counting.

        For verification, numbers from train folder:
        1--> 3672 ham, 1500 spam
        2--> 4361 ham, 1496 spam
        3--> 1500 ham, 4500 spam
        -------------------------------
        total--> 9533 ham, 7496 spam
        -------------------------------

        :return: P(ham), P(spam)
    """
    ham_count, spam_count = len(fpaths_ham), len(fpaths_spam)

    # print(ham_count, spam_count)
    p_ham = ham_count / (ham_count + spam_count)
    p_spam = spam_count / (ham_count + spam_count)
    return p_ham, p_spam


def get_conditional_probs(ham_files, spam_files, fmodel):
    """
        Given a token, what's the probability of it being spam and not?

        :return: P(token|ham), P(token|ham)
    """
    # make list of unique words in all messages, and count
    vocab = {}  # for fast lookup
    vocab_size = 0

    # Count instances of each token in HAM mail
    tokenCounts_ham = {}
    N_tokens_ham = 0
    for fpath in ham_files:
        fHandle = open(fpath, "r", encoding="latin1")
        lines = fHandle.readlines()
        for line in lines:
            words = line.split()
            N_tokens_ham += len(words)
            for token in words:
                if token not in vocab:
                    vocab[token] = 0
                    vocab_size += 1

                if token in tokenCounts_ham.keys():
                    tokenCounts_ham[token] += 1
                else:
                    tokenCounts_ham[token] = 1

    # Count instances of each token in SPAM mail
    tokenCounts_spam = {}
    N_tokens_spam = 0
    for fpath in spam_files:
        fHandle = open(fpath, "r", encoding="latin1")
        lines = fHandle.readlines()
        for line in lines:
            words = line.split()
            N_tokens_spam += len(words)
            for token in words:
                if token not in vocab:
                    vocab[token] = 0
                    vocab_size += 1

                if token in tokenCounts_spam.keys():
                    tokenCounts_spam[token] += 1
                else:
                    tokenCounts_spam[token] = 1

    # find probability of seeing each token given ham or spam
    # Add-one smoothing
    p_token_ham = {}
    p_token_spam = {}
    for key in vocab:
        if key in tokenCounts_ham:
            p_token_ham[key] = (tokenCounts_ham[key]+1)/(N_tokens_ham + vocab_size)
        else:
            p_token_ham[key] = 1 / (N_tokens_ham + vocab_size)

        if key in tokenCounts_spam:
            p_token_spam[key] = (tokenCounts_spam[key]+1)/(N_tokens_spam + vocab_size)
        else:
            p_token_spam[key] = 1 / (N_tokens_spam + vocab_size)

    # print(vocab_size)
    # print(vocab)

    # print(N_tokens_ham, N_tokens_spam)
    # print(tokenCounts_ham["following"])
    # print(p_token_ham["following"])

    # Write to file
    for key in p_token_ham:
        fmodel.write("tokenham" + " " + key + " " + str(p_token_ham[key]) + "\n")

    for key in p_token_spam:
        fmodel.write("tokenspam" + " " + key + " " + str(p_token_spam[key]) + "\n")
    fmodel.close()

    return p_token_ham, p_token_spam


if __name__== "__main__":
    train_path = sys.argv[1]
    file_paths_ham, file_paths_spam = read_input(train_path)
    P_ham, P_spam = get_prior_probs(file_paths_ham, file_paths_spam)

    # print(P_ham, P_spam)
    # print(file_paths_ham[0:1])
    # print(file_paths_spam[0:1])
    fmodel = open('nbmodel.txt', 'w', encoding='latin1')
    fmodel.write(str(P_ham) + "\n")
    fmodel.write(str(P_spam) + "\n")
    P_token_ham, P_token_spam = get_conditional_probs(file_paths_ham, file_paths_spam, fmodel)