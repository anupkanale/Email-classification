import sys


def percevaluate(outputfile):
    ham_count_correct = 0
    ham_count_wrong = 0
    spam_count_correct = 0
    spam_count_wrong = 0
    n_ham = 0
    n_spam = 0

    f = open(outputfile, 'r', encoding='latin1')
    lines = f.readlines()
    n = len(lines)
    for ii in range(n):
        line = lines[ii].split()
        y_pred = line[0]
        if y_pred == 'ham':
            if 'ham' in line[1]:
                n_ham += 1
                ham_count_correct += 1
            elif 'spam' in line[1]:
                n_spam += 1
                spam_count_wrong += 1
        if y_pred == 'spam':
            if 'ham' in line[1]:
                n_ham += 1
                ham_count_wrong += 1
            elif 'spam' in line[1]:
                n_spam += 1
                spam_count_correct += 1

    # Compute precision and recall
    precision_ham = ham_count_correct / (ham_count_correct + ham_count_wrong)
    recall_ham = ham_count_correct / n_ham
    f1_ham = 2 * precision_ham * recall_ham / (precision_ham + recall_ham)

    precision_spam = spam_count_correct / (spam_count_correct + spam_count_wrong)
    recall_spam = spam_count_correct / n_spam
    f1_spam = 2 * precision_spam * recall_spam / (precision_spam + recall_spam)

    print("Label: precision | recall | F1 score")
    print("Ham: %f %f %f" % (precision_ham, recall_ham, f1_ham))
    print("Spam: %f %f %f" % (precision_spam, recall_spam, f1_spam))
    print(ham_count_wrong, spam_count_wrong)

    f.close()
    return 0


if __name__=="__main__":
    op_file = sys.argv[1]
    percevaluate(op_file)
