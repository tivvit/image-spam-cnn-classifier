import sys
from load import load
from collections import Counter
import numpy as np


def main():
    pth = sys.argv[1]
    y = []
    c = 0
    for i in load(pth):
        y.append(i[0])
        c += 1
        if c == 1000:
            break
    samples = len(y)
    reactions = sum([1 for i in y for j in i])
    multiple = sum([1 for i in y for j in i if len(j) > 1])
    only_consistent = [j[0] for i in y for j in i if len(j) == 1]
    num_consistent = len(only_consistent)
    print("#Sample images: {}".format(samples))
    for k in range(10):
        print("#Sample images >{} reaction: {}".format(
            k, len([i for i in y if len(i) > k])))
    print("#Reactions: {}".format(reactions))
    print("Avg reactions per image: {:.2f}".format(reactions / float(samples)))
    print("Multiple reactions from one user: {}".format(multiple))
    print("Multiple reactions from one user ratio: {:.2f}%".format(
        multiple / float(reactions) * 100))
    print("-" * 10)
    print("Only valid reactions from now on")
    print("-" * 10)
    print("#Valid reactions: {}".format(num_consistent))
    non_spam = len([i for i in only_consistent if i ==
                    "USER_MESSAGE_UNMARK_SPAM"])
    spam = len([i for i in only_consistent if i ==
                "USER_MESSAGE_MARK_SPAM"])
    non_ad = len([i for i in only_consistent if i ==
                  "USER_MESSAGE_MARK_NONAD"])
    ad = len([i for i in only_consistent if i == "USER_MESSAGE_MARK_AD"])
    print("#spam: {:.2f}%".format(spam / float(num_consistent) * 100))
    print("#non-spam: {:.2f}%".format(non_spam / float(num_consistent) * 100))
    print("#non-ad: {:.2f}%".format(non_ad / float(num_consistent) * 100))
    print("#ad: {:.2f}%".format(ad / float(num_consistent) * 100))
    consist = []
    for i in y:
        t = [j[0] for j in i if len(j) == 1]
        if t:
            consist.append(t)
    avg_consistency = []
    for i in consist:
        avg_consistency.append(Counter(i).most_common(1)[0][1] / float(len(i)))
    print("avg consistency: {:.2f}% +-{:.2f}".format(
        np.mean(avg_consistency) * 100, np.std(avg_consistency) * 100))
    avg_consistency = []
    for i in consist:
        if len(i) == 1:
            continue
        avg_consistency.append(Counter(i).most_common(1)[0][1] / float(len(i)))
    print("avg consistency >1: {:.2f}% +-{:.2f}".format(
        np.mean(avg_consistency) * 100, np.std(avg_consistency) * 100))


if __name__ == '__main__':
    main()
