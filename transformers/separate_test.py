import sys
import gzip
import random
import ujson as json

from train.load import load


def main():
    limit = None
    part = ""
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        part = "_part"
    pth = sys.argv[1]
    prob = .15
    random.seed(42)
    c = 0
    with gzip.open("/features/image_spam_public" +
                           part + "_train.json.gz", "wb") as tr:
        with gzip.open("/features/image_spam_public" +
                               part + "_test.json.gz", "wb") as tst:
            for i in load(pth):
                if random.random() > prob:
                    tr.write(json.dumps(i).encode() + b'\n')
                else:
                    tst.write(json.dumps(i).encode() + b'\n')
                c += 1
                if limit and c > limit:
                    break


if __name__ == '__main__':
    main()
