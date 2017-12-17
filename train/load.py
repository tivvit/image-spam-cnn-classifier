import ujson as json
from collections import Counter
import numpy as np
import gzip


def load(data_pth, infinite=False):
    if infinite:
        f = gzip.open(data_pth, "rb")
        while True:
            for i in f:
                yield json.loads(i)
            f = gzip.open(data_pth, "rb")
    else:
        for i in gzip.open(data_pth, "rb"):
            yield json.loads(i)


def map_class(d):
    y = None
    if d == "USER_MESSAGE_UNMARK_SPAM":
        y = [1, 0, 0]  # ham
    if d == "USER_MESSAGE_MARK_NONAD":
        y = [1, 0, 0]  # ham
    if d == "USER_MESSAGE_MARK_SPAM":
        y = [0, 1, 0]  # spam
    if d == "USER_MESSAGE_MARK_AD":
        y = [0, 0, 1]  # ad
    return y


def main_reaction(d):
    d = [i for i in d if len(i) == 1]
    if not d:
        return
    c = Counter([j for i in d for j in i])
    return (map_class(c.most_common(1)[0][0]),
            c.most_common(1)[0][1] / float(len(d)),
            len(d))


def prepare_data(d, min_ractions=1):
    reactions = d[0]
    features = d[1]
    if len(reactions) <= min_ractions:
        return
    mr = main_reaction(reactions)
    if not mr:
        return
    return mr, features


def load_data(pth, min_reactions=4, min_ratio=.6, batch_size=5000):
    x = [(None, None)] * batch_size
    c = 0
    f = load(pth, infinite=True)
    while True:
        l = next(f)
        i = prepare_data(l, min_ractions=min_reactions)
        if not i:
            continue
        if i[0][1] >= min_ratio:
            x[c] = (i[1], i[0][0])
            c += 1
        if c == batch_size:
            xx = np.array(x)
            np.random.shuffle(xx)
            yield np.array([j[0] for j in xx]), np.array([j[1] for j in xx])
            c = 0


def load_data_whole(pth, min_reactions=4, min_ratio=.6):
    for l in load(pth):
        i = prepare_data(l, min_ractions=min_reactions)
        if not i:
            continue
        if i[0][1] >= min_ratio:
            yield (i[1], i[0][0])
