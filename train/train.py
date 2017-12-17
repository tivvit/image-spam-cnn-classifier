import sys

import numpy as np
import ujson as json
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from load import load_data as load_email_cz_ds
from load import load as load_email_cz_ds_file
from load import load_data_whole as load_email_cz_ds_whole

VERSION = "0.0.2"

data_pth_ham = sys.argv[1]
data_pth_spam = sys.argv[2]
name = "{}_{}".format(sys.argv[3], VERSION)
our_ds = False
in_mem = True
if len(sys.argv) >= 5:
    our_ds = True
    min_reaction = int(sys.argv[4])
    min_ratio = float(sys.argv[5])
    in_mem = bool(sys.argv[6])


def load(pth):
    for l in open(pth):
        yield json.loads(l)


def load_data():
    x = []
    y = []
    for i in load(data_pth_spam):
        x.append(i[1])
        y.append([0, 1])
    for i in load(data_pth_ham):
        x.append(i[1])
        y.append([1, 0])
    return x, y


def split_data(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.25,
                                                        random_state=42)
    return X_test, X_train, y_test, y_train


def main():
    epochs = 80
    if our_ds:
        classes = ["ham", "spam", "ad"]
        batch_size = 5000
        test_cnt = 0
        if not in_mem:
            print("Loading validation")
            for _ in load_email_cz_ds_file(data_pth_spam):
                test_cnt += 1
            print("Done validation load")
            y_train = []
            features_size = None
            train_samples = 0
            print("analyzing dataset")
            for i in load_email_cz_ds_whole(data_pth_ham,
                                            min_ratio=min_ratio,
                                            min_reactions=min_reaction):
                y_train.append(i[1])
                train_samples += 1
                if not features_size:
                    features_size = len(i[0])
            print("Done loading analysis")
        else:
            print("Loading training")
            X_test, X_train, y_test, y_train = [], [], [], []
            for i in load_email_cz_ds_whole(data_pth_ham,
                                            min_ratio=min_ratio,
                                            min_reactions=min_reaction):
                y_train.append(i[1])
                X_train.append(i[0])
            print("Done training load")
            print("Loading validation")
            for i in load_email_cz_ds_whole(data_pth_spam,
                                            min_ratio=min_ratio,
                                            min_reactions=min_reaction):
                y_test.append(i[1])
                X_test.append(i[0])
            print("Done validation load")
            features_size = len(X_train[0])
            train_samples = len(X_train)
            test_cnt = len(X_test)
    else:
        classes = ["ham", "spam"]
        x, y = load_data()
        X_test, X_train, y_test, y_train = split_data(x, y)
        features_size = len(X_train[0])
        train_samples = len(X_train)
        test_cnt = len(X_test)

    print("features: {}".format(features_size))
    print("train samples: {}".format(train_samples))
    print("test samples: {}".format(test_cnt))

    ham_samples = len([i for i in y_train if i[0] == 1])
    spam_samples = len([i for i in y_train if i[1] == 1])
    ad_samples = len([i for i in y_train if i[2] == 1])
    print("Ham: {}".format(ham_samples))
    print("Spam: {}".format(spam_samples))
    print("Ad: {}".format(ad_samples))

    ham_weight = spam_samples / float(ham_samples)
    print("Ham weight: {}".format(ham_weight))
    ad_weight = spam_samples / float(ad_samples)
    print("Ad weight: {}".format(ad_weight))

    model = Sequential([
        Dense(2048, input_dim=2048, activation='relu'),
        Dense(2048, input_dim=2048, activation='relu'),
        Dense(len(classes), activation='softmax'),
    ])

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if not in_mem:
        model.fit_generator(
            load_email_cz_ds(data_pth_ham,
                             min_reactions=min_reaction,
                             min_ratio=min_ratio,
                             batch_size=batch_size,
                             # validation_data=load_email_cz_ds_whole(
                             #     data_pth_spam,
                             #     min_ratio=min_ratio,
                             #     min_reactions=min_reaction)
                             ),
            steps_per_epoch=train_samples / batch_size,
            epochs=epochs,
            class_weight={
                0: ham_weight,
                1: 1,
                2: ad_weight,
            },
        )
    else:
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight={
                0: ham_weight,
                1: 1,
                2: ad_weight,
            }
        )

    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_pred = model.predict(X_test)
    report = classification_report(
        np.array([np.argmax(r) if r.shape else r for r in np.array(y_test)]),
        np.array([np.argmax(r) if r.shape else r for r in np.array(y_pred)]),
        target_names=classes)
    print(report)

    model.save("{}.{}".format(name, "h5"))

    with open("{}_eval.txt".format(name), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(str(scores))
        f.write(str(model.metrics_names))
        f.write("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        f.write(report)


if __name__ == "__main__":
    main()
