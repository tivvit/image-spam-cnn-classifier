import sys

import numpy as np
import ujson as json
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

VERSION = "0.0.2"

data_pth_ham = sys.argv[1]
data_pth_spam = sys.argv[2]
name = "{}_{}".format(sys.argv[3], VERSION)


def load(pth):
    for l in open(pth):
        yield json.loads(l)


def load_split_data():
    x = []
    y = []
    for i in load(data_pth_spam):
        x.append(i[1])
        y.append([0, 1])
    for i in load(data_pth_ham):
        x.append(i[1])
        y.append([1, 0])
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.25,
                                                        random_state=42)
    return X_test, X_train, y_test, y_train


def main():
    classes = ['ham', 'spam']
    X_test, X_train, y_test, y_train = load_split_data()

    print("features: {}".format(len(X_train[0])))
    print("train samples: {}".format(len(X_train)))
    print("test samples: {}".format(len(X_test)))

    ham_samples = len([i for i in y_train if i[0] == 1])
    spam_samples = len([i for i in y_train if i[1] == 1])
    print("Ham: {}".format(ham_samples))
    print("Spam: {}".format(spam_samples))

    ham_weight = spam_samples / float(ham_samples)
    print("Ham weight: {}".format(ham_weight))

    model = Sequential([
        Dense(2048, input_dim=2048, activation='relu'),
        Dense(2048, input_dim=2048, activation='relu'),
        Dense(len(classes), activation='softmax'),
    ])

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=5000,
        class_weight={
            0: ham_weight,
            1: 1,
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
