import matplotlib.pyplot as plt
import numpy as np
import random

from auxiliar import load_txt_file

PATH = './frgcv1/'
SETNAME = 'set_4_label.txt'

def split_known_unknown_sets(complete_tuple_list, set_size=0.5):
    label_set = set()
    for (path, label) in complete_tuple_list:
        label_set.add(label)

    known_set = set(random.sample(label_set, int(set_size * len(label_set))))
    unknown_set = label_set - known_set
    
    known_tuple = [(path, label) for (path, label) in complete_tuple_list if label in known_set]
    unknown_tuple = [(path, label) for (path, label) in complete_tuple_list if label in unknown_set]
    
    return known_tuple, unknown_tuple


def split_train_test_sets(complete_tuple_list, train_set_size=0.5):
    from sklearn.model_selection import train_test_split
    
    labels = []
    paths = []
    random_state = np.random.RandomState(0)
    for (path, label) in complete_tuple_list:
        labels.append(label)
        paths.append(label)

    random_gen = np.random.RandomState(0)
    path_train, path_test, label_train, label_test = train_test_split(paths, labels, train_size=train_set_size, random_state=random_gen)

    train_set = zip(path_train, label_train)
    test_set = zip(path_test, label_test)

    return train_set, test_set

def generate_cmc_curve(cmc_scores):
    x_axis = range(len(cmc_scores))
    y_axis = cmc_scores

    plt.clf()
    plt.plot(x_axis, y_axis, color='blue', linestyle='-')
    plt.xlim([0, len(cmc_scores)])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Rank')
    plt.ylabel('Accuracy Rate')
    plt.title('Cumulative Matching Characteristic')
    # plt.show()


def main():
    # set_list = load_txt_file(PATH + SETNAME)
    # knownT, unknownT = split_known_unknown_sets(set_list)
    # gallery, probe = split_train_test_sets(knownT)

    # print(len(gallery), len(probe))

    cmc_score = [0.1, 0.2, 0.32, 0.45, 0.58, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    generate_cmc_curve(cmc_score)

if __name__ == "__main__":
    main()