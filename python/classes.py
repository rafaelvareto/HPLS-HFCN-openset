from collections import defaultdict
from numpy import array
import random

from pls_classifier import PLSClassifier


class Classes:
    def __init__(self):
        self.disarrays = []
        self.hashes = []
        self.labels = []
        self.map = {}
        self.multimap = defaultdict(list)
        self.stamps = []
        self.pls = PLSClassifier()

    def add_element(self, feature, label):
        print(label[:3])
        self.labels.append(label[:3])
        self.map[label] = feature
        self.multimap[label[:3]].append(feature)
        self.stamps.append(label)

    def shuffle_labels(self, variety):
        unique = list(set(self.labels))
        size_u = len(unique) / 2
        pos_neg = [1 if item < size_u else -1 for item in range(0, len(unique))]
        for index in range(0, variety):
            temp_tuples = []
            to_shuffle = [item for item in unique]
            random.shuffle(to_shuffle)
            for sh,pn in zip(to_shuffle,pos_neg):
                temp_tuples.append((sh, pn))
            self.disarrays.append(temp_tuples)

    def learn_models(self):
        out_data = []
        for outer in self.disarrays:
            temp_featrs = []
            temp_labels = []
            temp_pos_ng = []
            for un, pn in outer:
                for item in self.multimap[un]:
                    temp_featrs.append(item)
                    temp_labels.append(un)
                    temp_pos_ng.append(pn)
            out_data.append([[temp_featrs], [temp_labels], [temp_pos_ng]])
            model = self.pls.fit(array(temp_featrs), array(temp_pos_ng))
            self.hashes.append(model)
