import cv2 as cv
import numpy as np


def load_images(path, image_list, display=False):
    for name in image_list:
        image = cv.imread(path + '/' + name, cv.IMREAD_COLOR)
        if display:
            cv.imshow('img', image)
            cv.waitKey(20)
    if display:
        cv.destroyAllWindows()


def load_txt_file(file_name):
    this_file = open(file_name, 'r')
    this_list = []
    for line in this_file:
        line = line.rstrip()
        components = line.split()
        this_list.append(components)
    return this_list


def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def generate_pos_neg_dict(labels):
    to_shuffle = [item for item in labels]
    np.random.shuffle(to_shuffle)
    neg_set = map(lambda neg: (neg, -1), to_shuffle[0:(len(labels) / 2)])
    pos_set = map(lambda pos: (pos, +1), to_shuffle[(len(labels) / 2):len(labels)])
    full_set = neg_set + pos_set
    full_dict = dict((key, val) for key, val in full_set)
    return full_dict
