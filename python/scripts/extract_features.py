import sys
sys.path.append('../')

import argparse
import cv2 as cv
import numpy as np
import pickle

from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets
from descriptor import Descriptor
from vggface import VGGFace

parser = argparse.ArgumentParser(description='Extracting Features for libSVM')
parser.add_argument('-p', '--path', help='Path do dataset', required=False, default='../frgcv1/')
parser.add_argument('-f', '--file', help='Input file name', required=False, default='train_2_small.txt')
parser.add_argument('-d', '--desc', help='Descriptor [hog/df]', required=False, default='hog')
parser.add_argument('-r', '--rept', help='Number of executions', required=False, default=5)
parser.add_argument('-iw', '--width', help='Default image width', required=False, default=128)
parser.add_argument('-ih', '--height', help='Default image height', required=False, default=144)
parser.add_argument('-ks', '--known_set_size', help='Default size of enrolled subjects', required=False, default=0.5)
parser.add_argument('-ts', '--train_set_size', help='Default size of training subset', required=False, default=0.5)
args = parser.parse_args()


def main():
    PATH = str(args.path)
    DATASET = str(args.file)
    DESCRIPTOR = str(args.desc)
    ITERATIONS = int(args.rept)
    KNOWN_SET_SIZE = float(args.known_set_size)
    OUTPUT_NAME = 'features_' + DATASET.replace('.txt','') + '_' + DESCRIPTOR + '_' + str(KNOWN_SET_SIZE) + '_' + str(ITERATIONS)

    print('EXTRACTING FAETURES')
    dataset_list = load_txt_file(PATH + DATASET)
    feat_x, feat_y = extract_features(args, dataset_list)
    
    for index in range(ITERATIONS):
        print('ITERATION #%s' % str(index+1))
        known_tuples, _ = split_known_unknown_sets(dataset_list, known_set_size=KNOWN_SET_SIZE)
        
        known_y = [item[1] for item in known_tuples]
        boolean_y = [1 if item in known_y else 0 for item in feat_y]
        
        f = open(OUTPUT_NAME + '_' + str(index+1) + '.string', 'w')
        for index in range(len(boolean_y)):
            row_y = str(boolean_y[index])
            row_x = feat_x[index]
            f.write(row_y + ' ' + ' '.join(str(x) for x in row_x) + '\n')
        f.close()

        # with open(OUTPUT_NAME + '.features', 'w') as outfile:
        #     pickle.dump([feat_x, feat_y], outfile)
        # print('Done')


def extract_features(arguments, dataset_list):
    PATH = str(arguments.path)
    DATASET = str(arguments.file)
    DESCRIPTOR = str(arguments.desc)
    IMG_WIDTH = int(arguments.width)
    IMG_HEIGHT = int(arguments.height)
    KNOWN_SET_SIZE = float(arguments.known_set_size)
    TRAIN_SET_SIZE = float(arguments.train_set_size)

    matrix_x = []
    matrix_y = []

    vgg_model = None
    if DESCRIPTOR == 'df':
        vgg_model = VGGFace()

    counterA = 0
    for sample in dataset_list:
        sample_path = sample[0]
        sample_name = sample[1]

        subject_path = PATH + sample_path
        subject_image = cv.imread(subject_path, cv.IMREAD_COLOR)

        if DESCRIPTOR == 'hog':
            subject_image = cv.resize(subject_image, (IMG_HEIGHT, IMG_WIDTH))
            feature_vector = Descriptor.get_hog(subject_image)
        elif DESCRIPTOR == 'df':
            feature_vector = Descriptor.get_deep_feature(subject_image, vgg_model, layer_name='fc6')

        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)
        
        counterA += 1
        print(counterA, sample_path, sample_name)

    return matrix_x, matrix_y


if __name__ == "__main__":
    main()