import sys
sys.path.append('../')

import argparse
import cv2 as cv
import numpy as np
import pickle

from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets
from descriptor import Descriptor

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('-p', '--path', help='Path do dataset', required=False, default='../datasets/frgcv1/')
parser.add_argument('-f', '--file', help='Input file name', required=False, default='train_2_small.txt')
parser.add_argument('-d', '--desc', help='Descriptor [hog/df]', required=False, default='hog')
parser.add_argument('-iw', '--width', help='Default image width', required=False, default=128)
parser.add_argument('-ih', '--height', help='Default image height', required=False, default=144)
parser.add_argument('-ks', '--known_set_size', help='Default size of enrolled subjects', required=False, default=0.5)
parser.add_argument('-ts', '--train_set_size', help='Default size of training subset', required=False, default=0.5)
args = parser.parse_args()


def main():
    PATH = str(args.path)
    DATASET = str(args.file)
    DESCRIPTOR = str(args.desc)
    KNOWN_SET_SIZE = float(args.known_set_size)
    OUTPUT_NAME = 'features_' + DATASET.replace('.txt','') + '_' + DESCRIPTOR + '_' + str(KNOWN_SET_SIZE)

    print('EXTRACTING FEATURES')
    dataset_list = load_txt_file(PATH + DATASET)
    feat_z, feat_y, feat_x = extract_features(args, dataset_list)
    
    print('SAVING TO FILE')
    outmatrix = [feat_z, feat_y, feat_x]
    with open(OUTPUT_NAME + '.bin', 'wb') as outfile:
        pickle.dump(outmatrix, outfile, protocol=pickle.HIGHEST_PROTOCOL)


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
    matrix_z = []

    vgg_model = None
    if DESCRIPTOR == 'df':
        from vggface import VGGFace
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
        matrix_z.append(sample_path)
        
        counterA += 1
        print(counterA, sample_path, sample_name)

    return matrix_z, matrix_y, matrix_x


if __name__ == "__main__":
    main()