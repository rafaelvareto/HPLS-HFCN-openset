from __future__ import print_function

import argparse
import cv2 as cv
import itertools
import matplotlib
import numpy as np
import os
import pickle
import time

matplotlib.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from auxiliar import compute_fscore
from auxiliar import generate_cmc_curve
from auxiliar import generate_pos_neg_dict
from auxiliar import generate_precision_recall, plot_precision_recall
from auxiliar import generate_roc_curve, plot_roc_curve
from auxiliar import load_txt_file, set_maximum_samples
from auxiliar import split_known_unknown_sets, split_train_test_sets
from joblib import Parallel, delayed
from matplotlib import pyplot

import keras
import keras.models
import keras.backend.tensorflow_backend as keras_backend
from keras.models import Sequential as keras_sequential
from keras.layers import Dense as keras_dense 
from keras.layers import Dropout as keras_dropout
from keras.utils import np_utils as keras_np_utils
import tensorflow

import types
import tempfile

parser = argparse.ArgumentParser(description='PLSH for Face Recognition with NO Feature Extraction')
parser.add_argument('-p', '--path', help='Path do binary feature file', required=False, default='./features/')
parser.add_argument('-f', '--file', help='Input binary feature file name', required=False, default='FRGC-SET-4-DEEP-FEATURE-VECTORS.bin')
parser.add_argument('-r', '--rept', help='Number of executions', required=False, default=1)
parser.add_argument('-m', '--hash', help='Number of hash functions', required=False, default=100)
parser.add_argument('-s', '--samples', help='Number of samples per subject', required=False, default=30, type=int)
parser.add_argument('-ks', '--known_set_size', help='Default size of enrolled subjects', required=False, default=0.5)
parser.add_argument('-ts', '--train_set_size', help='Default size of training subset', required=False, default=0.5)
args = parser.parse_args()

PATH = str(args.path)
DATASET = str(args.file)
ITERATIONS = int(args.rept)
NUM_HASH = int(args.hash)
SAMPLES = int(args.samples)
KNOWN_SET_SIZE = float(args.known_set_size)
TRAIN_SET_SIZE = float(args.train_set_size)

DATASET = DATASET.replace('-FEATURE-VECTORS.bin','')
OUTPUT_NAME = 'HFCN_' + DATASET + '_' + str(NUM_HASH) + '_' + str(KNOWN_SET_SIZE) + '_' + str(TRAIN_SET_SIZE) + '_' + str(ITERATIONS)


print('>> LOADING FEATURES FROM FILE')
with open(PATH + DATASET, 'rb') as input_file:
    list_of_paths, list_of_labels, list_of_features = pickle.load(input_file)


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def getModel(input_shape,nclasses=2):
    make_keras_picklable()
    model = keras_sequential()
    model.add(keras_dense(64, activation='relu', input_shape=input_shape))
    model.add(keras_dropout(0.2))
    model.add(keras_dense(nclasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])#RMSprop()
    return model


def learn_fc_model(X, Y, split):
    boolean_label = [(split[key]+1)/2 for key in Y]
    y_train = keras_np_utils.to_categorical(boolean_label, 2)
    model = getModel(input_shape=X[0].shape)
    model.fit(X, y_train, batch_size=40, nb_epoch=100, verbose=0)
    return (model, split)


def main():
    fscores = []
    prs = []
    rocs = []
    with Parallel(n_jobs=1, verbose=15, backend='multiprocessing') as parallel_pool:
        for index in range(ITERATIONS):
            keras_backend.clear_session()
            keras_session = tensorflow.Session()
            keras_backend.set_session(keras_session)

            print('ITERATION #%s' % str(index+1))
            pr, roc, fscore = fcnhface(args, parallel_pool)
            fscores.append(fscore)
            prs.append(pr)
            rocs.append(roc)

            with open('./files/plot_' + OUTPUT_NAME + '.file', 'w') as outfile:
                pickle.dump([prs, rocs], outfile)

            plot_precision_recall(prs, OUTPUT_NAME)
            plot_roc_curve(rocs, OUTPUT_NAME)
    print(fscores)


def fcnhface(args, parallel_pool):
    matrix_x = []
    matrix_y = []
    plotting_labels = []
    plotting_scores = []
    models = []
    splits = []
    
    print('>> EXPLORING DATASET')
    dataset_dict = {value:index for index,value in enumerate(list_of_paths)}
    dataset_list = zip(list_of_paths, list_of_labels)
    dataset_list = set_maximum_samples(dataset_list, number_of_samples=SAMPLES)
    known_tuples, unknown_tuples = split_known_unknown_sets(dataset_list, known_set_size=KNOWN_SET_SIZE)
    known_train, known_test = split_train_test_sets(known_tuples, train_set_size=TRAIN_SET_SIZE)

    print('>> LOADING GALLERY: {0} samples'.format(len(known_train)))
    counterA = 0
    for gallery_sample in known_train:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = list_of_features[sample_index] 

        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)

        counterA += 1
        # print(counterA, sample_path, sample_name)
    
    print('>> SPLITTING POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    cmc_score = np.zeros(len(individuals))
    for index in range(0, NUM_HASH):
        splits.append(generate_pos_neg_dict(individuals))

    print('>> LEARNING FC MODELS:')
    numpy_x = np.array(matrix_x)
    numpy_y = np.array(matrix_y)
    numpy_s = np.array(splits)

    # models = [learn_fc_model(numpy_x, numpy_y, split) for split in numpy_s]

    models = parallel_pool(
        delayed(learn_fc_model) (numpy_x, numpy_y, split) for split in numpy_s
    )

    print('>> LOADING KNOWN PROBE: {0} samples'.format(len(known_test)))
    counterB = 0
    for probe_sample in known_test:
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = np.array(list_of_features[sample_index])

        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for k in range (0, len(models)):
            pos_list = [key for key, value in models[k][1].iteritems() if value == 1]
            pred = models[k][0].predict(feature_vector.reshape(1, feature_vector.shape[0]))
            response = pred[0][1]
            #print(response)
            for pos in pos_list:
                vote_dict[pos] += response
        result = vote_dict.items()
        result.sort(key=lambda tup: tup[1], reverse=True)

        for outer in range(len(individuals)):
            for inner in range(outer + 1):
                if result[inner][0] == sample_name:
                    cmc_score[outer] += 1
                    break

        counterB += 1
        denominator = np.absolute(np.mean([result[1][1], result[2][1]]))
        if denominator > 0:
            output = result[0][1] / denominator
        else:
            output = result[0][1]
        # print(counterB, sample_name, result[0][0], output)

        # Getting known set plotting relevant information
        plotting_labels.append([(sample_name, 1)])
        plotting_scores.append([(sample_name, output)])

    print('>> LOADING UNKNOWN PROBE: {0} samples'.format(len(unknown_tuples)))
    counterC = 0
    for probe_sample in unknown_tuples:
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = np.array(list_of_features[sample_index])

        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        #print (vote_dict)
        for k in range (0, len(models)):
            pos_list = [key for key, value in models[k][1].iteritems() if value == 1]
            pred = models[k][0].predict(feature_vector.reshape(1, feature_vector.shape[0]))
            response = pred[0][1]
            for pos in pos_list:
                vote_dict[pos] += response
        result = vote_dict.items()
        result.sort(key=lambda tup: tup[1], reverse=True)

        counterC += 1
        denominator = np.absolute(np.mean([result[1][1], result[2][1]]))
        if denominator > 0:
            output = result[0][1] / denominator
        else:
            output = result[0][1]
        # print(counterC, sample_name, result[0][0], output)

        # Getting unknown set plotting relevant information
        plotting_labels.append([(sample_name, -1)])
        plotting_scores.append([(sample_name, output)])

    # cmc_score_norm = np.divide(cmc_score, counterA)
    # generate_cmc_curve(cmc_score_norm, DATASET + '_' + str(NUM_HASH) + '_' + DESCRIPTOR)

    del models[:]
    
    pr = generate_precision_recall(plotting_labels, plotting_scores)
    roc = generate_roc_curve(plotting_labels, plotting_scores)
    fscore = compute_fscore(pr)
    return pr, roc, fscore

if __name__ == "__main__":
    main()
