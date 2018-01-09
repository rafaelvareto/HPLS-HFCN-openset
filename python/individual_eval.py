import argparse
import numpy as np
import pickle
import random

from auxiliar import generate_pos_neg_dict
from joblib import Parallel, delayed
from pls_classifier import PLSClassifier


parser = argparse.ArgumentParser(description='PLSH for Face Recognition with NO Feature Extraction')
parser.add_argument('-p', '--path', help='Path do binary feature file', required=False, default='./features/')
parser.add_argument('-f', '--file', help='Input binary feature file name', required=False, default='FRGC-SET-4-DEEP.bin')
parser.add_argument('-r', '--rept', help='Number of executions', required=False, default=100)
parser.add_argument('-m', '--hash', help='Number of hash functions', required=False, default=1)
parser.add_argument('-ts', '--train_set_size', help='Default size of training subset', required=False, default=0.1)
args = parser.parse_args()


PATH = str(args.path)
DATASET = str(args.file)
ITERATIONS = int(args.rept)
NUM_HASH = int(args.hash)
TRAIN_SET_SIZE = float(args.train_set_size)
OUTPUT_NAME = DATASET.replace('.bin','') + '_' + str(NUM_HASH) + '_' + str(ITERATIONS)

print('>> LOADING FEATURES FROM FILE')
with open(PATH + DATASET, 'rb') as input_file:
    list_of_paths, list_of_labels, list_of_features = pickle.load(input_file)

def main():
    hit_rates = []
    with Parallel(n_jobs=-2, verbose=11, backend='multiprocessing') as parallel_pool:
        for index in range(ITERATIONS):
            print('ITERATION #%s' % str(index+1))
            hit = plshface(args, parallel_pool)
            hit_rates.append(hit)
    
    mean_value = np.mean(hit_rates)
    stdv_value = np.std(hit_rates)
    print(mean_value, stdv_value, max(hit_rates), min(hit_rates))


def split_train_test_sets(complete_tuple_list, train_set_size=0.5):
    complete_tuple_dict = dict()
    
    for (path, label) in complete_tuple_list:
        if label in complete_tuple_dict:
            complete_tuple_dict[label].append(path)
        else:
            complete_tuple_dict[label] = [path,]

    test_list = list()
    train_list = list()
    for (key, values) in complete_tuple_dict.iteritems():
        random_number = random.randint(0, len(values))
        for idx in range(len(values)):
            if idx == random_number:
                train_list.append((values[idx], key))
            else:
                test_list.append((values[idx], key))

    return train_list, test_list


def learn_plsh_model(matrix_x, matrix_y, split):
    classifier = PLSClassifier()
    boolean_label = [split[key] for key in matrix_y]
    model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
    return (model, split)


def learn_svmh_model(matrix_x, matrix_y, split):
    classifier = SVR(C=1.0,kernel='linear')
    boolean_label = [split[key] for key in matrix_y]
    model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
    return (model, split)


def plshface(args, parallel_pool):
    matrix_x = []
    matrix_y = []
    plotting_labels = []
    plotting_scores = []
    splits = []
    
    print('>> EXPLORING DATASET')
    dataset_dict = {value:index for index,value in enumerate(list_of_paths)}
    dataset_list = zip(list_of_paths, list_of_labels)
    known_train, known_test = split_train_test_sets(dataset_list, train_set_size=TRAIN_SET_SIZE)

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
        print(counterA, sample_path, sample_name)
    
    print('>> SPLITTING POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    cmc_score = np.zeros(len(individuals))
    for index in range(0, NUM_HASH):
        splits.append(generate_pos_neg_dict(individuals))

    print('>> LEARNING PLS or SVM MODELS:')
    numpy_x = np.array(matrix_x)
    numpy_y = np.array(matrix_y)
    numpy_s = np.array(splits)
    models = parallel_pool(
        delayed(learn_plsh_model) (numpy_x, numpy_y, split) for split in numpy_s # Change here PLS/SVM
    )

    print('>> LOADING KNOWN PROBE: {0} samples'.format(len(known_test)))
    counterB = 0
    counter_fn_tp = 0.0
    counter_tp = 0.0
    for probe_sample in known_test:
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = list_of_features[sample_index]

        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for model in models:
            pos_list = [key for key, value in model[1].iteritems() if value == 1]
            response = model[0].predict_confidence(feature_vector) # PLS
            # response = model[0].predict(np.float32(feature_vector).reshape(1, -1)) # SVM
            if sample_name in pos_list:
                counter_fn_tp += 1
            if response > 0 and sample_name in pos_list:
                counter_tp += 1
    result = counter_tp/counter_fn_tp
    print('>> HIT RATE', result)

    return result

if __name__ == "__main__":
    main()
