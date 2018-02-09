# python default or third-party classes
import argparse
import cv2 as cv
import itertools
import matplotlib
import numpy as np
import pickle

matplotlib.use('Agg')

from joblib import Parallel, delayed
from matplotlib import pyplot

# my classes or files
from auxiliar import compute_fscore, mean_results
from auxiliar import generate_pos_neg_dict, generate_oaa_splits
from auxiliar import generate_det_curve, generate_precision_recall, generate_roc_curve, plot_cmc_curve, plot_det_curve, plot_precision_recall, plot_roc_curve
from auxiliar import learn_plsh_model, learn_oaa_pls
from auxiliar import load_txt_file, set_maximum_samples
from auxiliar import split_known_unknown_sets, split_train_test_sets

# Parsing parameteres
parser = argparse.ArgumentParser(description='PLSH for Face Recognition with NO Feature Extraction')
parser.add_argument('-p', '--path', help='Path do binary feature file', required=False, default='./features/')
parser.add_argument('-f', '--file', help='Input binary feature file name', required=False, default='FRGC-SET-4-DEEP-FEATURE-VECTORS.bin')
parser.add_argument('-r', '--rept', help='Number of executions', required=False, default=1)
parser.add_argument('-m', '--hash', help='Number of hash functions', required=False, default=10)
parser.add_argument('-s', '--samples', help='Number of samples per subject', required=False, default=30, type=int)
parser.add_argument('-ks', '--known_set_size', help='Default size of enrolled subjects', required=False, default=0.5)
parser.add_argument('-ts', '--train_set_size', help='Default size of training subset', required=False, default=0.5)
args = parser.parse_args()

# Keeping parameters in variables (DEFINES)
PATH = str(args.path)
DATASET = str(args.file)
ITERATIONS = int(args.rept)
NUM_HASH = int(args.hash)
SAMPLES = int(args.samples)
KNOWN_SET_SIZE = float(args.known_set_size)
TRAIN_SET_SIZE = float(args.train_set_size)
DATASET = DATASET.replace('-FEATURE-VECTORS.bin','')
OUTPUT_NAME = 'HPLS_' + DATASET + '_' + str(NUM_HASH)  + '_' + str(SAMPLES) + '_' + str(KNOWN_SET_SIZE) + '_' + str(TRAIN_SET_SIZE) + '_' + str(ITERATIONS)

print('>> LOADING FEATURES FROM FILE')
with open(PATH + DATASET, 'rb') as input_file:
    list_of_paths, list_of_labels, list_of_features = pickle.load(input_file)

def main():
    os_cmcs = []
    oaa_cmcs = []
    dets = []
    prs = []
    rocs = []
    fscores = []
    with Parallel(n_jobs=-2, verbose=11, backend='multiprocessing') as parallel_pool:
        for index in range(ITERATIONS):
            print('ITERATION #%s' % str(index+1))
            os_cmc, oaa_cmc, det, pr, roc, fscore = plshface(args, parallel_pool)
            os_cmcs.append(os_cmc)
            oaa_cmcs.append(oaa_cmc)
            dets.append(det)
            prs.append(pr)
            rocs.append(roc)
            fscores.append(fscore)

            with open('./files/' + OUTPUT_NAME + '.file', 'w') as outfile:
                pickle.dump([prs, rocs], outfile)

            plot_cmc_curve(os_cmcs, oaa_cmcs, OUTPUT_NAME)
            plot_det_curve(dets, OUTPUT_NAME)
            plot_precision_recall(prs, OUTPUT_NAME)
            plot_roc_curve(rocs, OUTPUT_NAME)
    
    means = mean_results(fscores)
    with open('./values/' + OUTPUT_NAME + '.txt', 'a') as outvalue:
        for item in fscores:
            outvalue.write(str(item) + '\n')
        for item in means:
            outvalue.write(str(item) + '\n') 
    print(fscores)
    

def plshface(args, parallel_pool):
    matrix_x = []
    matrix_y = []
    plotting_labels = []
    plotting_scores = []
    splits = []
    
    print('>> EXPLORING DATASET')
    dataset_dict = {value:index for index,value in enumerate(list_of_paths)}
    dataset_list = zip(list_of_paths, list_of_labels)
    dataset_list = set_maximum_samples(dataset_list, number_of_samples=SAMPLES)
    # Split dataset into disjoint sets in terms of subjects and samples
    known_tuples, unknown_tuples = split_known_unknown_sets(dataset_list, known_set_size=KNOWN_SET_SIZE)
    known_train, known_test = split_train_test_sets(known_tuples, train_set_size=TRAIN_SET_SIZE)
    to_discard, unknown_test = split_train_test_sets(unknown_tuples, train_set_size=TRAIN_SET_SIZE)

    print('>> LOADING GALLERY: {0} samples'.format(len(known_train)))
    for counterA, gallery_sample in enumerate(known_train):
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = list_of_features[sample_index] 
        # create list of feature vectors and their corresponding target values
        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)
        # print(counterA, sample_path, sample_name)
    
    print('>> SPLITTING POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    os_cmc_score = np.zeros(len(individuals))
    oaa_cmc_score = np.zeros(len(individuals))
    for index in range(0, NUM_HASH):
        splits.append(generate_pos_neg_dict(individuals))

    # Converting list to numpy arrays
    numpy_x = np.array(matrix_x)
    numpy_y = np.array(matrix_y)
    numpy_s = np.array(splits)

    print('>> LEARNING OPEN-SET PLS MODELS:')
    os_models = parallel_pool(
        delayed(learn_plsh_model) (numpy_x, numpy_y, split) for split in numpy_s
    )
    
    print('>> LEARNING CLOSED-SET ONE-AGAINST-ALL PLS MODELS:')
    oaa_splits = generate_oaa_splits(numpy_y)
    oaa_models = parallel_pool(
        delayed(learn_oaa_pls) (numpy_x, split) for split in oaa_splits
    )
  
    print('>> LOADING KNOWN PROBE: {0} samples'.format(len(known_test)))
    for counterB, probe_sample in enumerate(known_test):
        # Obtaining probe feature vector and corresponding identity
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = list_of_features[sample_index] 
        
        # Projecting feature vector to every os_model
        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for model in os_models:
            pos_list = [key for key, value in model[1].iteritems() if value == 1]
            response = model[0].predict_confidence(feature_vector)
            for pos in pos_list:
                vote_dict[pos] += response

        # Sort open-set vote-list histogram
        vote_list = vote_dict.items()
        vote_list.sort(key=lambda tup: tup[1], reverse=True)
        denominator = np.absolute(np.mean([vote_list[1][1], vote_list[2][1], vote_list[3][1]]))
        vote_ratio = vote_list[0][1] / denominator if denominator > 0 else vote_list[0][1]
        # Computer cmc score for open-set classification
        for outer in range(0, len(individuals)):
            for inner in range(0, outer + 1):
                if vote_list[inner][0] == sample_name:
                    os_cmc_score[outer] += 1
                    break

        # Sort closed-set responses 
        ooa_mls_responses = [model[0].predict_confidence(feature_vector) for model in oaa_models]
        ooa_mls_labels = [model[1] for model in oaa_models]
        responses = zip(ooa_mls_labels, ooa_mls_responses)
        responses.sort(key=lambda tup: tup[1], reverse=True)
        # Computer cmc score for closed-set classification
        for outer in range(0, len(individuals)):
            for inner in range(0, outer + 1):
                if responses[inner][0] == sample_name:
                    oaa_cmc_score[outer] += 1
                    break

        # Getting known set plotting relevant information
        plotting_labels.append([(sample_name, 1.0)])
        plotting_scores.append([(sample_name, vote_ratio)])
        # print(counterB, sample_name, vote_ratio, vote_list[0][0], responses[0][0])

    print('>> LOADING UNKNOWN PROBE: {0} samples'.format(len(unknown_tuples)))
    for counterC, probe_sample in enumerate(unknown_test):
        # Obtaining probe feature vector and corresponding identity
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]
        sample_index = dataset_dict[sample_path]
        feature_vector = list_of_features[sample_index] 
        
        # Projecting feature vector to every os_model
        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for model in os_models:
            pos_list = [key for key, value in model[1].iteritems() if value == 1]
            response = model[0].predict_confidence(feature_vector)
            for pos in pos_list:
                vote_dict[pos] += response
        
        # Sort open-set vote-list histogram
        vote_list = vote_dict.items()
        vote_list.sort(key=lambda tup: tup[1], reverse=True)
        denominator = np.absolute(np.mean([vote_list[1][1], vote_list[2][1], vote_list[3][1]]))
        vote_ratio = vote_list[0][1] / denominator if denominator > 0 else vote_list[0][1]

        # Getting unknown set plotting relevant information
        plotting_labels.append([(sample_name, -1.0)])
        plotting_scores.append([(sample_name, vote_ratio)])
        # print(counterC, sample_name, vote_ratio, vote_list[0][0])

    del os_models[:]
    del oaa_models[:]
    
    os_cmc = np.divide(os_cmc_score, len(known_test))
    oaa_cmc = np.divide(oaa_cmc_score, len(known_test))
    det = generate_det_curve(plotting_labels, plotting_scores)
    pr = generate_precision_recall(plotting_labels, plotting_scores)
    roc = generate_roc_curve(plotting_labels, plotting_scores)
    fscore = compute_fscore(pr)
    return os_cmc, oaa_cmc, det, pr, roc, fscore

if __name__ == "__main__":
    main()
