import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import svmutil
import sys

sys.path.insert(0, './libsvm-3.21/python/')

from itertools import chain
from svmutil import *

from auxiliar import generate_cmc_curve
from auxiliar import generate_pos_neg_dict
from auxiliar import generate_precision_recall, plot_precision_recall
from auxiliar import generate_roc_curve, plot_roc_curve
from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets, split_train_test_sets
from descriptor import Descriptor
from pls_classifier import PLSClassifier

parser = argparse.ArgumentParser(description='PLSH for Face Recognition')
parser.add_argument('-p', '--path', help='Path do dataset', required=False, default='./frgcv1/')
parser.add_argument('-f', '--file', help='Input file name', required=False, default='set_2_label.txt')
parser.add_argument('-d', '--desc', help='Descriptor [hog/df]', required=False, default='hog')
parser.add_argument('-r', '--rept', help='Number of executions', required=False, default=1)
parser.add_argument('-m', '--hash', help='Number of hash functions', required=False, default=100)
parser.add_argument('-iw', '--width', help='Default image width', required=False, default=128)
parser.add_argument('-ih', '--height', help='Default image height', required=False, default=144)
args = parser.parse_args()


def main():
    PATH = str(args.path)
    DATASET = str(args.file)
    DESCRIPTOR = str(args.desc)
    ITERATIONS = int(args.rept)
    NUM_HASH = int(args.hash)
    OUTPUT_NAME = 'OC-SVM_' + DATASET.replace('.txt','') + '_' + str(NUM_HASH) + '_' + DESCRIPTOR + '_' + str(ITERATIONS)

    prs = []
    rocs = []
    for index in range(ITERATIONS):
        print('ITERATION #%s' % str(index+1))
        pr, roc = svm_oneclass(args)
        prs.append(pr)
        rocs.append(roc)

    with open('./files/' + OUTPUT_NAME + '.file', 'w') as outfile:
        pickle.dump([prs, rocs], outfile)

    plot_precision_recall(prs, OUTPUT_NAME)
    plot_roc_curve(rocs, OUTPUT_NAME)

def svm_oneclass(args):
    PATH = str(args.path)
    DATASET = str(args.file)
    DESCRIPTOR = str(args.desc)
    NUM_HASH = int(args.hash)
    IMG_WIDTH = int(args.width)
    IMG_HEIGHT = int(args.height)

    matrix_x = []
    matrix_y = []
    models = []
    splits = []
    nmatrix_x = []
    nmatrix_y = []
    
    x_train=[]
    y_train=[]
    nx_train=[]
    ny_train=[]
    plotting_labels = []
    plotting_scores = []

    vgg_model = None
    if DESCRIPTOR == 'df':
        from vggface import VGGFace
        vgg_model = VGGFace()
    
    print('>> EXPLORING DATASET')
    dataset_list = load_txt_file(PATH + DATASET)
    known_tuples, unknown_tuples = split_known_unknown_sets(dataset_list, known_set_size=0.5)
    known_train, known_test = split_train_test_sets(known_tuples, train_set_size=0.5)
    print(known_train)

    counterA = 0
    for gallery_sample in known_train:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]

        gallery_path = PATH + sample_path
        gallery_image = cv.imread(gallery_path, cv.IMREAD_COLOR)
        
        if DESCRIPTOR == 'hog':
            gallery_image = cv.resize(gallery_image, (IMG_HEIGHT, IMG_WIDTH))
            feature_vector = Descriptor.get_hog(gallery_image)
        elif DESCRIPTOR == 'df':
            feature_vector = Descriptor.get_deep_feature(gallery_image, vgg_model, layer_name='fc6')
    
        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)

        counterA += 1
        print(counterA, sample_path, sample_name)
    
    print('>> GENERATING FILES TO SVM')
    counterSVM=0
    for feature in matrix_x:
        y_train.insert(counterSVM,1)
        x_train.insert(counterSVM,{})
        count_inner=0
        for pos in feature:
            x_train[counterSVM].update({count_inner:pos})
            count_inner+=1
        counterSVM += 1

    print('>> GENERATING THE SVM MODEL');
    x_train_total = x_train + nx_train
    y_train_total = y_train + ny_train	
    besthit = 0
    bestn = 0
    bestg = 0
    for n in range(1,50):
        for g in range(-15,3):
            nu = n/100
            gamma=pow(2,g)
            parameters = '-s 2 -t 2'
            parameters = parameters + ' -g '+str(gamma)+' -n '+str(nu);
            m = svm_train(y_train_total, x_train_total, parameters)
            hits=0
            #print('>> LOADING KNOWN PROBE: {0} samples'.format(len(known_test)))
            counterB = 0
            for probe_sample in known_test:
                sample_path = probe_sample[0]
                sample_name = probe_sample[1]
                query_path = PATH + sample_path
                query_image = cv.imread(query_path, cv.IMREAD_COLOR)
                if DESCRIPTOR == 'hog':
                    query_image = cv.resize(query_image, (IMG_HEIGHT, IMG_WIDTH))
                    feature_vector = Descriptor.get_hog(query_image)
                elif DESCRIPTOR == 'df':
                    feature_vector = Descriptor.get_deep_feature(query_image, vgg_model)
                count_inner=0
                x_teste=[]
                y_teste=[]
                y_teste.insert(0,1)
                x_teste.insert(0,{})
                for pos in feature_vector:
                    x_teste[0].update({count_inner:pos})
                    count_inner+=1
                p_label, p_acc, p_val = svm_predict(y_teste, x_teste, m)
                counterB += 1
                # Getting known set plotting relevant information
                plotting_labels.append([(sample_name, 1)])
                plotting_scores.append([(sample_name, p_label[0])])
                if p_label[0]==1:
                    hits=hits+1
					
            print('>> LOADING UNKNOWN PROBE: {0} samples'.format(len(unknown_tuples)))
            counterC = 0
            for probe_sample in unknown_tuples:
                sample_path = probe_sample[0]
                sample_name = probe_sample[1]
                query_path = PATH + sample_path 
                query_image = cv.imread(query_path, cv.IMREAD_COLOR)
                if DESCRIPTOR == 'hog':
                    query_image = cv.resize(query_image, (IMG_HEIGHT, IMG_WIDTH))
                    feature_vector = Descriptor.get_hog(query_image)
                elif DESCRIPTOR == 'df':    
                    feature_vector = Descriptor.get_deep_feature(query_image, vgg_model)

                count_inner=0
                x_teste=[]
                y_teste=[]
                y_teste.insert(0,-1)
                x_teste.insert(0,{})
                for pos in feature_vector:
                    x_teste[0].update({count_inner:pos})
                    count_inner+=1
                p_label, p_acc, p_val = svm_predict(y_teste, x_teste, m)
                counterC += 1
                # Getting unknown set plotting relevant information
                plotting_labels.append([(sample_name, -1)])
                plotting_scores.append([(sample_name, p_label[0])])
                if p_label[0]==-1:
                    hits=hits+1
            if hits>besthit:
                besthit=hits
                bestn=nu
                bestg=gamma
    # cmc_score_norm = np.divide(cmc_score, counterA)
    # generate_cmc_curve(cmc_score_norm, DATASET + '_' + str(NUM_HASH) + '_' + DESCRIPTOR)
    
    print(besthits)
    print(bestn)
    print(bestg)
    
    pr = generate_precision_recall(plotting_labels, plotting_scores)
    roc = generate_roc_curve(plotting_labels, plotting_scores)
    return pr, roc
	
if __name__ == "__main__":
    main()
