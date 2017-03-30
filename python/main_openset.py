import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from auxiliar import generate_cmc_curve
from auxiliar import generate_pos_neg_dict
from auxiliar import generate_precision_recall
from auxiliar import generate_roc_curve
from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets
from auxiliar import split_train_test_sets

from descriptor import Descriptor
from vggface import VGGFace
from pls_classifier import PLSClassifier


IMG_WIDTH = 128
IMG_HEIGHT = 144
NUM_HASH = 100


SETNAME = 'set_4'
DATASET = SETNAME + '_label.txt'
PATH = './frgcv1/'


def main():
    matrix_x = []
    matrix_y = []
    models = []
    splits = []

    plotting_labels = []
    plotting_scores = []

    vgg_model = VGGFace()
    
    print('>> EXPLORING DATASET')
    dataset_list = load_txt_file(PATH + DATASET)
    known_tuples, unknown_tuples = split_known_unknown_sets(dataset_list, known_set_size=0.5)
    known_train, known_test = split_train_test_sets(known_tuples, train_set_size=0.5)

    print('>> LOADING GALLERY: {0} samples'.format(len(known_train)))
    for gallery_sample in known_train:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]
        
        gallery_path = PATH + sample_path
        gallery_image = cv.imread(gallery_path, cv.IMREAD_COLOR)
        # gallery_image = cv.resize(gallery_image, (IMG_HEIGHT, IMG_WIDTH))
        # feature_vector = Descriptor.get_hog(gallery_image)
        feature_vector = Descriptor.get_deep_feature(gallery_image, vgg_model, layer_name='fc6')
    
        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)
    
    print('>> SPLITTING POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    cmc_score = np.zeros(len(individuals))
    for index in range(0, NUM_HASH):
        splits.append(generate_pos_neg_dict(individuals))

    print('>> LEARNING PLS MODELS:')
    counter = 0
    for split in splits:
        classifier = PLSClassifier()
        boolean_label = [split[key] for key in matrix_y]
        model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
        models.append((model, split))
        counter += 1
        print(counter)
  
    print('>> LOADING KNOWN PROBE: {0} samples'.format(len(known_test)))
    counterA = 0
    for probe_sample in known_test:
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]

        query_path = PATH + sample_path
        query_image = cv.imread(query_path, cv.IMREAD_COLOR)
        query_image = cv.resize(query_image, (IMG_HEIGHT, IMG_WIDTH))
        feature_vector = Descriptor.get_hog(query_image)

        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for model in models:
            pos_list = [key for key, value in model[1].iteritems() if value == 1]
            response = model[0].predict_confidence(feature_vector)
            for pos in pos_list:
                vote_dict[pos] += response
        result = vote_dict.items()
        result.sort(key=lambda tup: tup[1], reverse=True)

        for outer in range(len(individuals)):
            for inner in range(outer + 1):
                if result[inner][0] == sample_name:
                    cmc_score[outer] += 1
                    break
        
        counterA += 1
        denominator = np.absolute(np.mean([result[1][1], result[2][1]]))
        if denominator > 0:
            output = result[0][1] / denominator
        else:
            output = result[0][1]
        print(counterA, sample_name, result[0][0], output)

        # Getting known set plotting relevant information
        plotting_labels.append([(sample_name, 1)])
        plotting_scores.append([(sample_name, output)])

    print('>> LOADING UNKNOWN PROBE: {0} samples'.format(len(unknown_tuples)))
    counterB = 0
    for probe_sample in unknown_tuples:
        sample_path = probe_sample[0]
        sample_name = probe_sample[1]

        query_path = PATH + sample_path 
        query_image = cv.imread(query_path, cv.IMREAD_COLOR)
        # query_image = cv.resize(query_image, (IMG_HEIGHT, IMG_WIDTH))
        # feature_vector = Descriptor.get_hog(query_image)
        feature_vector = Descriptor.get_deep_feature(query_image, vgg_model)

        vote_dict = dict(map(lambda vote: (vote, 0), individuals))
        for model in models:
            pos_list = [key for key, value in model[1].iteritems() if value == 1]
            response = model[0].predict_confidence(feature_vector)
            for pos in pos_list:
                vote_dict[pos] += response
        result = vote_dict.items()
        result.sort(key=lambda tup: tup[1], reverse=True)

        counterB += 1
        denominator = np.absolute(np.mean([result[1][1], result[2][1]]))
        if denominator > 0:
            output = result[0][1] / denominator
        else:
            output = result[0][1]
        print(counterB, sample_name, result[0][0], output)

        # Getting unknown set plotting relevant information
        plotting_labels.append([(sample_name, -1)])
        plotting_scores.append([(sample_name, output)])

    cmc_score_norm = np.divide(cmc_score, counterA)
    generate_cmc_curve(cmc_score_norm, SETNAME + '_' + str(NUM_HASH))
    generate_precision_recall(1, plotting_labels, plotting_scores, SETNAME + '_' + str(NUM_HASH))
    generate_roc_curve(1, plotting_labels, plotting_scores, SETNAME + '_' + str(NUM_HASH))
    


if __name__ == "__main__":
    main()
