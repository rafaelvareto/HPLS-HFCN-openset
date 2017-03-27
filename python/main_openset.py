import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets
from auxiliar import split_train_test_sets
from auxiliar import generate_pos_neg_dict


IMG_WIDTH = 128
IMG_HEIGHT = 144
NUM_HASH = 100

DATASET = 'set_1_label.txt'
PATH = './frgcv1/'
SETNAME = 'openset'


def main():
    matrix_x = []
    matrix_y = []
    
    print('>> EXPLORING DATASET')
    dataset_list = load_txt_file(PATH + DATASET)
    known_tuple, unknown_tuple = split_known_unknown_sets(dataset_list, known_set_size=0.5)
    known_train, known_test = split_train_test_sets(known_tuple, train_set_size=0.5)

    print('>> LOADING GALLERY')
    for gallery_sample in known_train:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]
        
        gallery_path = PATH + sample_path
        gallery_image = cv.imread(gallery_path, cv.IMREAD_COLOR)
        gallery_image = cv.resize(gallery_image, (IMG_HEIGHT, IMG_WIDTH))
        feature_vector = Descriptor.get_hog(gallery_image)
    
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
        print (counter)
    
    print('>> LOADING PROBE')
    counter = 0
    probe_plot_labels = []
    probe_plot_scores = []

    for probe_sample in known_test:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]

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




if __name__ == "__main__":
    main()