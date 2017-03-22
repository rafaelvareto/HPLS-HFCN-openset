import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from auxiliar import generate_pos_neg_dict
from auxiliar import generate_precision_recall
from auxiliar import load_txt_file
from descriptor import Descriptor
from pls_classifier import PLSClassifier

# from vggface import VGGFace

NUM_DIM = 128
NUM_hashing = 100

PATH = '/Users/Vareto/Downloads/Databases-feature-frgc1/frgcv1/'
GAL = 'train_1_label.txt'
PRO = 'test_1_label.txt'


def main():
    matrix_x = []
    matrix_y = []
    models = []
    splits = []

    # vgg_model = VGGFace()

    print('>> LOADING GALLERY')
    gallery_list = load_txt_file(PATH + GAL)
    for gallery_sample in gallery_list:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]

        gallery_path = PATH + sample_path
        gallery_image = cv.imread(gallery_path, cv.IMREAD_COLOR)
        gallery_image = cv.resize(gallery_image, (NUM_DIM, NUM_DIM))
        # feature_vector = Descriptor.get_deep_feature(gallery_image, vgg_model, layer_name='fc6')
        feature_vector = Descriptor.get_hog(gallery_image)

        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)

    print('>> SPLITTING POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    cmc_score = np.zeros(len(individuals))
    for index in range(0, NUM_hashing):
        splits.append(generate_pos_neg_dict(individuals))

    print('>> LEARNING PLS MODELS:')
    counter = 0
    for split in splits:
        classifier = PLSClassifier()
        boolean_label = [split[key] for key in matrix_y]
        model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
        models.append((model, split))
        counter += 1
        print counter,
    print(' ')

    print('>> LOADING PROBE')
    counter = 0
    pc_labels = []
    pc_scores = []
    query_list = load_txt_file(PATH + PRO)
    for query_sample in query_list:
        sample_path = query_sample[0]
        sample_name = query_sample[1]

        query_path = PATH + sample_path
        query_image = cv.imread(query_path, cv.IMREAD_COLOR)
        query_image = cv.resize(query_image, (NUM_DIM, NUM_DIM))
        # feature_vector = Descriptor.get_deep_feature(query_image, vgg_model)
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
        values = vote_dict.values()
        print(counter, sample_name, result[0], np.mean(values))
        plt.bar(range(len(individuals)), values)
        plt.savefig('res/closedset_' + str(counter) + '_' + sample_name + '_' + result[0][0])
        counter += 1

        # Getting Precision-Recall relevant information
        pc_label_dict = {key: (1 if key == sample_name else 0) for (key, value) in vote_dict.iteritems()}
        pc_label = pc_label_dict.items()
        pc_label.sort(key=lambda tup: tup[0])
        pc_labels.append(pc_label)
        pc_score = vote_dict.items()
        pc_score.sort(key=lambda tup: tup[0])
        pc_scores.append(pc_score)

    cmc_score = np.divide(cmc_score, counter)
    print(cmc_score)
    generate_precision_recall(individuals, pc_labels, pc_scores)


if __name__ == "__main__":
    main()
