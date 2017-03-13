import cv2 as cv
import numpy as np

from auxiliar import generate_pos_neg_dict
from auxiliar import load_txt_file
from descriptor import Descriptor
from pls_classifier import PLSClassifier
#from vggface import VGGFace

NUM_DIM = 128
NUM_hashing = 100

PATH = '/home/vareto/Downloads/Databases-feature-frgc1/frgcv1/'
GAL = 'train_1_label(copy).txt'
PRO = 'test_1_label(copy).txt'


def main():
    matrix_x = []
    matrix_y = []
    models = []
    scores_y = []
    splits = []

    #vgg_model = VGGFace()

    print('>> LOADING GALLERY')
    gallery_list = load_txt_file(PATH + GAL)
    for gallery_sample in gallery_list:
        sample_path = gallery_sample[0]
        sample_name = gallery_sample[1]

        gallery_path = PATH + sample_path
        gallery_image = cv.imread(gallery_path, cv.IMREAD_COLOR)
        gallery_image = cv.resize(gallery_image, (NUM_DIM, NUM_DIM))
        #feature_vector = Descriptor.get_deep_feature(gallery_image, vgg_model, layer_name='fc6')
        feature_vector = Descriptor.get_hog(gallery_image)

        matrix_x.append(feature_vector)
        matrix_y.append(sample_name)
        scores_y.append(0)
        scores_y.append(0)

    print('>> SPLITTING POSITIVE_NEGATIVE SETS')
    individuals = list(set(matrix_y))
    for index in range(0, NUM_hashing):
        splits.append(generate_pos_neg_dict(individuals))

    print('>> PLS CLASSIFICATION: LEARNING MODELS')
    counter = 0
    for split in splits:
        classifier = PLSClassifier()
        boolean_label = [split[key] for key in matrix_y]
        model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
        models.append((model,split))
        counter += 1
        print(counter)

    print('>> LOADING PROBE')
    counter = 0
    query_list = load_txt_file(PATH + PRO)
    for query_sample in query_list:
        sample_path = query_sample[0]
        sample_name = query_sample[1]

        query_path = PATH + sample_path
        query_image = cv.imread(query_path, cv.IMREAD_COLOR)
        query_image = cv.resize(query_image, (NUM_DIM, NUM_DIM))
        feature_vector = Descriptor.get_hog(query_image)

        response = []
        for model in models:
            # ans = model[0].predict(feature_vector).tolist()
            response.append(model[0].predict(feature_vector).tolist()[0])
        print('Done')


if __name__ == "__main__":
    main()
