import cv2 as cv
import numpy as np

#from vggface import VGGFace

from auxiliar import generate_pos_neg_dict
from auxiliar import load_txt_file
from descriptor import Descriptor

NUM_DIM = 128
NUM_hashing = 100

PATH = '/home/vareto/Downloads/Databases-feature-frgc1/frgcv1/'
GAL = 'train_2_label.txt'
PRO = 'test_2_label.txt'


def main():
    matrix_x = []
    matrix_y = []
    scores_y = []

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

    print('>> POSITIVE/NEGATIVE SETS')
    individuals = list(set(matrix_y))
    for index in range(0, NUM_hashing):
        value = generate_pos_neg_dict(individuals)

    #np.random.binomial()




if __name__ == "__main__":
    main()
