import cv2 as cv
import numpy as np

from classes import Classes
from descriptor import Descriptor
from descriptor_aux import load_images, load_txt_file

image_path = 'icbrw_cropped/icbrw_GalleryImages'
image_file = 'icbrw_cropped/annotations_GalleryImages.txt'


def main():
    list = load_txt_file(image_file)
    gallerySet = Classes()

    # Load images and extracting features
    for name in list:
        image = cv.imread(image_path + '/' + name, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (150, 150))
        desc_hog = Descriptor.get_hog(image)
        gallerySet.add_element(desc_hog, name)

    gallerySet.shuffle_labels(3)
    data = gallerySet.learn_models()

    print data


if __name__ == "__main__": main()
