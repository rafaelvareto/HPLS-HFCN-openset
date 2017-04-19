#!/bin/bash

#HOG
echo '../main_openset_feat: HOG, Variable hashing, 10% Known size'
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d hog -r 5 -m 100 -ks 0.1
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d hog -r 5 -m 300 -ks 0.1
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d hog -r 5 -m 500 -ks 0.1
