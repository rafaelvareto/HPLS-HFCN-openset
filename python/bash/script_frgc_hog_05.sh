#!/bin/bash

#HOG    
echo '../main_openset: HOG, Variable hashing, 50% Known size'
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 500 -ks 0.5

python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 500 -ks 0.5

python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 3 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 3 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 3 -m 500 -ks 0.5

