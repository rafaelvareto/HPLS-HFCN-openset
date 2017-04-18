#!/bin/bash

#HOG    
echo '../main_openset: HOG, Variable hashing, 50% Known size'
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 200 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d hog -r 5 -m 400 -ks 0.5

python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 200 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d hog -r 5 -m 400 -ks 0.5

python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 5 -m 100 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 5 -m 200 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 5 -m 300 -ks 0.5
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d hog -r 5 -m 400 -ks 0.5

