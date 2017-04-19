#!/bin/bash

#DF
echo '../main_openset: DF, Variable hashing, 10% Known size'
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d df -r 5 -m 100 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d df -r 5 -m 200 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_1_label.txt -d df -r 5 -m 300 -ks 0.1

python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d df -r 5 -m 100 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d df -r 5 -m 200 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_4_label.txt -d df -r 5 -m 300 -ks 0.1

python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d df -r 3 -m 100 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d df -r 3 -m 200 -ks 0.1
python ../main_openset.py -p ../datasets/frgcv1/ -f set_2_label.txt -d df -r 3 -m 300 -ks 0.1
