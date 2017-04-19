#!/bin/bash

#DF
echo '../main_openset_feat: DF, Variable hashing, 10% Known size'
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_1.txt -d df -r 5 -m 100 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_1.txt -d df -r 5 -m 200 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_1.txt -d df -r 5 -m 300 -ks 0.1

python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_4.txt -d df -r 5 -m 100 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_4.txt -d df -r 5 -m 200 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_4.txt -d df -r 5 -m 300 -ks 0.1

python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_2.txt -d df -r 3 -m 100 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_2.txt -d df -r 3 -m 200 -ks 0.1
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f set_2.txt -d df -r 3 -m 300 -ks 0.1
