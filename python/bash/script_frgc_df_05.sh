#!/bin/bash

#DF
echo '../main_openset_feat: DF, Variable hashing, 50% Known size'
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_1.txt -d df -r 5 -m 100 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_1.txt -d df -r 5 -m 200 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_1.txt -d df -r 5 -m 300 -ks 0.5

python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_4.txt -d df -r 5 -m 100 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_4.txt -d df -r 5 -m 200 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_4.txt -d df -r 5 -m 300 -ks 0.5

python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_2.txt -d df -r 3 -m 100 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_2.txt -d df -r 3 -m 200 -ks 0.5
python ../main_openset_feat.py -p ../datasets/frgcv1/ -f frgc_set_2.txt -d df -r 3 -m 300 -ks 0.5
