#!/bin/bash

#HOG    
echo '../main_openset_feat: HOG, Variable hashing, 90% Known size'
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 5 -m 100 -ks 0.9 -ts 0.9
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 5 -m 300 -ks 0.9 -ts 0.9
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 5 -m 500 -ks 0.9 -ts 0.9

#DF
echo '../main_openset_feat: DF, Variable hashing, 90% Known size'
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 5 -m 100 -ks 0.9 -ts 0.9
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 5 -m 300 -ks 0.9 -ts 0.9
python ../main_openset_feat.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 5 -m 500 -ks 0.9 -ts 0.9
