#!/bin/bash

#DF
echo '../main_openset_feat: DF, Variable hashing, 50% Known size'
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d df -r 5 -m 100 -ks 0.5
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d df -r 5 -m 300 -ks 0.5
python ../main_openset_feat.py -p ../datasets/vggface/files/ -f vgg_set.txt -d df -r 5 -m 500 -ks 0.5
