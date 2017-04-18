#!/bin/bash

#DF
echo '../main_openset: DF, Variable hashing, 50% Known size'
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 100 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 200 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 300 -ks 0.9

python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 100 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 200 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 300 -ks 0.9

python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 100 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 200 -ks 0.9
python ../main_openset.py -p ../datasets/vggface/ -f vgg_set.txt -d df -r 2 -m 300 -ks 0.9