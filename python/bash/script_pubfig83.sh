#!/bin/bash

#HOG    
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 3 -m 100 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 3 -m 200 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 3 -m 300 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 3 -m 400 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d hog -r 3 -m 500 -ks 0.9 -ts 0.9

#DF
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 3 -m 100 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 3 -m 200 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 3 -m 300 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 3 -m 400 -ks 0.9 -ts 0.9
python ../main_openset.py -p ../datasets/pubfig83/ -f pubfig83_set.txt -d df -r 3 -m 500 -ks 0.9 -ts 0.9
