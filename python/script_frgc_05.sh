#!/bin/bash

#HOG    
echo 'Main_OpenSet: HOG, Variable hashing, 10% Known size'
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 400 -ks 0.5

python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 5 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 5 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 5 -m 300 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 5 -m 400 -ks 0.5

python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 300 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 400 -ks 0.5


#DF
echo 'Main_OpenSet: DF, Variable hashing, 10% Known size'
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 300 -ks 0.5

python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 300 -ks 0.5

python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 100 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 200 -ks 0.5
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 300 -ks 0.5
