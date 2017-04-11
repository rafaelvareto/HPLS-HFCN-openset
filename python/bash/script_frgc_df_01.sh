#!/bin/bash

#DF
echo 'Main_OpenSet: DF, Variable hashing, 10% Known size'
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 100 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 200 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 300 -ks 0.1

python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 100 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 200 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 2 -m 300 -ks 0.1

python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 100 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 200 -ks 0.1
python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 2 -m 300 -ks 0.1
