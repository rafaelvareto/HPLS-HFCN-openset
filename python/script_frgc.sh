#!/bin/bash

#HOG    
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 10 -m 100
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 10 -m 200
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 10 -m 300
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d hog -r 10 -m 400

# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 100
# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 200
# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d hog -r 5 -m 300

echo 'python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5'
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 200 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 400 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 500 -kss 0.1

python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 200 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 400 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 500 -kss 0.5

python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 100 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 200 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 300 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 400 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d hog -r 5 -m 500 -kss 0.9

#DF
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 10 -m 100
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 10 -m 200
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 10 -m 300
# python main_openset.py -p ./frgcv1/ -f set_4_label.txt -d df -r 10 -m 400

# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 5 -m 100
# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 5 -m 200
# python main_openset.py -p ./frgcv1/ -f set_2_label.txt -d df -r 5 -m 300

echo 'python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 5'
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 100 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 200 -kss 0.1
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 300 -kss 0.1

python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 100 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 200 -kss 0.5
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 300 -kss 0.5

python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 100 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 200 -kss 0.9
python main_openset.py -p ./frgcv1/ -f set_1_label.txt -d df -r 2 -m 300 -kss 0.9
