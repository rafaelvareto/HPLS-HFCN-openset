#!/bin/bash

#HOG    
python main_openset.py -p ./pubfig83/ -f set_label.txt -d hog -r 5 -m 100 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d hog -r 5 -m 200 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d hog -r 5 -m 300 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d hog -r 5 -m 400 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d hog -r 5 -m 500 -ks 0.9 -ts 0.9

#DF
python main_openset.py -p ./pubfig83/ -f set_label.txt -d df -r 5 -m 100 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d df -r 5 -m 200 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d df -r 5 -m 300 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d df -r 5 -m 400 -ks 0.9 -ts 0.9
python main_openset.py -p ./pubfig83/ -f set_label.txt -d df -r 5 -m 500 -ks 0.9 -ts 0.9
