#!/bin/bash

#DF
echo '../main_openset_fc: DF, Variable hashing, 10% Known size'
echo 'PUBFIG83'
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 200 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 400 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 200 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 400 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 200 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -r 10 -m 400 -ks 0.9
