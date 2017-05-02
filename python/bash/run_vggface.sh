#!/bin/bash

#DF
echo '../main_openset_fc: DF, Variable hashing, 10% Known size'
echo 'VGGFACE'
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 100 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 200 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 300 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 400 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 500 -ks 0.1
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 100 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 200 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 300 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 400 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 500 -ks 0.5
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 100 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 200 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 300 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 400 -ks 0.9
python ../main_openset_fc.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 5 -m 500 -ks 0.9



