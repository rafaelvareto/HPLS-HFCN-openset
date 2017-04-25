#!/bin/bash

#DF
echo '../main_openset_feat: DF, Variable hashing, 50% Known size'
python ../main_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 100 -ks 0.5
python ../main_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 300 -ks 0.5
python ../main_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 500 -ks 0.5

python ../main_openset_load.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 100 -ks 0.5
python ../main_openset_load.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 300 -ks 0.5
python ../main_openset_load.py -p ../features/ -f PUPFIG83-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 500 -ks 0.5

python ../main_openset_load.py -p ../features/ -f FRGC-SET-4-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 100 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-4-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 300 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-4-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 500 -ks 0.5

python ../main_openset_load.py -p ../features/ -f FRGC-SET-1-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 100 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-1-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 300 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-1-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 500 -ks 0.5

python ../main_openset_load.py -p ../features/ -f FRGC-SET-2-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 100 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-2-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 300 -ks 0.5
python ../main_openset_load.py -p ../features/ -f FRGC-SET-2-DEEP-FEATURE-VECTORS.bin -d df -r 10 -m 500 -ks 0.5