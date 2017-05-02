#!/bin/bash

#HOG
echo 'VGGFACE: HOG, Variable hashing, 10% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.1 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.1 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.1 -ts 0.5

echo 'VGGFACE: HOG, Variable hashing, 50% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.5 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.5 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.5 -ts 0.5

echo 'VGGFACE: HOG, Variable hashing, 90% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.9 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.9 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-HOG-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.9 -ts 0.5

#DF
echo 'VGGFACE: DF, Variable hashing, 10% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.1 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.1 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.1 -ts 0.5

echo 'VGGFACE: DF, Variable hashing, 50% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.5 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.5 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.5 -ts 0.5

echo 'VGGFACE: DF, Variable hashing, 90% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 100 -ks 0.9 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 300 -ks 0.9 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f VGGFACE-15-DEEP-FEATURE-VECTORS.bin -r 10 -m 500 -ks 0.9 -ts 0.5