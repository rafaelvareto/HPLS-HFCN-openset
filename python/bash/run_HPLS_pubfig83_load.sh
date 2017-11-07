#!/bin/bash

#DF
echo 'PUPFIG83: DF, Variable hashing, 10% Known size, 90% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 10 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 20 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 30 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 50 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 100 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 300 -ks 0.1 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 500 -ks 0.1 -ts 0.9

echo 'PUPFIG83: DF, Variable hashing, 50% Known size, 90% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 10 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 20 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 30 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 50 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 100 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 300 -ks 0.5 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 500 -ks 0.5 -ts 0.9

echo 'PUPFIG83: DF, Variable hashing, 90% Known size, 90% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 10 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 20 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 30 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 50 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 100 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 300 -ks 0.9 -ts 0.9
python ../HPLS_openset_load.py -p ../features/ -f PUPFIG83-DEEP.bin -r 10 -m 500 -ks 0.9 -ts 0.9