#!/bin/bash

#DF
echo 'PUBFIG83: DF, Variable hashing, 20% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 010 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 020 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 030 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 050 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 100 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 300 -s 30 -ks 0.2 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 500 -s 30 -ks 0.2 -ts 0.5

echo 'PUBFIG83: DF, Variable hashing, 40% Known size, 50% Train/Test'
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 010 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 020 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 030 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 050 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 100 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 300 -s 40 -ks 0.4 -ts 0.5
python ../HPLS_openset_load.py -p ../features/ -f PUBFIG83-DEEP.bin -r 10 -m 500 -s 40 -ks 0.4 -ts 0.5