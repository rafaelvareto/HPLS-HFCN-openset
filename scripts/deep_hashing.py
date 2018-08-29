#!/usr/bin/env python -W ignore::DeprecationWarning

from __future__ import print_function
#import os
#os.environ["THEANO_FLAGS"] = "device=gpu0"

import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils

from auxiliar import generate_cmc_curve
from auxiliar import generate_pos_neg_dict
from auxiliar import generate_precision_recall, plot_precision_recall
from auxiliar import generate_roc_curve, plot_roc_curve
from auxiliar import load_txt_file
from auxiliar import split_known_unknown_sets, split_train_test_sets
from descriptor import Descriptor
from pls_classifier import PLSClassifier


file_name = '/Users/filipe/git/plshface/python/features/FRGC-SET-4-DEEP-FEATURE-VECTORS.bin'
with open(file_name, 'rb') as infile:
    matrix_z, matrix_y, matrix_x = pickle.load(infile)


#print(matrix_y)

uniqueIDs = set(matrix_y)
#print(len(uniqueIDs))
numclasses = len(uniqueIDs)


idDict = {}
idInt = 0
for idx in uniqueIDs:
    indexes = [i for i,x in enumerate(matrix_y) if x == idx]
    idDict[idInt] = indexes
    idInt = idInt + 1


randPerm = np.random.permutation(range(0,numclasses))
negIds = randPerm[0:numclasses/2]
posIds = randPerm[numclasses/2:]
ntrain = 2  # number of samples per class

elementsNeg = np.empty([0, 0], dtype=int)
testNeg = np.empty([0, 0], dtype=int)
for id in negIds:
    elementsNeg = np.append(elementsNeg, idDict[id][0:ntrain])
    testNeg = np.append(testNeg, idDict[id][ntrain:])

elementsPos = np.empty([0, 0], dtype=int)
testPos = np.empty([0, 0], dtype=int)
for id in posIds:
    elementsPos = np.append(elementsPos, idDict[id][0:ntrain])
    testPos = np.append(testPos, idDict[id][ntrain:])
    
# class labels
Y = np.zeros(len(elementsNeg), dtype=int)
Y = np.append(Y, np.zeros(len(elementsPos), dtype=int)+1)

Ytest = np.zeros(len(testNeg), dtype=int)
Ytest = np.append(Ytest, np.zeros(len(testPos), dtype=int)+1)

x_array = np.asarray(matrix_x)
X = x_array[elementsNeg,:]
X = np.append(X, x_array[elementsPos,:], axis=0)
#print(X.shape[:])

Xtest = x_array[testNeg,:]
Xtest = np.append(Xtest, x_array[testPos,:], axis=0)
#print(Xtest.shape[:])

#print(Y.shape[:])
#print(Ytest.shape[:])

#print(x_array.shape[:])

nfeatures = x_array.shape[1]
nclasses = 2

######################################################
#PLSH

models = []
splits = []

cmc_score = np.zeros(len(Y))
#print('>> LEARNING PLS MODELS:')
individuals = list(set(Y))
classifier = PLSClassifier()
boolean_label = [key for key in Y]
model = classifier.fit(np.array(X), np.array(boolean_label))

#print('>> LOADING KNOWN PROBE: {0} samples'.format(len(Ytest)))
counterB = 0
counterAcc = 0

for sample in range(0, len(Ytest)):
    feature_vector = Xtest[sample]
    #print (feature_vector)
    vote_dict = dict(map(lambda vote: (vote, 0), Ytest))
    response = model.predict_confidence(feature_vector)
    for pos in pos_list:
        vote_dict[pos] += response

    result = vote_dict.items()
    result.sort(key=lambda tup: tup[1], reverse=True)

    for outer in range(len(individuals)):
        for inner in range(outer + 1):
            if result[inner][0] == Ytest[sample]:
                cmc_score[outer] += 1
                break
    counterB += 1    
    
#print('Accuracy - PLSH: ',(cmc_score[0]/cmc_score[1]) * 100)

    #print(cmc_score,len(Ytest))
    # Getting known set plotting relevant information
    #plotting_labels.append([(sample_name, 1)])
    #plotting_scores.append([(sample_name, output)])


######################################################
# Keras -- FULL CONNECTED



y_train = np_utils.to_categorical(Y, nclasses)
y_test = np_utils.to_categorical(Ytest, nclasses)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(nfeatures,)))
model.add(Dropout(0.2))
model.add(Dense(nclasses, activation='softmax'))

#model.summary()


# In[ ]:

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# In[ ]:

history = model.fit(X, y_train,
                    batch_size=20,
                    nb_epoch=100,
                    verbose=0, 
                    validation_data=(Xtest, y_test))

#print(predictions[:,0])
score = model.evaluate(Xtest, y_test, verbose=0)

print((cmc_score[0]/cmc_score[1]) * 100,' ', score[1]*100)
#print(score[1]*100)




