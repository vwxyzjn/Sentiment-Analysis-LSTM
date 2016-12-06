"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

from six.moves import cPickle
import numpy
import os
import pandas as pd
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer



#target_file = open('imdb_full.pkl', 'rb')
#(x_train, labels_train), (x_test, labels_test) = cPickle.load(target_file)
#train, test = cPickle.load(target_file)

path = ["books/", "camera/", "dvd/", "health/", "music/", "software/"]  # camera, has missing data
ff = []
input_label = []
for i in range(6):
    ff += [path[i] + "train/pos/" + x for x in os.listdir(path[i] + "train/pos")] + \
         [path[i] + "train/neg/" + x for x in os.listdir(path[i] + "train/neg")] + \
         [path[i] + "test/pos/" + x for x in os.listdir(path[i] + "test/pos")] + \
         [path[i] + "test/neg/" + x for x in os.listdir(path[i] + "test/neg")]
    
    # Because of missing data, we need to measure how many reviews are there in each folder in order to label them correctly.
    train_pos = len(os.listdir(path[i] + "train/pos"))
    train_neg = len(os.listdir(path[i] + "train/neg"))
    test_pos = len(os.listdir(path[i] + "test/pos"))
    test_neg = len(os.listdir(path[i] + "test/neg"))
    input_label += [1] * train_pos + [0] * train_neg + [1] * test_pos + [0] * test_neg
    
      
input_text  = []
for f in ff:
    with open(f, 'rb') as fin:
        temp = fin.read().splitlines()
        x = " ".join([x.decode("utf-8", errors = 'ignore') for x in temp])
        input_text += [x]

cut_index = int(len(input_text)/2)
tok = Tokenizer()
tok.fit_on_texts(input_text[:cut_index])

X_train = tok.texts_to_sequences(input_text[:cut_index])
X_test  = tok.texts_to_sequences(input_text[cut_index:])
y_train = input_label[:cut_index]
y_test  = input_label[cut_index:]

# reconstruct word
words = {k:v for v,k in tok.word_index.items()}
def reconstruct_text(index, words):
    text = []
    for ind in index:
        if ind != 0:
            text += [words[ind]]
        else:
            text += [""]
    return text
    
print(input_text[100])
print("\n\n")
print(reconstruct_text(X_train[100], words))

f = open('amzn_full.pkl', 'wb')
train_tuple = (X_train, y_train)
test_tuple = (X_test, y_test)
combine = (train_tuple, test_tuple)
cPickle.dump(combine, f)
f.close()