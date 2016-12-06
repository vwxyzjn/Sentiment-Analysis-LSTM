'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.layers.wrappers import TimeDistributed
import pickle
from six.moves import cPickle
import pandas as pd
import math
import matplotlib.pyplot as plt


max_features = 20000
maxlen = 180  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(XX_train, y_train), (XX_test, y_test) = imdb.load_data(path='imdb_full.pkl', nb_words=max_features, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=0)
print(len(XX_train), 'train sequences')
print(len(XX_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(XX_train, maxlen=maxlen)
X_test = sequence.pad_sequences(XX_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
          validation_data=(X_test, y_test))


############################################################################################
# below is what we wrote
# PLEASE CHECK https://s3.amazonaws.com/text-datasets/imdb_full.pkl  in the same folder as this program.
##############################################################################################

def translate(array, index_to_word):
    string = ""
    for i in array:
        string += " " + index_to_word[i]
        
    return string
    
def translate_df(df, row, index_to_word):
    string = ""
    for i in range(len(df.loc[0])):
        if not math.isnan(df.loc[row, i]):
            string += " " + index_to_word[int(df.loc[row, i])]
        
    return string    

    
def decode(string, word_to_index):
    """given a string, return a list of integers that represents the string: for each
       integer represents the populartiy of that word
       NOTE: DON'T GIVE PUNCTUATION
    """
    l = string.split()
    for i in range(len(l)):
        l[i] = word_to_index[l[i]]
    
    return l

def create_eq_len_list(lst):
    l = []
    for x in lst:
        l += [1]
    
    return l

def predict_string(string, word_to_index, model):
    array = decode(string, word_to_index)
    equal_len_array = create_eq_len_list(array)
    return model.predict([array, equal_len_array])[0]
                         


""" Visualization part """                         
def reconstruct_text(index, index_to_word):
    text = []
    for ind in index:
        if ind != 0:
            text += [index_to_word[ind]]
        else:
            text += [""]
    return text

word_to_index = imdb.get_word_index()
index_to_word = {k:v for v,k in word_to_index.items()}
f = open('imdb_full.pkl', 'rb')
(x_train, labels_train), (x_test, labels_test) = cPickle.load(f)
f.close()
df = pd.DataFrame(x_train)

model2 = Sequential()
model2.add(Embedding(max_features, 128, dropout=0.2))
model2.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))  # try using a GRU instead, for fun
model2.add(TimeDistributed(Dense(1)))
model2.add(Activation('sigmoid'))
model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model2.set_weights(model.get_weights())
y_hat2 = model2.predict(X_train)

ind = 100
tokens = reconstruct_text(X_train[ind], index_to_word)

plt.figure(figsize=(16, 10))
plt.plot(y_hat2[ind],alpha=0.5)
for i in range(len(tokens)):
    plt.text(i,0.5,tokens[i],rotation=90)


# testing
#print(translate_df(df, 0, index_to_word))
#print()
#print(df)
#print()
#print(predict_string('this movie is so bad i just want to leave i hate all the people in this movie', word_to_index, model))
#print(predict_string('one of the best movie i have ever seen the hero is so great', word_to_index, model))