import csv
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle 



DS = pd.read_csv("ObjectSubset150SentenceLength.csv")
DS['tag'].fillna('text', inplace=True)
words = list(set(DS["word"].values))
n_words = len(words)
tags = list(set(DS["tag"].values))
n_tags = len(tags)

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("text_ID").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
getter = SentenceGetter(DS)
sentences = getter.sentences

max_len = 150
tags2 = ["PAD","object","text"] 
tag2idx = {t: i + 1 for i, t in enumerate(tags2)}
# tag2idx["PAD"] = 0

X = [[w[0] for w in s] for s in sentences]

new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PAD")
    new_X.append(new_seq)
X = new_X

y = [[tag2idx[w[1]] for w in s] for s in sentences]

from keras.preprocessing.sequence import pad_sequences
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PAD"], truncating='post')

idx2tag = {i: w for w, i in tag2idx.items()}

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=2018)
batch_size = 32


import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K


from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "options.json"
weight_file = "swb_weights.hdf5"
elmo_model  = Elmo(options_file, weight_file, 2, dropout=0)

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda


X_tr_character_ids = batch_to_ids(X_tr)
X_tr_embeddings = elmo_model(X_tr_character_ids)
pickle.dump(X_tr_embeddings, open('X_tr_embeddings.pickle', 'wb'))

X_te_character_ids = batch_to_ids(X_te)
X_te_embeddings = elmo_model(X_te_character_ids)
pickle.dump(X_te_embeddings, open('X_te_embeddings.pickle', 'wb'))



