import csv
import numpy as np
import pandas as pd
import nltk
import os
from collections import Counter

TextForElmo = pd.read_csv("textforEmbedding.csv") 
oneString = ''.join(map(str,  TextForElmo["text"].tolist()))
sentence_tokenizer = nltk.data.load('tokenizers/punkt/estonian.pickle')
listOfSentences = sentence_tokenizer.tokenize(oneString)
myTextDF = pd.DataFrame({"transcript":listOfSentences})
train_df = myTextDF.sample(frac=0.8, random_state=100)
test_df = myTextDF[~myTextDF.index.isin(train_df.index)]

if not os.path.exists("swb/train"):
    os.makedirs("swb/train")
print("Created folder")
for i in range(0,train_df.shape[0],6):
    text = "\n".join(train_df['transcript'][i:i+6].tolist())
    fp = open("swb/train/"+str(i)+".txt","w",encoding="UTF-8")
    fp.write(text)
    fp.close()
print("Create TraingDS")

test_df['transcript'] = test_df['transcript'] + " ."
if not os.path.exists("swb/dev"):
    os.makedirs("swb/dev")
 
for i in range(0,test_df.shape[0],6):
    text = "\n".join(test_df['transcript'][i:i+6].tolist())
    fp = open("swb/dev/"+str(i)+".txt","w",encoding="UTF-8")
    fp.write(text)
    fp.close()
print("After validation DS")


vocab = " ".join(train_df['transcript'].tolist())
vocab_words = vocab.split(" ")
print("Number of tokens in Training data = ",len(vocab_words))
dictionary = Counter(vocab_words)
print("Size of Vocab",len(dictionary))
sorted_vocab = ["&lt;S&gt;","&lt;/S&gt;","&lt;UNK&gt;"]
sorted_vocab.extend([pair[0] for pair in dictionary.most_common()])
 
sorted_text = "\n".join(sorted_vocab)
fp = open("swb/vocab.txt","w",encoding="UTF-8")
fp.write(sorted_text)
fp.close()
print("finish")
