import csv
import numpy as np
import pandas as pd
from estnltk import Text
TextForElmo = pd.read_csv("textforEmbedding.csv")
ListOfText = TextForElmo["text"].tolist()
oneString = ''.join(map(str, ListOfText))
text = Text(oneString)
from estnltk.taggers import TokensTagger
tokens_tagger = TokensTagger()
tokens_tagger.tag(text)
from estnltk.taggers import CompoundTokenTagger
compound_token_tagger = CompoundTokenTagger()
compound_token_tagger
compound_token_tagger.tag(text)
from estnltk.taggers import WordTagger
word_tagger = WordTagger()
word_tagger
word_tagger.tag(text)
from estnltk.taggers import SentenceTokenizer
sentence_tokenizer = SentenceTokenizer()
sentence_tokenizer.tag(text)
listOfSentences = list(text.sentences["text"])
myTextDF = pd.DataFrame({"transcript":listOfSentences})
myTextDF['transcript'] = myTextDF['transcript'].astype(str)
myTextDF['transcript'] = myTextDF['transcript'].apply(' '.join)
train_df = myTextDF.sample(frac=0.8, random_state=100)
test_df = myTextDF[~myTextDF.index.isin(train_df.index)]
train_df['transcript'] = train_df['transcript'] + " ."
import os
if not os.path.exists("swb/train"):
    os.makedirs("swb/train")
 
for i in range(0,train_df.shape[0],6):
    text = "\n".join(train_df['transcript'][i:i+6].tolist())
    fp = open("swb/train/"+str(i)+".txt","w",encoding="UTF-8")
    fp.write(text)
    fp.close()

test_df['transcript'] = test_df['transcript'] + " ."
if not os.path.exists("swb/dev"):
    os.makedirs("swb/dev")
 
for i in range(0,test_df.shape[0],6):
    text = "\n".join(test_df['transcript'][i:i+6].tolist())
    fp = open("swb/dev/"+str(i)+".txt","w",encoding="UTF-8")
    fp.write(text)
    fp.close()


from collections import Counter
vocab = " ".join(train_df['transcript'].tolist())
vocab_words = vocab.split(" ")
dictionary = Counter(vocab_words)
sorted_vocab = ["&lt;S&gt;","&lt;/S&gt;","&lt;UNK&gt;"]
sorted_vocab.extend([pair[0] for pair in dictionary.most_common()])
 
sorted_text = "\n".join(sorted_vocab)
fp = open("swb/vocab.txt","w",encoding="UTF-8")
fp.write(sorted_text)
fp.close()



