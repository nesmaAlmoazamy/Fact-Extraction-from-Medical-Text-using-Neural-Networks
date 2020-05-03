import csv
import numpy as np
import pandas as pd

DS = pd.read_csv("ObjectSubset150SentenceLength.csv")
DS['tag'].fillna('text', inplace=True)
Chars_DS = pd.DataFrame(columns=["text_ID","char","tag"])
for i in range(len(DS)):
    wordList =  list(DS["word"][i])
    for j in range(len(wordList)):
        Chars_DS = Chars_DS.append({'text_ID': DS["text_ID"][i] , 'char': wordList[j], 'tag': DS["tag"][i]},ignore_index=True)

Chars_DS.to_csv("CharsDS.csv",index=False)