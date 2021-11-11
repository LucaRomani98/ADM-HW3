import glob
import os
from numpy.core.numeric import NaN
import pandas as pd
import nltk
import requests
import csv 
import re
import json
import pickle
from collections import defaultdict
import nltk
import nltk
import re
import heapq
import math
import string
import random
import time
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from PyDictionary import PyDictionary
dictionary=PyDictionary()
import heapq

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
path = r"C:\Users\matte\Documents\II Year\ADM\Homework3_ADM\html-20211105T160926Z-001"
workpath = path + r"\tsv\page_"

'''
As per our homework's requirements, it is time to create some auxiliary files:

* A python dictionary, called "vocabulary", that maps every word appearing in every Plot of each book to a unique "ID"
* A "json" file, called "dictionary", that contains, for each of the IDs contained in "vocabulary", a list of all the anime where that term appears at least once in the Synopsis.
Note that we are making use of the RegexpTokenizer and the PorterStemmer methods from the ntlk library to stem and tokenize each word in the Plots.
'''



nodigit = lambda wordslist : [word for word in wordslist if word.isalpha()]
f = open(path+"\merged.tsv", 'r', encoding="utf8")
anime = f.readlines()
new_file = open(path+r'\vocabulary.tsv', 'w')

ps = PorterStemmer()
term_id = 1
document_id = 1
vocabulary = dict()
diz = defaultdict(set)
for i in anime[1:]:
    try:
        tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
        text_tokens = nodigit(tokenizer.tokenize(i.split('\t')[11]))
        tokens_without_sw = {word for word in text_tokens if not word in stopwords.words()}
        for word in tokens_without_sw:
            w = ps.stem(word.lower())
            if w not in vocabulary:
                vocabulary[w] = term_id
                diz[term_id].add(document_id)
                new_file.write(w + "," + str(term_id) + '\n')
                term_id += 1
            else:
                diz[vocabulary[w]].add(document_id)
        document_id += 1
    except IndexError:
        pass

new_file.close()
f.close()

with open(path+"\dictionary.json", "w") as outfile: 
    json.dump(dict(zip(diz.keys(), map(list, diz.values()))), outfile, indent = 4)


#do the first query

ds = pd.read_csv(path+'\merged.tsv', header = None, sep='\t')
ds.rename(columns={0: 'Index', 1:'animeTitle', 11:'animeDescriptions', 16:'Url'}, inplace=True)
voc = dict()

with open(path+r'\vocabulary.tsv') as f:
    for col1, col2 in csv.reader(f, delimiter=','):
        voc[col1] = col2

with open(path+'\dictionary.json') as f:
    dt = json.load(f) # dictionary

def query(q):
    ps = PorterStemmer()
    q = q.strip().split() # input from user

    q = [ps.stem(w).lower() for w in q]

    # elaborate query
    
    # take term_id(s)
    term = list()
    for w in q:
        try:
            term.append(voc[w])
        except:
            pass
    # matching documents
    if len(term):
        doc = set(dt[term[0]])
        for i in range(1, len(term)):
            doc = doc.intersection(dt[term[i]])
        # take row from books
        print(ds[ds['Index'].isin(list(doc))][['animeTitle', 'animeDescriptions', 'Url']].head())
        return ds[ds['Index'].isin(list(doc))][['animeTitle', 'animeDescriptions', 'Url']].head()
    else:
        return "There aren't documents for each word of this query"



query(input('Insert query : '))
