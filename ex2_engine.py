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
nodigit = lambda wordslist : [word for word in wordslist if word.isalpha()]
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
       # print(ds[ds['Index'].isin(list(doc))])
        return ds[ds['Index'].isin(list(doc))][['Index','animeTitle', 'animeDescriptions', 'Url']].head()
    else:
        return "There aren't documents for each word of this query"



#query(input('Insert query : '))



d = dict()
voc = dict()
result = defaultdict(list)
inv_ind = defaultdict(list)

term_idf = defaultdict(float)


with open(path+r'\vocabulary.tsv') as f:
    for col1, col2 in csv.reader(f, delimiter=','):
        voc[col1] = col2

with open(path+r'\dictionary.json') as f:
    dt = json.load(f) # dictionary
    
with open(path+r'\merged.tsv', encoding="utf-8") as f:
    for row in csv.reader(f, delimiter='\t'):
        if len(row) == 17:
            d[row[0]] = row[11]

for doc_id in ds:
    
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+") 
    text_tokens = nodigit(tokenizer.tokenize(str(ds[doc_id])))
    #text_tokens = re.sub(r'\text_tokens', '', string(text_tokens))
    tokens_without_sw = [ps.stem(w.lower()) for w in text_tokens if not w in stopwords.words()]
    
    plotLength = len(tokens_without_sw)
    count = Counter(tokens_without_sw)
    
    for word in count:
        freq = count[word]
        try:
            term_id = str(voc[word])
            idf = 1.0 + math.log( float(len(ds)) / len( dt[term_id] ) )
            tf = freq / plotLength
            tfIdf = tf * idf
            
            heapq.heappush(result[term_id], (tfIdf, doc_id))
            term_idf[term_id] = idf

        except:
            pass


for term, tup_list in result.items():
    for tup in tup_list:
        if(tup[0] != '' and tup[1] != ''):
            inv_ind[term].append( (int(tup[1]), tup[0]) )
        else:
            inv_ind[term].append( int())


with open(path+r"\inverted_index.json", "w") as outfile: 
    json.dump(result, outfile, indent = 4)

with open(path+r"\term_idf.json", "w") as outfile: 
    json.dump(term_idf, outfile, indent = 4)


inv_ind = defaultdict(dict)
dot = lambda x, y : sum(xi*yi for xi, yi in zip(x, y))
square = lambda x : [v**2 for v in x]
det = lambda x : math.sqrt(sum(square(x)))

with open(path+r'\term_idf.json') as f:
    term_idf = json.load(f)

with open(path+r'\inverted_index.json') as f:
    inverted = json.load(f)
    
for term in inverted:
    for t in inverted[term]:
        inv_ind[term][t[0]] = t[1]


a_file = open(path+r"\test.json", "w")
json.dump(inv_ind, a_file)

#print(inv_ind)
def similarity(q):
    ps = PorterStemmer()
    # execute query
    err = "There aren't documents for each word of this query"
    q_result = query(q)
    if not isinstance(q_result, str):
        q = q.strip().split() # input from user
        q = [ps.stem(w).lower() for w in q]
        # create a list of ifidf of terms
        term_tfidf = list()
        tf = 1/len(q)
        for w in q:
            term_tfidf += [term_idf[voc[w]]*tf]
        # create a list of ifidf of document
        doc_tfidf = defaultdict(list)
        for d_id in q_result['Index']:
            d_id = int(d_id-1)
            for w in q:
                for key, value in inv_ind[voc[w]].items():
                    if int(value) == int(d_id):
                         doc_tfidf[d_id].append(float(key))
        #compare value and calculate similarity
        cos_sim = list()
        det_q = det(term_tfidf)
        for doc in q_result['Index']:
            prod = dot(doc_tfidf[doc], term_tfidf)
            det_doc = det(doc_tfidf[doc])
            cos_sim += [(prod / (det_q * det_doc))]
        q_result['similarity'] = cos_sim
        return q_result
        #print(q_result.sort_values(by=['similarity', 'Index'], ascending=False)[['animeTitle', 'AnimeDescriptions', 'Url', 'similarity']].head())
        #return q_result.sort_values(by=['similarity', 'Index'], ascending=False)[['animeTitle', 'AnimeDescriptions', 'Url', 'similarity']].head()
    else:
        return err

     
similarity(input('Insert query: '))
