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


voc = dict()
d = dict()
result = defaultdict(list)
inv_ind = defaultdict(list)
term_idf = defaultdict(float)

dot = lambda x, y : sum(xi*yi for xi, yi in zip(x, y))
square = lambda x : [v**2 for v in x]
det = lambda x : math.sqrt(sum(square(x)))

with open(path+r'\vocabulary.tsv') as f:
    for col1, col2 in csv.reader(f, delimiter=','):
        voc[col1] = col2

with open(path+'\dictionary.json') as f:
    dt = json.load(f) # dictionary

with open(path+r'\merged.tsv', encoding="utf-8") as f:
    for row in csv.reader(f, delimiter='\t'):
        if len(row) == 17:
            d[row[0]] = row[11]


nodigit = lambda wordslist : [word for word in wordslist if word.isalpha()]


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
        print( q_result.sort_values(by=['similarity'], ascending = False))
        #print(q_result.sort_values(by=['similarity', 'Index'], ascending=False)[['animeTitle', 'AnimeDescriptions', 'Url', 'similarity']].head())
        #return q_result.sort_values(by=['similarity', 'Index'], ascending=False)[['animeTitle', 'AnimeDescriptions', 'Url', 'similarity']].head()
    else:
        return err



#normalize dates
def normDate(x):
    r = list()
    for e in x.fillna(''):
        v = e.replace('th', ' ').replace('nd', ' ').replace('st', ' ').replace('rd', ' ')\
                       .translate(str.maketrans('', '', string.punctuation)).split()
        if len(v) <= 3 or len(v) != 0:
            r.append(v)
    return r

def truncate(n):
    n = str(n).replace('',' ').split()
    n.reverse()
    for i in range(1, len(n)):
        v = int(n[i])
        if int(n[i-1]) >= 5:
            n[i] = str(v+1)
    n.reverse()
    return int(n[0] + '0'*(len(n)-1))

normString = lambda x : [i.translate(str.maketrans('', '', string.punctuation)).lower().split() if len(i) > 0 else None for i in x.fillna('')  ]
normFloat = lambda x : [round(float(str(i).replace(',',''))) if i == i else 0 for i in x]
normInt = lambda x : [int(truncate(i.replace(',',''))) if i == i else 0 for i in x]
normEpisode = lambda x : [int(truncate(i.replace(',',''))) if i == i and i.isnumeric() else 0 for i in x]
ds = pd.read_csv(path+'\merged.tsv', header = None, sep='\t')
ds.rename(columns={0: 'Index', 1:'animeTitle', 2:'animeType', 3:'animeNumEpisode', 4:'releaseDate', 5:'endDate', 6:'animeNumMembers', 7:'animeScore', 8:'animeUsers', 9:'animeRank', 10:'animePopularity', 11:'animeDescriptions', 12:'animeRelated', 13:'animeCharacters', 14:'animeVoices', 15:'animeStaff', 16:'Url'}, inplace=True)

n_ds = pd.DataFrame(ds['Index'])
n_ds['animeTitle'] = normString(ds['animeTitle'])
n_ds['animeType'] = normString(ds['animeType'])
n_ds['animeNumEpisode'] = normEpisode(ds['animeNumEpisode'])
n_ds['releaseDate'] = normDate(ds['releaseDate'])
n_ds['endDate'] = normDate(ds['endDate'])
n_ds['animeScore'] = normFloat(ds['animeScore'])
n_ds['animeUsers'] = normInt(ds['animeUsers'])
n_ds['animeRank'] = normInt(ds['animeRank'])
n_ds['animePopularity'] = normInt(ds['animePopularity'])
n_ds['animeDescriptions'] = ds['animeDescriptions']
n_ds['animeRelated'] = normString(ds['animeRelated'])
n_ds['animeCharacters'] = normString(ds['animeCharacters'])
n_ds['animeVoices'] = normString(ds['animeVoices'])
n_ds['animeStaff'] = normString(ds['animeStaff'])
n_ds['Url'] = ds['Url']

# title -> score += (1/len(t))*2
# animetype ->  score += (1/len(ty))*1.5
# episode -> score += 0.5
# released -> score += (1/len(d1))
# endD -> score += (1/len(d2))
# score -> score += 0.5
# users -> score += 0.5
# rank -> score += 0.5
# popularity -> score += 0.5
# descriptions -> already scored
# related ->
# characters -> score += (1/len(c))*1.5
# voices ->  score += (1/len(v))
# staff -> score += (1/len(a))*2



def search(q):
    # execute query
    err = "There aren't documents for each word of this query"
    qs = re.sub('\d', '', q.translate(str.maketrans('', '', string.punctuation)).lower())
    q_result = similarity(qs)
    if not isinstance(q_result, str):
        q = q.strip().split()
        q = [w.lower() for w in q]
        # power up of the score
        doc_score = []
        for doc_id in q_result['Index']:
            score = q_result[q_result['Index'] == doc_id]['similarity'].to_list()[0]
            # calculate score
            for w in q:
                t = n_ds[n_ds['Index']==doc_id]['animeTitle'].to_list()[0]
                ty = n_ds[n_ds['Index']==doc_id]['animeType'].to_list()[0]
                a = n_ds[n_ds['Index']==doc_id]['animeStaff'].to_list()[0]
                c = n_ds[n_ds['Index']==doc_id]['animeCharacters'].to_list()[0]
                v = n_ds[n_ds['Index']==doc_id]['animeVoices'].to_list()[0]
                d1 = n_ds[n_ds['Index']==doc_id]['releaseDate'].to_list()[0]
                d2 = n_ds[n_ds['Index']==doc_id]['endDate'].to_list()[0]
                r = n_ds[n_ds['Index']==doc_id]['animeRelated'].to_list()[0]
                if t != None and w in t:
                    score += (1/len(t))*2
                if ty != None and w in ty:
                    score += (1/len(ty))*1.5
                if a != None and w in a:
                    score += (1/len(a))*2
                if c != None and w in c:
                    score += (1/len(c))*1.5
                if v != None and w in v:
                    score += (1/len(v))
                if d1 != None and w in d1:
                    score += (1/len(d1))
                if d2 != None and w in d2:
                    score += (1/len(d2))
                if r != None and w in r:
                    score += (1/len(r))
                if w.isnumeric():
                    if truncate(w) == n_ds[n_ds['Index']==doc_id]['animeRank'].to_list()[0]:
                        score += 0.5
                    if truncate(w) == n_ds[n_ds['Index']==doc_id]['animeNumEpisode'].to_list()[0]:
                        score += 0.5
                    if truncate(w) == n_ds[n_ds['Index']==doc_id]['animeUsers'].to_list()[0]:
                        score += 0.5
                    if round(float(w)) == n_ds[n_ds['Index']==doc_id]['animeScore'].to_list()[0]:
                        score += 0.5
                    if round(float(w)) == n_ds[n_ds['Index']==doc_id]['animePopularity'].to_list()[0]:
                        score += 0.5
            heapq.heappush(doc_score, (score, doc_id))
        order_doc_id = [i[1] for i in doc_score]
        order_score = [i[0] for i in doc_score]
        r = pd.DataFrame(q_result[q_result['Index']==order_doc_id[0]][['Index', 'animeTitle', 'animeDescriptions', 'Url']])
        for d_id in range(1, len(order_doc_id)):
            r = r.append(q_result[q_result['Index']==order_doc_id[d_id]][['Index', 'animeTitle', 'animeDescriptions', 'Url']])
        r['score'] = order_score
        return r.sort_values(by=['score', 'Index'], ascending=False)[['Index', 'animeTitle', 'animeDescriptions', 'Url', 'score']].head()
    else:
        return err
