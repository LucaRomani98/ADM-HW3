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
import types

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
path = r"C:\Users\matte\Documents\II Year\ADM\Homework3_ADM\html-20211105T160926Z-001"


'''
As per our homework's requirements, it is time to create some auxiliary files:

* A python dictionary, called "vocabulary", that maps every word appearing in every Plot of each book to a unique "ID"
* A "json" file, called "dictionary", that contains, for each of the IDs contained in "vocabulary", a list of all the anime where that term appears at least once in the Synopsis.
Note that we are making use of the RegexpTokenizer and the PorterStemmer methods from the ntlk library to stem and tokenize each word in the Plots.

'''

def handle():
    #print('Defining vocabularies')
    voc = dict()
    ps = PorterStemmer()
    vocabulary = dict()
    diz = defaultdict(set)
    if(os.path.exists(path+r'\vocabulary.tsv') and os.path.exists(path+r'\dictionary.json')):
        #print('Loading existing vocabularies')
        with open(path+r'\vocabulary.tsv') as f:
            for col1, col2 in csv.reader(f, delimiter='\t'):
                voc[col1] = col2
        #print('loading dictionary')
        with open(path+r'\dictionary.json') as f:
            dt = json.load(f) # dictionary
        return voc, dt
    else:
        #print('Creating vocabularies')
        nodigit = lambda wordslist : [word for word in wordslist if word.isalpha()]
        f = open(path+"\merged.tsv", 'r', encoding="utf8")
        anime = f.readlines()
        new_file = open(path+r'\vocabulary.tsv', 'w')
        term_id = 1
        document_id = 1
        print('started creation vocabulary')
        index = 0
        for i in anime[1:]:
            print('Completed', round((index/len(anime)*100), 2) , '%', end = '\r')
            try:
                tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
                text_tokens = nodigit(tokenizer.tokenize(i.split('\t')[11]))
                tokens_without_sw = {word for word in text_tokens if not word in stopwords.words()}
                for word in tokens_without_sw:
                    w = ps.stem(word.lower())
                    if w not in vocabulary:
                        vocabulary[w] = term_id
                        diz[term_id].add(document_id)
                        new_file.write(w + "\t" + str(term_id) + '\n')
                        term_id += 1
                    else:
                        diz[vocabulary[w]].add(document_id)
                document_id += 1
            except IndexError:
                pass
            index = index+1
        #print('saving vocabulary')
        with open(path+r'\vocabulary.tsv') as f:
            for col1, col2 in csv.reader(f, delimiter='\t'):
                voc[col1] = col2
        #print('Ok')
        #print('saving dictionary')
        with open(path+r'\dictionary.json', "w") as outfile: 
            json.dump(dict(zip(diz.keys(), map(list, diz.values()))), outfile, indent = 4)
        #print('Ok')
        new_file.close()
        f.close()
        

        with open(path+r'\dictionary.json') as f:
            dt = json.load(f) # dictionary
        return voc, dt
        

#defining first query

def query(q):
    ds = pd.read_csv(path+'\merged.tsv', sep='\t')
    ds.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    #print('First query started, checking vocabulary and dictionary')
    voc, dt = handle()
    #print('Ok')
    #print('Loading DataSet')
    ps = PorterStemmer()
    q = q.strip().split() # input from user
    q = [ps.stem(w).lower() for w in q]
    # elaborate query
    # - > take term_id(s)
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
        #print('Checking result')
        #print(ds[ds['Index'].isin(list(doc))][['Index','animeTitle', 'animeDescriptions', 'Url']].head())
        return ds[ds['Index'].isin(list(doc))][['Index','animeTitle', 'animeDescriptions', 'Url']]
    else:
        return "There aren't documents for each word of this query"


def handle_2():
    #print('Defining second fase dictionaries')
    nodigit = lambda wordslist : [word for word in wordslist if word.isalpha()]
    ds = dict()
    result = defaultdict(list)
    inv_ind = defaultdict(list)
    term_idf = defaultdict(float)    
    if(os.path.exists(path+r'\inverted_index.json') and os.path.exists(path+r'\term_idf.json')):
        #print('Loading term_idf.json')
        with open(path+r'\term_idf.json') as f:
            term_idf = json.load(f)
        #print('loading inverted_index.json')
        with open(path+r'\inverted_index.json') as f:
            inverted = json.load(f)
        for term in inverted:
            for tup in inverted[term]:
                inv_ind[term].append( (int(tup[1]), tup[0]))
        

        return term_idf, inv_ind
    else:
        #print('creating inverted index of : ')
        with open(path+r'\merged.tsv', encoding="utf-8") as f:
            for row in csv.reader(f, delimiter='\t'):
                if len(row) == 17:
                    ds[row[0]] = row[11]
        voc, dt = handle()
        index = 0
        #print('1/2')
        for doc_id in ds:
            print('completed: ', round((index/len(ds)*100), 2) , '%', end = '\r')
            ps = PorterStemmer()
            tokenizer = RegexpTokenizer(r"[a-zA-Z]+") 
            text_tokens = nodigit(tokenizer.tokenize(str(ds[doc_id])))
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
            index = index + 1
            
        
        index = 0
        #print('2/2')
        for term, tup_list in result.items():
            print('completed: ', round((index/len(result)*100), 2) , '%', end = '\r')
            for tup in tup_list:
                print(tup)
                if(tup[0] != '' and tup[1] != '' ):
                    inv_ind[term].append( (int(tup[1]), tup[0]))
                else:
                    inv_ind[term].append( int())
            index = index + 1
        #print('Saving inverted index')
        with open(path+r'\inverted_index.json', "w") as outfile: 
            json.dump(result, outfile, indent = 4)
        #print('Ok')
        #print('Saving term inverted document frequency')
        with open(path+r'\term_idf.json', "w") as outfile: 
            json.dump(term_idf, outfile, indent = 4)
        #print('Ok')

        with open(path+r'\term_idf.json') as f:
            term_idf = json.load(f)
        
        with open(path+r'\inverted_index.json') as f:
            inverted = json.load(f)
        for term in inverted:
            for tup in inverted[term]:
                inv_ind[term].append( (int(tup[1]), tup[0]))
        
        return term_idf, inv_ind


#print('test default query : saiyan race')
#query("saiyan race")

#print('final similarity query dafinition')
def similarity(q):
    print('defining final variables')
    #inv_ind = defaultdict(dict)
    dot = lambda x, y : sum(xi*yi for xi, yi in zip(x, y))
    square = lambda x : [v**2 for v in x]
    det = lambda x : math.sqrt(sum(square(x)))
    voc, dt = handle()
    #print('started querying')
    term_idf, inv_ind = handle_2()
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
            d_id = int (d_id)
            for w in q:
                for key, value in inv_ind[voc[w]]:
                    #print(key, d_id)
                    if int(key) == int(d_id):
                        doc_tfidf[d_id].append(float(value))
                        #print(doc_tfidf[d_id])
        #compare value and calculate similarity
        cos_sim = list()
        det_q = det(term_tfidf)
        for doc in q_result['Index']:
            doc = doc -1
            #print('procut between ',doc_tfidf[doc], 'and ', term_tfidf, ' is equal to')
            prod = dot(doc_tfidf[doc], term_tfidf)
            det_doc = det(doc_tfidf[doc])
            #print('determinant of ', doc_tfidf[doc], 'is equal to', det_doc)
            #print('operation =', prod, '/(',det_q,'*',det_doc,')' )
            cos_sim += [(prod / (det_q * det_doc))]
            #print('result cosine similarity:', cos_sim)
        q_result['similarity'] = cos_sim
        #print( q_result.sort_values(by=['similarity'], ascending = False))
        return q_result.sort_values(by=['similarity'], ascending = False)
    else:
        return err

#print('testing default query: saiyan race')
#similarity("saiyan race")


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
    if (n is float):
        n = int(n)
    if( ':' not in n and n is not None):
        n = str(n).replace('',' ').split()
        n.reverse()
        for i in range(1, len(n)):
            v = int(n[i])
            if int(n[i-1]) >= 5:
                n[i] = str(v+1)
                n.reverse()
                return int(n[0] + '0'*(len(n)-1))
    else:
        return 0

def norm_df():
    normString = lambda x : [i.translate(str.maketrans('', '', string.punctuation)).lower().split() if len(i) > 0 else None for i in x.fillna('')  ]
    normFloat = lambda x : [round(float(str(i).replace(',',''))) if i == i and i != 'Unnamed: 6' else 0 for i in x]
    normInt = lambda x : [int(truncate(i.replace(',',''))) if i == i else 0 for i in x]
    normEpisode = lambda x : [int(truncate(i.replace(',',''))) if i == i and i.isnumeric() and i is not None else 0 for i in x]

    ds = pd.read_csv(path+'\merged.tsv', sep='\t')
    ds.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    n_ds = ds
    n_ds.dropna()
    n_ds['animeTitle'] = normString(ds['animeTitle'])
    n_ds['animeType'] = normString(ds['animeType'])
    for ep in n_ds['animeNumEpisode']:
        try:
            n_ds['animeNumEpisode'== ep] = normEpisode(ds['animeNumEpisode'==ep])
        except:
            n_ds['animeNumEpisode'== ep] = 0
    n_ds['releaseDate'] = normDate(ds['releaseDate'])
    n_ds['endDate'] = normDate(ds['endDate'])
    n_ds['animeScore'] = normFloat(ds['animeScore'])
    for us in n_ds['animeUsers']:
        try:
            n_ds['animeUsers'== us] = normInt(ds['animeUsers'==us])
        except:
            n_ds['animeUsers'== us] = 0

    for ra in n_ds['animeRank']:
        try:
            n_ds['animeRank'== ra] = normInt(ds['animeRank'==ra])
        except:
            n_ds['animeRank'== ra] = 0
    for po in n_ds['animePopularity']:
        try:
            n_ds['animePopularity'== po] = normInt(ds['animePopularity'==po])
        except:
            n_ds['animePopularity'== po] = 0
    n_ds['animeDescriptions'] = ds['animeDescriptions']
    n_ds['animeRelated'] = normString(ds['animeRelated'])
    n_ds['animeCharacters'] = normString(ds['animeCharacters'])
    n_ds['animeVoices'] = normString(ds['animeVoices'])
    n_ds['animeStaff'] = normString(ds['animeStaff'])
    n_ds['Url'] = ds['Url']
    return n_ds

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
    n_ds = norm_df()
    #print(n_ds)
    if not isinstance(q_result, str):
        q = q.strip().split()
        q = [w.lower() for w in q]
        # power up of the score
        doc_score = []
        for doc_id in q_result:
            score = list(q_result[q_result['Index'] == doc_id]['similarity'])
            print(score)
            # calculate score
            for w in q:
                t = list(n_ds[n_ds['Index']==doc_id]['animeTitle'])
                ty = list(n_ds[n_ds['Index']==doc_id]['animeType'])
                a = list(n_ds[n_ds['Index']==doc_id]['animeStaff'])
                c = list(n_ds[n_ds['Index']==doc_id]['animeCharacters'])
                v = list(n_ds[n_ds['Index']==doc_id]['animeVoices'])
                d1 = list(n_ds[n_ds['Index']==doc_id]['releaseDate'])
                d2 = list(n_ds[n_ds['Index']==doc_id]['endDate'])
                r = list(n_ds[n_ds['Index']==doc_id]['animeRelated'])
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
        print(r[['Index', 'animeTitle', 'animeDescriptions', 'Url', 'score']].head())
        return r[['Index', 'animeTitle', 'animeDescriptions', 'Url', 'score']].head()
    else:
        return err


search("saiyan race")
