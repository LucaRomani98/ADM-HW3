import glob
import os
from numpy.core.numeric import NaN
import pandas as pd
import requests
import csv 
import re
import json
import pickle
from collections import defaultdict
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from PyDictionary import PyDictionary
dictionary=PyDictionary()
import heapq


from PyDictionary import PyDictionary
dictionary=PyDictionary()



def preprocess(df):

    x = re.sub('[^a-zA-Z]', '', df)

    lower = str.lower(df).split()


    stop_words = set(stopwords.words('english'))
    no_sw = [w for w in lower if not w in stop_words]

    lemmatizer = WordNetLemmatizer()

    output = [lemmatizer.lemmatize(w) for w in no_sw]

    return(' '.join(output))

def clean(df):
    df['animeDescriptions']=df.apply(lambda x: preprocess(x['animeDescriptions']),axis=1)
    return df




def save_dict(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_vocabulary_and_dictionary():
    nodigit = lambda wordslist: [word for word in wordslist if word.isalpha()]
    f = open("merged.tsv", 'r', encoding="utf8")
    anime = f.readlines()
    new_file = open('vocabulary.tsv', 'w')

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
    return


def open_dictionary():
    with open('dictionary.json') as f:
        dt = json.load(f)  # dictionary
    return(dt)

def open_vocabulary():
    voc = dict()
    with open('vocabulary.tsv') as f:
        for col1, col2 in csv.reader(f, delimiter=','):
            voc[col1] = col2
    return(voc)


def query(q, dataset, dictionary, voc):
    ps = PorterStemmer()
    q = q.strip().split()  # input from user

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
        doc = set(dictionary[term[0]])
        for i in range(1, len(term)):
            doc = doc.intersection(dt[term[i]])
        # take row from books
        return dataset[dataset['Index'].isin(list(doc))][['animeTitle', 'animeDescriptions', 'Url']].head()
    else:
        return "There aren't documents for each word of this query"















