import glob
import os
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


#nltk.download('wordnet')

path = r"C:\Users\matte\Documents\II Year\ADM\Homework3_ADM\html-20211105T160926Z-001"
workpath = path + r"\tsv\page_"

#create search engine

df = pd.read_csv(path+'\merged.tsv')
df.head()


def preprocess(data):
    
    #removing punctuation (not removing numbers because title can be like "1984")
    x=re.sub(r'[^a-zA-Z]*', ' ',str(data)) 
    
    #lowering words
    lower=str.lower(x).split() 
    words=set(stopwords.words('english'))
    
    #removing stopwords
    no_stopwords=[w for w in lower if not w in words]  
    lmtzr = WordNetLemmatizer()
    
    #stemming
    cleaned=[lmtzr.lemmatize(w) for w in no_stopwords] 
    
    
    return (" ".join( cleaned ))

#clean
def cleaning(daf):
    daf['animeDescriptions']=daf.apply(lambda x: preprocess(x['animeDescriptions']),axis=1)
    return daf


#clean synopsis here
df = cleaning(df)


#saving dict function
def save_dict(obj, name ):
    with open(path  + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#load dict function
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

''' 
vocabulary
:key:= word_str
:value:= word_id
'''
#the Dict we wanna build
vocabulary = defaultdict()

def create_vocabulary(data):

    #set in which i collect all the terms
    term_set = set()
    syn = list(data['animeDescriptions'])
    for elem in syn:
        try:
            term_set =term_set.union(set(elem.split()))
        except:
            pass

    #convert the set in list to enumerate
    term_list = list(term_set)

    for i, elem in enumerate(term_list):
        vocabulary[elem]= i 
    
    save_dict(vocabulary,r'\vocabulary')

#create first vocabulary here
create_vocabulary(df)

''' 
word_doc (Inverted Index)
:key:= word_id
:value:= list of doc_id that cointains that word
'''
def create_dict(text, book_id, word_doc):
    try: 
        word_list = set(text.split())
        for word in word_list:      
            word_doc[vocabulary[word]]+=[book_id]
    except:
        pass

word_doc = defaultdict(list)
def create_word_doc(da):
    da.apply(lambda x: create_dict(x['animeDescriptions'],x['animeTitle'],word_doc), axis = 1)

    save_dict(word_doc,r'\word_doc')

#Here you create the inverted index (reffered to the document)
create_word_doc(df)

'''
rev_vocabulary
:key: word_id
:value:= word_str
'''
rev_vocabolary = {}
def create_rev_vocabulary(voc):
    for key in voc.keys():
        rev_vocabolary[voc[key]]=key
    save_dict(rev_vocabolary,r'\rev_vocabolary')

#here you create a reverse vocabulary
create_rev_vocabulary(vocabulary)

'''
tf_word_doc (Inverted Index)
:key:= word_id of j
:value:= list tuple (doc_id of i ,tf_ij)
'''
    
def tf_idf(id_word,id_doc,f):
    id_doc = re.sub("\*", ' ', id_doc)
    word = rev_vocabolary[id_word]
    d_j = f[f['animeTitle'].str.match(id_doc)]['animeDescriptions'] #document j
    n_ij = d_j.str.count(word) #number of occurrences of term i in document j
    tf_ij = n_ij/(len(d_j)) 
    
    n = len(f) #total number of docs
    N_i = len(word_doc[id_word]) #number of docs that contain the document i  
    
    idf_i = np.log10(n/N_i)
    
    return tf_ij*idf_i

tf_word_doc = defaultdict(list)
def create_tf_word_doc(wd):
    for word in wd.keys():
        for doc in wd[word]:
            tf_word_doc[word] += [(doc, tf_idf(word,doc, df))]
            
    save_dict(tf_word_doc,r'\tf_word_doc')

#Here you create TF-IDF
create_tf_word_doc(word_doc)
'''
word_occ
key: doc_id
values: (word_id, freq_word) freq_word = frequency of word_id in doc
'''
def get_occ(text):
    out = []
    for word in text.split():
        out+= [(vocabulary[word], text.count(word))]
    return out

def create_word_occ(da):
    word_occ = defaultdict(list)
    for i, row in da.iterrows():
        word_occ[i] = get_occ(row['animeDescriptions'])

#here create occurrences vocabulary
create_word_occ(df)

'''
occ_word_doc
:key:= word_id of j
:value:= list tuple (doc_id of i ,occ_ij)
'''
def create_occ_word_doc(wd,rv):
    occ_word_doc = defaultdict(list)
    for word_id in wd.keys():
        word_str = rv[word_id]
        for doc_id in wd[word_id]:
            doc_id = re.sub("\*", ' ', doc_id)
            freq = df[df['animeTitle'].str.match(doc_id)]['animeDescriptions'].str.count(word_str)
            occ_word_doc[word_id] += [(doc_id, freq)]
            
    save_dict(occ_word_doc,r'\occ_word_doc')

#here create document for word occurrences 
create_occ_word_doc(word_doc, rev_vocabolary)

'''
doc_norm
:key: the document_id
:value: the norm of the document
'''
def norm(l):
    return np.linalg.norm(np.array(l))

def create_doc_norm():
    doc_vector = defaultdict(list)
    doc_norm = defaultdict(None)
    for word_val in list(tf_word_doc.values()):
        for tup in word_val:
            doc_id = tup[0]
            tf_idf = tup[1]
            doc_vector[doc_id] += [tf_idf]

    for doc in doc_vector.keys():
        doc_norm[doc] = norm(doc_vector[doc])
    save_dict(doc_norm,r'\doc_norm')

#jere create doc norm
create_doc_norm()

#########################################################
vocabulary = load_obj(path+'/vocabulary')
word_doc = load_obj(path+'/word_doc')
tf_word_doc = load_obj(path+'/tf_word_doc')
doc_norm = load_obj(path+'/doc_norm')
rev_vocabolary = load_obj(path+'/rev_vocabolary')
#######


###################################################################################################################################################################
# FROM THIS POINT THE CODE IS NOT TESTED/EDITED
# WIP
##################################################################################################################################################################
def format_query(q):
    '''
    get in input a str of words
    return a list with stemmed word
    '''
    q = q.split()
    return [preprocess(w) for w in q]
     

def all_equal(l):
    '''
    check if all the documents in the list of tuples (docs, tf_idf) are equal
    '''
    l = [x[0] for x in l]
    return bool(l == [l[0]]*len(l))
    
def update_positions(l,index):
    """[Update position of the index in query function, increase of 1 the index of the doc with the minimum ID]
    Args:
        l ([list]): [list of current tuples (doc, tf_idf)]
        index ([list]): [list of indies for all the values of the queries]
    Returns:
        [list]: [updated index]
    """
    
    #we want to consider the minimum id of the documents, t = list of tuples (doc, tf_idf), t[0] = documents
    min_value= min(l, key = lambda t: t[0])[0]
    
    #update only the index of minimum values 
    for i,elem in enumerate(l):
        if elem[0]==min_value:
            index[i]+=1
            
    return index

def get_prod(l):
    """[Get the doct product between the query and the doc]
    Args:
        l ([list of tuples(doc, tf_idf_word)]): [doc are the same, tf_idf for all the words of the query]
    Returns:
        [list]: [document, value of the dot product between the doc and the words of the query]
    """
    #first element first couple, we know that the documents are the same for all the couples since all_equal ==True
    doc = l[0][0]
    
    #the norm of the doc
    docnorm = doc_norm[doc]
    
    #sum of the tf_idf of the documents 
    comp_sum =  sum([t[1] for t in l]) 
    
    return [doc,comp_sum/docnorm]

    
def query(q,df):
    """[given a query, returns a dictionary key:doc, value: doc product between the doc and the query]
    Args:
        q ([string]): [query in input]
        df ([dataframe]): [dataframe for whome plot we consider the tf_idfs ]
    Returns:
        [type]: [description]
    """
    d_query = defaultdict()
    
    #preprocessing the words of the query
    q = format_query(q)
    
    #list of the ID of the words in the query
    term_id_list = [vocabulary[word] for word in q if word in vocabulary.keys()]
    
    #list of the [(doc, tf_idf)...] for the words in the query
    doc_tf_list = [tf_word_doc[term_id] for term_id in term_id_list]
    
    #sort the id of the Documents
    doc_tf_list.sort(key=lambda x:x[0])
    
    #empty list, will collect the documents that are in the values of every of the words of the query (intersection)
    q_out = []
    #A: Doc1,tf1 , Doc2 .......
    #B: Doc1, .. , Doc3 
    #index for parsing all the documents, increase of one for the minimun document
    index = [0]*len(doc_tf_list)
    while True:
        try:
            #document in position [index] in the values of the dictiorary doc_tf_list
            current = []
            
            #qi = i-th word of the query, docs_of_qi document in the values of tf_word_doc[qi]
            for qi, docs_of_qi  in enumerate(doc_tf_list):
                
                #list of all the couples (document,tf-idf) of the current docs, for every qi
                current += [doc_tf_list[qi][index[qi]]]
                
            index = update_positions(current,index)
            
            #check if all the minimum indicies are equal
            if all_equal(current):
                
                #list of couples doc, score 
                q_out += [get_prod(current)]

        #just for interrupting if index overcome the maximum lenght of one value of the dictonary
        except:
            for couple in q_out:
                
                #dict key = doc, value = score
                d_query[couple[0]] = couple[1]
            return d_query

def df_query(q,df):
    """[returns the df with the similarity score, rows sorted by similarity score]
    Args:
        q ([str]): [query in input]
        df ([dataframe]): []
    Returns:
        [df]: [df with the similarity score, sorted by it]
    """
    d_query = query(q,df)
    df_q = df[df['index'].isin(d_query.keys())]

    #we apply the function score = d_query[doc_index]
    df_q['similarity'] = df_q.apply(lambda x: d_query[x['index']], axis = 1)
    return df_q.sort_values('similarity',ascending = False).reset_index(drop = True)


