import pandas as pd
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle


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