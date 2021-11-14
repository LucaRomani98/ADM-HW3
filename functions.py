import glob
import os
from numpy.core.numeric import NaN
import pandas as pd
import requests
import csv 
import re
import json
from collections import defaultdict
import nltk
import re
import heapq
import math
import string
from datetime import datetime as dt
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
from collections import Counter
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
from tqdm import tqdm

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
path = r"C:\Users\matte\Documents\II Year\ADM\Homework3_ADM\html-20211105T160926Z-001"

def get_html(url, path, index):
    req = requests.get(url)
    if(req.status_code != 200):    #if site stops you from connecting the script will stop, so that you can restart from the last index
        raise Exception(f"Site closed connection. Current index was",index)
    with open(path, 'w', encoding = 'utf-8') as file:
        file.write(req.text)
    
    
    
def download_pages():
    for page_index in range(1,400,1):
        p_p = path +r'\html\page_'+{page_index}
        os.mkdir(p_p)
    with open('links.txt', 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        index = 0  #with this we can restart the download from a given index
        while lines[index]:
            save_path = f"html\page_{index//50 + 1}"
            file_name = f"article_{index+1}.html"
            completeName = os.path.join(save_path, file_name)
            get_html(lines[index], completeName, index)
            index+=1

def parse_info(html_dir, tsv_dir, index):
    row = []
    with open(html_dir, encoding = 'utf-8') as fp:
        page_soup = BeautifulSoup(fp, "html.parser")

    animeTitle = str(page_soup.find('meta', {'property': 'og:title'}).get('content'))
    row.append(animeTitle)
    
    animeType = ""
    for t in page_soup.find_all('div', class_ = 'spaceit_pad'):
        type_ = t.text.split()
        if type_[0] == "Type:":
            animeType = str(type_[1])
    row.append(animeType)

    animeNumEpisode = 0
    for t in page_soup.find_all('div', class_ ='spaceit_pad'):
        ep = t.text.split()
        if ep[0] == "Episodes:":
            try:
                animeNumEpisode = int(ep[1])
            except ValueError:
                animeNumEpisode = ''
    row.append(animeNumEpisode)


    releaseDate = ''
    endDate = ''
    date = page_soup.find_all('div', class_='spaceit_pad')
    for tag in date:
        date_info = tag.text.split()
        if date_info[0] == "Aired:":
            only_date = date_info[1:]
            if not "Not available" in " ".join(only_date) or "Not Available" in " ".join(only_date):
                data = []
                for string in only_date:
                    prova = re.findall(r'[a-zA-Z]{0,3}[0-9]{0,2}[0-9]{0,4}', string)
                    data.append(prova[0])
                data = list(filter(None, data))
                first_date_list = data
                second_date_list = []
                first_date = ''
                second_date = ''
                if 'to' in data:
                    ind = data.index('to')
                    first_date_list = data[:ind]
                    second_date_list = data[ind+1:]

                first_count = len(first_date_list)
                second_count = len(second_date_list)
                if first_count == 3:
                    first_date = " ".join(first_date_list)
                    releaseDate = dt.strptime(first_date, '%b %d %Y').date()
                if first_count == 2:
                    first_date = " ".join(first_date_list)
                    releaseDate = dt.strptime(first_date, '%b %Y').date()
                if first_count == 1:
                    releaseDate = dt.strptime(first_date_list[0], '%Y').date()
                if second_count == 3:
                    second_date = " ".join(second_date_list)
                    endDate = dt.strptime(second_date, '%b %d %Y').date()
                if second_count == 2:
                    second_date = " ".join(second_date_list)
                    endDate = dt.strptime(second_date, '%b %Y').date()
                if second_count == 1:
                    endDate = dt.strptime(second_date_list[0], '%Y').date()
    row.append(releaseDate)
    row.append(endDate)

    try:
        animeNumMembers = int(''.join(re.findall(r'\d+', page_soup.find('span', class_ = "numbers members").text)))
    except ValueError:
        animeNumMembers = ''
    row.append(animeNumMembers)

    try:
        animeScore = float(page_soup.find('div', class_ = "fl-l score").find('div').text)
    except ValueError:
        animeScore = ''
    row.append(animeScore)

    try:
        animeUsers = int(''.join(re.findall(r'\d+', page_soup.find('div', class_ = "fl-l score").get('data-user'))))
    except ValueError:
        animeUsers = ''
    row.append(animeUsers)

    try:
        animeRank = int(re.findall(r'\d+', page_soup.find('span', class_ = 'numbers ranked').text)[0])
    except IndexError:
        animeRank = ''
    row.append(animeRank)

    try:
        animePopularity = int(re.findall(r'\d+', page_soup.find('span', class_ = 'numbers popularity').text)[0])
    except ValueError:
        animePopularity = ''
    row.append(animePopularity)

    animeDescription = ""
    descr = page_soup.find('p', itemprop="description")
    try:
        if "No synopsis" not in descr.text:
            if descr.find('span'):
                span = descr.find('span')
                animeDescription = descr.text.replace(span.text, '')
            else:
                animeDescription = descr.text
        else:
            animeDescription = ''
    except:
        animeDescription = ''
    row.append(animeDescription)
    
    try:
        animeRelated = []
        for i in page_soup.find('table', class_ = "anime_detail_related_anime").find_all('a'):
            if i.text not in animeRelated:
                animeRelated.append(i.text)
        if len(animeRelated ) > 0:
            animeRelated = animeRelated
        else:
            animeRelated = ''
    except AttributeError:
        animeRelated = ''
    row.append(animeRelated)

    animeCharacters = []
    if len(page_soup.find_all('h3', class_ = "h3_characters_voice_actors")) == 0:
        animeCharacters = ''
    for i in page_soup.find_all('h3', class_ = "h3_characters_voice_actors"):
        animeCharacters.append(i.text)
    animeCharacters = list(dict.fromkeys(animeCharacters))
    row.append(animeCharacters)

    animeVoices = []
    if len(page_soup.find_all('td', class_ = "va-t ar pl4 pr4")) == 0:
        animeVoices = ''
    for i in page_soup.find_all('td', class_ = "va-t ar pl4 pr4"):
        animeVoices.append(i.find('a').text)
    animeVoices = list(dict.fromkeys(animeVoices))
    row.append(animeVoices)

    animeStaff = []
    staff = page_soup.find_all('div', class_='detail-characters-list clearfix')
    if len(staff) == 1:
        staff_2 = staff[0]
    elif len(staff) == 2:
        staff_2 = staff[1]
    else:
        animeStaff = ''

    if len(staff) == 1 or len(staff) == 2:
        staff_s = []
        for i in staff_2.find_all('a'):
            if ' '.join(i.text.split()) != '':
                staff_s.append(' '.join(i.text.split()))
        staff_t = []
        for i in staff_2.find_all('small'):
            staff_t.append(i.text)
        for i, j in zip(staff_s, staff_t):
            animeStaff.append([i, j])
        row.append(animeStaff)


    Url = page_soup.find_all('link')[13]['href']
    row.append(Url)

    with open(tsv_dir, 'w', encoding = 'utf-8') as tsv:
        tsv_writer = csv.writer(tsv, delimiter='\t')
        tsv_writer.writerow(row)


    print(index)


def parse_pages():
    for page_index in range(1,384,1):     
        page_d = path+r'\page_'+page_index        
        os.mkdir(page_d)
    index = 0
    while index < 19119:
        save_path = f"tsv/page_"+str(index//50+1)
        file_name = f"/anime_"+str(index+1)+".tsv"
        directory = f"html/page_"+str(index//50+1)+"/article_"+str(index+1)+".html"
        completeName = save_path + file_name
        parse_info(directory, completeName, index)
        index+=1


#merge all the .tsvs on each of the 383 folders
col = ['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore','animeUsers','animeRank','animePopularity','animeDescriptions','animeRelated','animeCharacters','animeVoices','animeStaff','Url']

#merge all the .tsvs on each of the 383 folders
workpath = r"tsv\page_"

lista = []

for i in range(383):
    all_files = glob.glob(os.path.join((workpath+str(i+1)), "*.tsv"))
    for file in all_files:
        #print(list(pd.read_csv(file, sep=',',nrows=1)))
        lista.append(list(pd.read_csv(file, sep='\t',nrows=1)))
        #print(lista) 
        #df.append(pd.read_csv(file, header = None, sep=','))
    print(i)
df = pd.DataFrame(lista, columns = col)
df.to_csv("merged.tsv", sep='\t')


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
    #print('defining final variables')
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
    if not isinstance(q_result, str):
        q = q.strip().split()
        q = [w.lower() for w in q]
        # power up of the score
        doc_score = []
        score = []
        i = 0
        for doc_id in q_result['Index']:
            score.append(q_result[q_result['Index']==doc_id]['similarity'])
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
                    score[i] += (1/len(t))*2
                if ty != None and w in ty:
                    score[i] += (1/len(ty))*1.5
                if a != None and w in a:
                    score[i] += (1/len(a))*2
                if c != None and w in c:
                    score[i] += (1/len(c))*1.5
                if v != None and w in v:
                    score[i] += (1/len(v))
                if d1 != None and w in d1:
                    score[i] += (1/len(d1))
                if d2 != None and w in d2:
                    score[i] += (1/len(d2))
                if r != None and w in r:
                    score[i] += (1/len(r))
                if w.isnumeric():
                    if truncate(w) == list(n_ds[n_ds['Index']==doc_id]['animeRank']):
                        score[i] += 0.5
                    if truncate(w) == list(n_ds[n_ds['Index']==doc_id]['animeNumEpisode']):
                        score[i] += 0.5
                    if truncate(w) == list(n_ds[n_ds['Index']==doc_id]['animeUsers']):
                        score[i] += 0.5
                    if round(float(w)) == list(n_ds[n_ds['Index']==doc_id]['animeScore']):
                        score[i] += 0.5
                    if round(float(w)) == list(n_ds[n_ds['Index']==doc_id]['animePopularity']):
                        score[i] += 0.5
            heapq.heappush(doc_score, (score, doc_id))
            i = i+1
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

#search("saiyan race")
