import glob
import os
import pandas as pd
import re
import json
import nltk
import re
import heapq
import math
import string
import random
import time

#merge all the .tsvs on each of the 383 folders
path = r"C:\Users\matte\Documents\II Year\ADM\Homework3_ADM\html-20211105T160926Z-001"
workpath = path + r"\tsv\page_"
lista = []
for i in range(383):
    if i == 285 or i == 292:
        i+=1
    all_files = glob.glob(os.path.join((workpath+str(i+1)), "*.tsv"))
    df_from_each_file = (pd.read_csv(csvfiles, sep = ',', index_col=None) for csvfiles in all_files)
    df_merged = pd.concat(df_from_each_file)
    lista.append(df_merged)

df = pd.DataFrame(lista)
df.drop_duplicates()
df.to_csv(path+"\merged.tsv", header=['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore','animeUsers','animeRank','animePopularity','animeDescriptions','animeRelated','animeCharacters','animeVoices','animeStaff','Url'])
