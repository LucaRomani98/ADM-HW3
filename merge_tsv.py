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
