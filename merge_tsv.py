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
df = pd.DataFrame(columns = ['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore','animeUsers','animeRank','animePopularity','animeDescriptions','animeRelated','animeCharacters','animeVoices','animeStaff','Url'])
lista = []
for i in range(383):
    if i == 285 or i == 292:
        i+=1
    all_files = glob.glob(os.path.join((workpath+str(i+1)), "*.tsv"))
    for file in all_files:
        df.loc[len(df)] = file
print(df.head())
df.to_csv(path+"\merged.tsv")
