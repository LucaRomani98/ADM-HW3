import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime as dt
import os
import re
import csv

#for page_index in range(1,384,1):       #this creates the folders where you store the html
#    path = f"tsv\page_{page_index}"     #only need to run 1 time, comment this block after first run
#    os.mkdir(path)

def parse_info(html_dir, tsv_dir, index):
    with open(html_dir, encoding = 'utf-8') as fp:
        soup = BeautifulSoup(fp, "html.parser")

    animeTitle = soup.find('h1', class_='title-name h1_bold_none').text

    finder = soup.find(text=re.compile('Type:'))
    animeType = finder.parent.parent.text.split()[-1]

    finder = soup.find(text=re.compile('Episodes:'))
    animeNumEpisode = int(finder.parent.parent.text.split()[-1])
    
    finder = soup.find('span', text=re.compile('Aired:').parent.text.split('\n')[2].strip().split(' to ')
    releaseDate = dt.strptime(finder[0], '%b %d %Y')

    if len(finder) == 2:
        endDate = dt.strptime(finder[1], '%b %d %Y')
    else:
        endDate = ''
    
    finder = soup.find(text=re.compile('Members '))
    animeNumMembers = int(finder.parent.parent.text.split()[3].replace(',',''))

    animeScore = float(soup.find('span', itemprop = 'ratingValue').text)

    animeUsers = int(soup.find('span', itemprop = 'ratingCount').text)

    finder = soup.find(text=re.compile('Members '))
    if finder.parent.parent.text.replace('#','').replace('P',' ').split()[1].isnumeric():
        animeRank = int(finder.parent.parent.text.replace('#','').replace('P',' ').split()[1])
    else:
        animeRank = ''

    animePopularity = int(soup.find(text=re.compile('Popularity:')).parent.parent.text.replace('#','').split()[-1])

    animeDescription = soup.find('p', itemprop = 'description').text

    animeRelated = []

    animeCharacters = []

    animeVoices = []

    animeStaff = []

    with open(tsv_dir, encoding = 'utd-8', 'w') as tsv:
        tsv.write(headers()+'\n \n')
        tsv.write('{}')



def headers():
    return "animeTitle\t animeType\t animeNumEpisode\t releaseDate\t endDate\t animeNumMembers\t animeScore\t animeUsers\t animeRank\t animePopularity\t animeDescriptions\t animeRelated\t animeCharacters\t animeVoices\t animeStaff"




index = 0                               #with this we can restart the download from a given index
for index in range(19119):               #number of downloaded html
    save_path = f"tsv\page_{index//50 + 1}"
    file_name = f"anime_{index+1}.tsv"
    directory = f"html\page_{index//50 + 1}\\article_{index+1}.html"
    completeName = os.path.join(save_path, file_name)
    parse_info(directory, completeName, index)
    index+=1