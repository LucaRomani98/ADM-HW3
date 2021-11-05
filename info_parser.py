import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re

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

    releaseDate = 0

    endDate = 0

    animeNumMembers = 0

    animeScore = 0

    animeUsers = 0

    animeRank = 0

    animePopularity = 0

    animeDescription = ""

    animeRelated = []

    animeCharacters = []

    animeVoices = []

    animeStaff = []




index = 0                               #with this we can restart the download from a given index
for index in range(19119):               #number of downloaded html
    save_path = f"tsv\page_{index//50 + 1}"
    file_name = f"anime_{index+1}.tsv"
    directory = f"html\page_{index//50 + 1}\\article_{index+1}.html"
    completeName = os.path.join(save_path, file_name)
    parse_info(directory, completeName, index)
    index+=1