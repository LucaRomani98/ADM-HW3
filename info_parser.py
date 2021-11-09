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
        page_soup = BeautifulSoup(fp, "html.parser")

    animeTitle = str(page_soup.find('meta', {'property': 'og:title'}).get('content'))
    
    animeType = ""
    for t in page_soup.find_all('div', class_ = 'spaceit_pad'):
        type_ = t.text.split()
        if type_[0] == "Type:":
            animeType = str(type_[1])

    animeNumEpisode = 0
    for t in page_soup.find_all('div', class_ ='spaceit_paid'):
        ep = t.text.split()
        if ep[0] == "Episodes:":
            try:
                animeNumEpisode = int(ep[1])
            except ValueError:
                animeNumEpisode = ''

    releaseDate = ''
    endDate = ''
    for i in page_soup.find_all('div', class_ = 'spaceit_paid'):
        date_content = i.text.split()
        if date_content[0] == "Aired":
            date = date_content[1:]
            if "Not available" in " ".join(date) or "Not Available" in " ".join(date):
                releaseDate, endDate = '', ''
            dates_l = []
            for sr in date:
                t = re.findall(r'[a-zA-Z]{0,3}[0-9]{0,2}[0-9]{0,4}', sr)
                dates_l.append(t[0])
            dates_l = list(filter(None, dates_l))
            f_l = dates_l
            s_l = []
            if 'to' in dates_l:
                i = dates_l.index('to')
                f_l = dates_l[i]
                s_l = dates_l[i+1:]
            f_counter = len(f_l)
            s_counter = len(s_l)
            if f_counter == 3:
                releaseDate = " ".join(f_l)
                releaseDate = dt.strptime(releaseDate, '%b %d %Y').date()
            elif f_counter == 2:
                releaseDate = " ".join(f_l)
                releaseDate = dt.strptime(releaseDate, '%b %Y').date()
            elif f_counter == 1:
                releaseDate = dt.strptime(f_l[0], '%Y').date()
            if s_counter == 3:
                endDate = " ".join(s_l)
                endDate = dt.strptime(endDate, '%b %d %Y').date()
            elif s_counter == 2:
                endDate = " ".join(s_l)
                endDate = dt.strptime(endDate, '%b %Y').date()
            elif s_counter == 1:
                endDate = dt.strptime(s_l[0], '%Y').date()
    try:
        animeNumMembers = int(''.join(re.findall(r'\d+', page_soup.find('span', class_ = "numbers members").text)))
    except ValueError:
        animeNumMembers = ''

    try:
        animeScore = float(page_soup.find('div', class_ = "fl-l score").find('div').text)
    except ValueError:
        animeScore = ''

    try:
        animeUsers = int(''.join(re.findall(r'\d+', page_soup.find('div', class_ = "fl-l score").get('data-user'))))
    except ValueError:
        animeUsers = ''

    try:
        animeRank = int(re.findall(r'\d+', page_soup.find('span', class_ = 'numbers ranked').text)[0])
    except IndexError:
        animeRank = ''

    try:
        animePopularity = int(re.findall(r'\d+', page_soup.find('span', class_ = 'numbers popularity').text)[0])
    except ValueError:
        animePopularity = ''

    animeDescription = ""
    synopsis = page_soup.find('p', intemprop = "description")
    try:
        if "No synopsis" not in synopsis.text:
            if synopsis.find('span'):
                s = synopsis.find('span')
                animeDescription = synopsis.text.replace(s.text, '')
            else :
                animeDescription = synopsis.text
        else:
            animeDescription = ''
    except:
        animeDescription = ''
    
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

    animeCharacters = []
    if len(page_soup.find_all('h3', class_ = "h3_characters_voice_actors")) == 0:
        animeCharacters = ''
    for i in page_soup.find_all('h3', class_ = "h3_characters_voice_actors"):
        animeCharacters.append(i.text)
    animeCharacters = list(dict.fromkeys(animeCharacters))

    animeVoices = []
    if len(page_soup.find_all('td', class_ = "va-t ar pl4 pr4")) == 0:
        animeVoices = ''
    for i in page_soup.find_all('td', class_ = "va-t ar pl4 pr4"):
        animeVoices.append(i.find('a').text)
    animeVoices = list(dict.fromkeys(animeVoices))

    animeStaff = []
    staff = page_soup.find_all('div', class_ = 'detail-characters-list clearfix')
    if len(staff) == 1:
        staff = staff[0]
    elif len(staff) == 2:
        staff = staff[1]
    else:
        animeStaff = ''
        return
    staff_s = []
    for i in staff.find_all('a'):
        if ' '.join(i.text.split()) != '':
            staff_s.append(' '.join(i.text.split()))
    staff_t = []
    for i in staff.find_all('small'):
        staff_t.append(i.text)
    for i,j in zip(staff_s, staff_t):
        animeStaff.append([i, j])


    URL = page_soup.find_all('link')[0]['href']

    with open(tsv_dir, encoding = 'utd-8', 'w') as tsv:
        tsv.write(headers()+'\n \n')
        tsv.write('{}')



def headers():
    return "animeTitle\t animeType\t animeNumEpisode\t releaseDate\t endDate\t animeNumMembers\t animeScore\t animeUsers\t animeRank\t animePopularity\t animeDescriptions\t animeRelated\t animeCharacters\t animeVoices\t animeStaff\t URL"




index = 0                               #with this we can restart the download from a given index
for index in range(19119):               #number of downloaded html
    save_path = f"tsv\page_{index//50 + 1}"
    file_name = f"anime_{index+1}.tsv"
    directory = f"html\page_{index//50 + 1}\\article_{index+1}.html"
    completeName = os.path.join(save_path, file_name)
    parse_info(directory, completeName, index)
    index+=1
