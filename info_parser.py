from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime as dt
import os
import re
import csv





#for page_index in range(1,384,1):     #this creates the folders where you store the html
#    path = other+f"/page_{page_index}"     #only need to run 1 time, comment this block after first run
#    os.mkdir(path)

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
            if "Not available" in " ".join(only_date) or "Not Available" in " ".join(only_date):
                return ['', '']
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
    row.append(animeStaff)


    Url = page_soup.find_all('link')[13]['href']
    row.append(Url)

    with open(tsv_dir, 'w', encoding = 'utf-8') as tsv:
        tsv_writer = csv.writer(tsv, delimiter=',')
        #tsv_writer.writerow(['animeTitle','animeType','animeNumEpisode','releaseDate','endDate','animeNumMembers','animeScore','animeUsers','animeRank','animePopularity','animeDescriptions','animeRelated','animeCharacters','animeVoices','animeStaff','Url'])
        tsv_writer.writerow(row)


    print(index)



index = 0                             
while index < 19119:
    save_path = f"tsv/page_"+str(index//50+1)
    file_name = f"/anime_"+str(index+1)+".tsv"
    directory = f"html/page_"+str(index//50+1)+"/article_"+str(index+1)+".html"
    completeName = save_path + file_name
    parse_info(directory, completeName, index)
    index+=1
