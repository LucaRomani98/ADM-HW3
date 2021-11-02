import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

anime = []

f = open('links.txt','w')

for page in tqdm(range(0, 10)):
    url = 'https://myanimelist.net/topanime.php?limit=' + str(page * 50)
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup.find_all('tr'):
        links = tag.find_all('a')
        for link in links:        
            if type(link.get('id')) == str and len(link.contents[0]) > 1:
                anime.append((link.contents[0], str(link.get('href'))) )
                
for el in anime:
    f.write(el[1].text.encode("utf-8")+'\n')