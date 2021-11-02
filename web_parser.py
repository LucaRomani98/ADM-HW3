import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


anime = []

for page in tqdm(range(0, 400)):
    url = 'https://myanimelist.net/topanime.php?limit=' + str(page * 50)
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup.find_all('tr'):
        links = tag.find_all('a')
        for link in links:        
            if type(link.get('id')) == str and len(link.contents[0]) > 1:
                anime.append((link.contents[0], link.get('href')) )
                
with open('links.txt', 'w', encoding = 'utf-8') as f:
    for el in anime:
        f.write(str(el[1])+'\n')

#after the 382th iteration we arrive to the end of the list, so there are less than 20000 anime
