import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os


#for page_index in range(1,400,1):      #only need to run 1 time, comment this block after first run
#    path = f"html\page_{page_index}"
#    os.mkdir(path)

def get_html(url, path):
    req = requests.get(url)
    with open(path, 'w', encoding = 'utf-8') as file:
        file.write(req.text)

with open('links.txt', 'r', encoding = 'utf-8') as f:
    lines = f.readlines()

index = 926   #with this we can restart the download from a given index
while lines[index]:
    save_path = f"html\page_{index//50 + 1}"
    file_name = f"article_{index+1}.html"
    completeName = os.path.join(save_path, file_name)
    get_html(lines[index], completeName)
    index+=1


#index = 887
