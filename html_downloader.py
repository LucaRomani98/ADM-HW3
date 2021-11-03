import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os


#for page_index in range(1,400,1):      #only need to run 1 time, comment this block after first run
#    path = f"html\page_{page_index}"
#    os.mkdir(path)

def get_html(url, path, index):
    req = requests.get(url)
    if(req.status_code != 200):    #if site stops you from connecting the script will stop, so that you can restart from the last index
        raise Exception(f"Site closed connection. Current index was {index}")
    with open(path, 'w', encoding = 'utf-8') as file:
        file.write(req.text)

with open('links.txt', 'r', encoding = 'utf-8') as f:
    lines = f.readlines()

index = 4474  #with this we can restart the download from a given index
while lines[index]:
    save_path = f"html\page_{index//50 + 1}"
    file_name = f"article_{index+1}.html"
    completeName = os.path.join(save_path, file_name)
    get_html(lines[index], completeName, index)
    index+=1




