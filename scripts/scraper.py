import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
numbers = np.arange(start=101, stop=5901, step=100)

response_list = list()
for page_number in numbers:
    url = f"https://www.the-numbers.com/movie/budgets/all/{page_number}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    response_list.append(soup)
    time.sleep(1)


rows_list = list()

for soup in response_list:
    rows = soup.find_all('tr')
    for tr in rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        rows_list.append(row)


df = pd.DataFrame(rows_list, 
                  columns=['Index','Release_date', 'Title', 'Production_Budget', 
                            'Domestic_Gross', 'Worldwide_Gross'])

df.dropna(inplace=True)
df.drop('Index', axis=1, inplace=True)

df['Production_Budget'] = df['Production_Budget'].str.replace('$', '')
df['Production_Budget'] = df['Production_Budget'].str.replace(',', '')

df['Domestic_Gross'] = df['Domestic_Gross'].str.replace('$', '')
df['Domestic_Gross'] = df['Domestic_Gross'].str.replace(',', '')

df['Worldwide_Gross'] = df['Worldwide_Gross'].str.replace('$', '')
df['Worldwide_Gross'] = df['Worldwide_Gross'].str.replace(',', '')

df = df.astype({'Production_Budget': 'int64', 'Domestic_Gross': 'int64', 'Worldwide_Gross': 'int64'})

df.to_csv('scraped_budgets.csv', index=False, header=True)