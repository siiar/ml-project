import sys
import logging
import time
import requests
import pandas as pd
import poster_download
logging.basicConfig(level=logging.INFO)
import os


CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
IMG_PATTERN = 'http://api.themoviedb.org/3/movie/{imdbid}/images?api_key={key}' 
KEY = 'AAAAAAAAAAAAAAA'

NAME = 'poster_id'
with open(NAME + '.txt', 'r', encoding='utf8') as f:
        logging.info(f'Reading movie_ids from {NAME}.txt')
        lines = f.readlines()
        movie_ids = [line.strip() for line in lines]         
        
      
for i in movie_ids:
    tmdb_posters(i)
   
