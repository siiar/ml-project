%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from fastai import *
from fastai.vision import *

import json

from utils import *
#generate data from csv

PATH = Path('data/txt/train')
def create_train_txts_from_df(path):
    df = pd.read_csv(path)
    klass = '_'.join(path.stem.split())
    (PATH/klass).mkdir(exist_ok=True)
    for row in df.iterrows():
        example = {
            'countrycode': row[1].countrycode,
            'drawing': json.loads(row[1].drawing),
            'key_id': row[1].key_id,
            'recognized': row[1].recognized
        }
        with open(PATH/klass/f'{example["key_id"]}.txt', mode='w') as f: json.dump(example, f)

%time for p in Path('data/train').iterdir(): create_train_txts_from_df(p)

%%time

countrycodes=set()
for p in Path('data/train').iterdir():
    df = pd.read_csv(p)
    countrycodes = countrycodes.union(set(df.countrycode))
pd.to_pickle(countrycodes, 'data/countrycodes.pkl')