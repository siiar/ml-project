import sys
import logging
import time
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO)
assert sys.version_info >= (3, 6)  # make sure you have Python 3.5+


# TODO: Enter your API Key and name here
API_KEY = 'AAAAAAAAAAAAAAAAa'
NAME = 'sina'


def save_error_movie_ids(IDX, NAME):
    pd.DataFrame.from_dict([IDX]).to_csv(path_or_buf=f'{NAME}_error_movie_ids.csv',
                        mode='a',
                        header=False,
                        index=False,
                        encoding='utf8',
                        chunksize=10000)

def init_output_df_to_csv_file(DF, NAME):
    DF.to_csv(path_or_buf=f'{NAME}.csv',
                            mode='w',
                            index=False,
                            encoding='utf8',
                            chunksize=10000)

def append_df_to_csv_file(DF, NAME):
    DF.to_csv(path_or_buf=f'{NAME}.csv',
                            mode='a',
                            header=False,
                            index=False,
                            encoding='utf8',
                            chunksize=10000)

def main(API_KEY, NAME):
    with open(NAME + '.txt', 'r', encoding='utf8') as f:
        logging.info(f'Reading movie_ids from {NAME}.txt')
        lines = f.readlines()

    movie_ids = [line.strip() for line in lines]
    movie_list = list()

    min_count = 1
    max_count = 10
    count = min_count
    first_batch = True
    
    payload = {'api_key': API_KEY, 'language': 'en-US'}
    for idx in movie_ids:
        url = f"https://api.themoviedb.org/3/movie/{idx}"
        # logging.info(f'Sending GET request with movie_id={idx}')
        try:
            response = requests.get(url, params=payload, timeout=600)
            response.raise_for_status()
            movie_list.append(response.json())
            count += 1

            if (count == max_count):

                count = min_count
                movies_df = pd.DataFrame.from_dict(movie_list)

                if first_batch:
                    first_batch = False
                    init_output_df_to_csv_file(movies_df, NAME)
                else:
                    append_df_to_csv_file(movies_df, NAME)

                movie_list.clear()

            time.sleep(0.01)

        except Exception as error:
            save_error_movie_ids(idx, NAME)
            logging.exception(f'Encountered error for movie id {idx}: \n {error}')

    if len(movie_list) > 0:
        movies_df = pd.DataFrame.from_dict(movie_list)
        append_df_to_csv_file(movies_df, NAME)


    print('Finished writing all entries to CSV!')

        
if __name__ == '__main__':
    main(API_KEY, NAME)