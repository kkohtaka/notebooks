# from utils import split_dataset
# from utils import shuffle_dataset
from utils import csv_to_npy
import glob
from os import path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# split_dataset(
#     input_zip='data/train_raw.zip',
#     output_base='data/train_raw_split',
#     chunksize=1000000,
#     n_splits=100,
# )

# shuffle_dataset(
#     input_csvs=glob.glob('data/train_raw_split_*.csv.gz'),
#     output_base='data/train_shuffle',
# )

shuffled_csvs = sorted(glob.glob('data/train_shuffle_*.csv.gz'))

word_encoder = LabelEncoder()
countrycode_encoder = LabelEncoder()

if path.isfile('data/words.npy') and path.isfile('data/countrycodes.npy'):
    word_encoder.classes_ = np.load('data/words.npy')
    countrycode_encoder.classes_ = np.load('data/countrycodes.npy')
else:
    words = np.asarray([])
    countrycodes = np.asarray([])
    for csv in shuffled_csvs:
        for df in pd.read_csv(
            csv,
            usecols=['word', 'countrycode'],
            chunksize=1000000,
        ):
            df.countrycode.fillna('OTHER', inplace=True)

            words = np.concatenate(
                (words, df.word.unique()),
                axis=0,
            )
            words = np.unique(words)

            countrycodes = np.concatenate(
                (countrycodes, df.countrycode.unique()),
                axis=0,
            )
            countrycodes = np.unique(countrycodes)

    word_encoder.fit(words)
    np.save('data/words.npy', word_encoder.classes_)

    countrycode_encoder.fit(countrycodes)
    np.save('data/countrycodes.npy', countrycode_encoder.classes_)

print(f'words: {word_encoder.classes_}')
print(f'country codes: {countrycode_encoder.classes_}')

img_size = 128
line_width = 3
alpha = 1.0
output_base = f'npy.new/{img_size:04d}_{line_width:03d}_{alpha:.2f}/train'

csv_to_npy(
    word_encoder,
    countrycode_encoder,
    input_csvs=shuffled_csvs,
    img_size=128,
    line_width=3,
    alpha=1.0,
    chunksize=10000,
    n_pools=4,
    output_base=output_base,
)
