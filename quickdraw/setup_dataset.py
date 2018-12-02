# from utils import split_dataset
# from utils import shuffle_dataset
# from utils import csv_to_npy
# import glob
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

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

# for csv in sorted(glob.glob('data/train_*.csv.gz')):
#     words = None
#     countrycodes = None
#
#     df = pd.read_csv(csv, usecols=['word', 'countrycode'])
#     df.countrycode.fillna('OTHER', inplace=True)
#
#     new_words = df.word.unique()
#     if words is None:
#         words = new_words
#     else:
#         words = np.concatenate((words, new_words), axis=0)
#
#     new_countrycodes = df.countrycode.unique()
#     if countrycodes is None:
#         countrycodes = new_countrycodes
#     else:
#         countrycodes = np.concatenate((countrycodes, new_countrycodes), axis=0)
#
#     del df
#
# word_encoder = LabelEncoder()
# countrycode_encoder = LabelEncoder()
#
# word_encoder.fit(words)
# countrycode_encoder.fit(countrycodes)
#
# csv_to_npy(
#     word_encoder,
#     countrycode_encoder,
#     input_csvs=glob.glob('data/train_*.csv.gz'),
#     img_size=128,
#     line_width=3,
#     alpha=1.0,
#     # chunksize=25000,
#     output_base='data/train',
# )
