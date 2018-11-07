import glob

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from convnet import ConvNet


VALIDATION_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 5


def count_datasets_total(csvs):
    count = 0
    for csv in csvs:
        df = pd.read_csv(csv, usecols=['key_id'])
        count += len(df)
        del df
    return count


def get_encoders(csvs):
    word_encoder = LabelEncoder()
    countrycode_encoder = LabelEncoder()
    for csv in csvs:
        df = pd.read_csv(csv, usecols=['word', 'countrycode'])
        df.countrycode.fillna('OTHER', inplace=True)
        word_encoder.fit(df.word.values)
        countrycode_encoder.fit(df.countrycode.values)
        del df
    return word_encoder, countrycode_encoder


csvs = sorted(glob.glob('data/train_simplified_*.csv.gz'))

train_csvs_size = np.floor(len(csvs)*(1.0-VALIDATION_SPLIT)).astype('int')

assert(train_csvs_size > 0)
assert(train_csvs_size < len(csvs))

train_csvs = csvs[:train_csvs_size]
valid_csvs = csvs[train_csvs_size:]

n_train = count_datasets_total(train_csvs)
n_valid = count_datasets_total(valid_csvs)

print(f'# of training set: {n_train}')
print(f'# of validation set: {n_valid}')

word_encoder, countrycode_encoder = get_encoders(csvs)

with tf.Session() as sess:
    K.set_session(sess)

    convnet = ConvNet(
        word_encoder,
        countrycode_encoder,
    )

    model = convnet.get_model()
    model.summary()
    hist = model.fit_generator(
        convnet.get_generator(
            train_csvs,
            batch_size=BATCH_SIZE,
        ),
        steps_per_epoch=np.ceil(n_train/BATCH_SIZE).astype('int'),
        epochs=EPOCHS,
        validation_data=convnet.get_generator(
            valid_csvs,
            batch_size=BATCH_SIZE,
        ),
        validation_steps=np.ceil(n_valid/BATCH_SIZE).astype('int'),
        callbacks=convnet.get_callbacks(),
    )
