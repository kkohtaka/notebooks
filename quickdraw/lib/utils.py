import os
from os import path
import glob
import zipfile
import pandas as pd
import keras


def get_dataset(
    max_bytes=1024 * 1024,
    dir='data',
):
    def _remove_last_line(file):
        file.seek(0, os.SEEK_END)
        pos = file.tell() - 1
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()

    from google.cloud.storage.client import Client

    client = Client.create_anonymous_client()
    bucket = client.get_bucket('quickdraw_dataset')
    blob_iterator = bucket.list_blobs(prefix='full/simplified')
    for blob in blob_iterator:
        file_name = '{}/{}'.format(dir, path.basename(blob.name))
        with open(file_name, 'wb+') as f:
            blob.download_to_file(
                f,
                start=0,
                end=max_bytes,
            )
        with open(file_name, 'r+') as f:
            _remove_last_line(f)


def load_df_from_json(
    path='data/*.ndjson',
):
    file_names = glob.glob(path)
    res = None
    for file_name in file_names:
        df = pd.read_json(file_name, lines=True)
        res = df if res is None else pd.concat([res, df], ignore_index=True)
    return res


def export_df_to_hd5(
    df,
    path='data/data.h5',
):
    df.to_hdf(path, key='df')


def load_df_from_hd5(
    path='data/data.h5',
):
    df = pd.read_hdf(path, key='df')
    return df


def is_gpu_available():
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    return len([x.name for x in local_devices if x.device_type == 'GPU']) > 0


def top_3_accuracy(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, 3)


def shuffle_dataset(
    input_zip='data/train_simplified.zip',
    n_splits=20,
):
    zip_file = zipfile.ZipFile(input_zip)
    for file_info in zip_file.infolist():
        print(f'Processing {file_info.filename}...')
        with zip_file.open(file_info) as file:
            df = pd.read_csv(file)
            df.loc[:, 'hash'] = df.apply(
                lambda x: hash(tuple(x)),
                axis='columns',
            )
        for idx in range(n_splits):
            sub_df = df.loc[df.hash % n_splits == idx, :].copy()
            sub_df.drop(['hash'], axis='columns', inplace=True)
            file_name = f'data/train_simplified_{idx+1:04d}.csv.gz'
            if path.isfile(file_name):
                sub_df.to_csv(
                    file_name,
                    mode='a',
                    header=False,
                    compression='gzip',
                    index=False,
                )
            else:
                sub_df.to_csv(
                    file_name,
                    compression='gzip',
                    index=False,
                )
