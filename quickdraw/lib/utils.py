import os
from os import path
import glob
import ast
import zipfile
import numpy as np
import pandas as pd
from tensorflow import keras
from convnet import generate_image
from multiprocessing.dummy import Pool as ThreadPool


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


def split_dataset(
    input_zip='data/train_simplified.zip',
    output_base='data/train_simplified',
    chunksize=1000000,
    n_splits=20,
):
    zip_file = zipfile.ZipFile(input_zip)
    for file_info in zip_file.infolist():
        print(f'Processing {file_info.filename}...')
        with zip_file.open(file_info) as file:
            for df in pd.read_csv(file, chunksize=chunksize):
                df.loc[:, 'hash'] = df.apply(
                    lambda x: hash(tuple(x)),
                    axis='columns',
                )
                for idx in range(n_splits):
                    sub_df = df.loc[df.hash % n_splits == idx, :].copy()
                    sub_df.drop(['hash'], axis='columns', inplace=True)
                    file_name = f'{output_base}_{idx+1:04d}.csv.gz'
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


def shuffle_dataset(
    input_csvs=[],
    random_state=42,
    output_base='data/train_simplified_shuffled',
    n_pools=4,
):
    def _shuffle_dataset(
        arg,
    ):
        input_csv = arg['input_csv']
        output_csv = arg['output_csv']
        pd.read_csv(
            input_csv,
        ).sample(
            frac=1,
            random_state=random_state,
        ).to_csv(
            output_csv,
            index=False,
            compression='gzip',
        )
    args = []
    for idx, csv in enumerate(sorted(input_csvs)):
        args.append(dict(
            input_csv=csv,
            output_csv=f'{output_base}_{idx+1:04d}.csv.gz',
        ))
    pool = ThreadPool(n_pools)
    pool.map(_shuffle_dataset, args)
    pool.close()
    pool.join()


def to_categorical(encoder, values):
    return keras.utils.to_categorical(
        encoder.transform(values),
        num_classes=len(encoder.classes_),
    )


def csv_to_npy(
    word_encoder,
    countrycode_encoder,
    input_csvs=[],
    img_size=128,
    line_width=3,
    alpha=1.0,
    chunksize=25000,
    output_base='data/train',
    n_pools=4,
    overwrite=False,
):
    def _csv_to_npy(arg):
        input_csv = arg['input_csv']
        output_base = arg['output_base']
        print(f'Processing {input_csv}...')

        for output_idx, df in enumerate(
            pd.read_csv(input_csv, chunksize=chunksize),
        ):
            output_file = f'{output_base}_{output_idx+1:04d}'
            if not overwrite:
                if path.isfile(f'{output_file}.npz'):
                    print(f'Skip generating {output_file}...')
                    continue

            print(f'Generating {output_file}...')

            df.loc[:, 'drawing'] = df.drawing.apply(ast.literal_eval)
            df.countrycode.fillna('OTHER', inplace=True)
            image = np.zeros((len(df), img_size, img_size, 1))
            strokecount = np.zeros((len(df), 1))
            for idx, strokes in enumerate(df.drawing.values):
                image[idx, :, :, 0] = generate_image(
                    strokes,
                    img_size=img_size,
                    line_width=line_width,
                    alpha=alpha,
                )
                strokecount[idx, :] = np.asarray([len(strokes)])
            np.savez_compressed(
                output_file,
                image=image.astype(np.float32),
                countrycode=to_categorical(
                    countrycode_encoder,
                    df.countrycode.values,
                ).astype(np.float32),
                strokecount=strokecount.astype(np.float32),
                recognized=df.recognized.values.astype(np.float32),
                word=to_categorical(
                    word_encoder,
                    df.word.values,
                ).astype(np.float32),
            )

    base_dir = path.dirname(output_base)
    if not path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    args = []
    for idx, csv in enumerate(sorted(input_csvs)):
        args.append(dict(
            input_csv=csv,
            output_base=f'{output_base}_{idx+1:04d}',
        ))
    pool = ThreadPool(n_pools)
    pool.map(_csv_to_npy, args)
    pool.close()
    pool.join()
