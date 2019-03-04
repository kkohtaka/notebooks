from datetime import datetime, timedelta, timezone
import os
import random
import math

import click

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras


def print_versions():
    print(f'====== VERSIONS ======')
    print(f'NumPy {np.__version__}')
    print(f'Pandas {pd.__version__}')
    print(f'sklearn {sklearn.__version__}')
    print(f'TensorFlow {tf.__version__}')
    print(f'======================')


def reset_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = f'{seed}'
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


reset_random_seed()
print_versions()


class Generator(keras.utils.Sequence):

    def __init__(
        self,
        df,
        column_name='ltp',
        batch_size=1,
        lookback_timesteps=100,
        target_timesteps=1,
    ):
        self.df = df
        self.data_size = len(df)
        self.column_name = column_name
        self.n_classes = 1
        self.batch_size = batch_size
        self.lookback_timesteps = lookback_timesteps
        self.target_timesteps = target_timesteps

    def __getitem__(self, idx):
        batch_size = self.batch_size
        if batch_size > self.data_size - idx:
            batch_size = self.data_size - idx

        lookback_windows = np.empty(
            (batch_size, self.lookback_timesteps, self.n_classes))
        target_windows = np.empty(
            (batch_size, self.target_timesteps))
        for i in range(batch_size):
            offset = idx + i
            lookback_window_start = offset
            lookback_window_end = lookback_window_start + self.lookback_timesteps
            target_window_start = lookback_window_end
            target_window_end = target_window_start + self.target_timesteps
            lookback_windows[i, :] = (
                self.df[self.column_name].iloc[lookback_window_start:lookback_window_end].values[:, np.newaxis])
            target_windows[i, :] = (
                self.df[self.column_name].iloc[target_window_start:target_window_end].values)
        return lookback_windows, target_windows

    def __len__(self):
        return max(0, int((self.data_size - (self.lookback_timesteps + self.target_timesteps) - 1) / self.batch_size) + 1)

    def on_epoch_end(self):
        pass


def check(df):
    print(df.head(n=10))
    print(df.describe())


@click.command()
@click.option(
    '--input',
    default='input.csv',
    help='Path to an input CSV file',


)
def train(
    input,
):
    df = pd.read_csv(input, index_col='timestamp')
    df.index = pd.to_datetime(df.index)
    # check(df)

    df = df.loc[~(df.ltp == 0)]
    # check(df)

    df = df.resample('T').first().interpolate('quadratic')
    # check(df)

    df['ltp'] = preprocessing.StandardScaler().fit_transform(df[['ltp']])
    # check(df)

    history_path = 'history.csv'
    model_path = 'checkpoint.h5'

    lstm_units = 16
    lookback_timesteps = 100
    n_classes = 1

    verbose = True

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=lstm_units,
        batch_input_shape=(None, lookback_timesteps, n_classes),
        return_sequences=True,
        name='lstm_0',
    ))
    model.add(keras.layers.LSTM(
        units=lstm_units,
        name='lstm_1',
    ))
    model.add(keras.layers.Dense(
        units=n_classes,
        activation=keras.activations.linear,
        name='dense_0',
    ))
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.metrics.mean_squared_error,
        metrics=[
            keras.metrics.mean_squared_error,
            keras.metrics.mean_absolute_error,
            keras.metrics.mean_absolute_percentage_error,
        ],
    )
    model.summary()

    callbacks = []
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_absolute_percentage_error',
            factor=0.5,
            patience=5,
            min_delta=0.005,
            mode='max',
            cooldown=3,
            verbose=verbose,
        ),
    )
    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            history_path,
            append=False,
        ),
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_mean_absolute_percentage_error',
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            verbose=verbose,
        )
    )

    epochs = 1
    batch_size = 32
    n_splits = 2
    validation_split = 0.25
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        valid_len = math.ceil(float(len(train_df)) * validation_split)
        train_df, valid_df = train_df.iloc[:-valid_len], train_df[-valid_len:]
        # print(train_df.head(n=1))
        # print(valid_df.head(n=1))
        # print(test_df.head(n=1))

        if verbose:
            print(f'Training...')
        model.fit_generator(
            generator=Generator(
                train_df,
                batch_size=batch_size,
            ),
            validation_data=Generator(
                valid_df,
                batch_size=batch_size,
            ),
            epochs=epochs,
            shuffle=True,
            callbacks=callbacks,
            verbose=verbose,
        )

        if verbose:
            print(f'Testing...')
        scores = model.evaluate_generator(
            generator=Generator(
                test_df,
                batch_size=batch_size,
            ),
            verbose=verbose,
        )

        if len(model.metrics_names) == 1:
            scores = {model.metrics_names[0]: scores}
        else:
            scores = dict(zip(model.metrics_names, scores))
        print(f'scores: {scores}')


if __name__ == '__main__':
    train()
