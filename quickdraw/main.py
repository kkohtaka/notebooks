import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from utils import load_df_from_hd5


def drop_columns(df, col_names=[]):
    new_df = df.copy()
    for col_name in col_names:
        if col_name in df.columns:
            new_df.drop([col_name], axis='columns', inplace=True)
    return new_df


def count_strokes(df):
    def _counter(drawing):
        assert(type(drawing) == list)
        return len(drawing)

    assert('drawing' in df.columns)
    df.loc[:, 'strokecount'] = df.drawing.map(_counter)
    return df


def format_drawing(
    df,
    maxlen=256,
):
    def _parser(drawing):
        if type(drawing) == np.ndarray and drawing.shape == (maxlen, 3):
            return drawing
        assert(type(drawing) == list)
        data = [
            (xi, yi, i)
            for i, (x, y) in enumerate(drawing)
            for xi, yi in zip(x, y)
        ]
        data = np.stack(data)
        data[:, 2] = [1] + np.diff(data[:, 2]).tolist()
        data[:, 2] += 1
        return keras.preprocessing.sequence.pad_sequences(
            data.swapaxes(0, 1),
            maxlen=maxlen,
            padding='post',
        ).swapaxes(0, 1)

    assert('drawing' in df.columns)
    df.loc[:, 'drawing'] = df.drawing.map(_parser)
    return df


def label_encode(s):
    encoder = LabelEncoder()
    return encoder.fit_transform(s.values), encoder


def label_decode(encoder, s):
    return encoder.inverse_transform(s.values)


def is_gpu_available():
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    return len([x.name for x in local_devices if x.device_type == 'GPU']) > 0


def top_3_accuracy(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, 3)


def get_RNN(
    n_dims=3,  # x, y, and t
    dropout=0.3,
    n_classes=340,
    optimizer='adam',
):
    inputs = keras.layers.Input(
        shape=(None, n_dims,),
        name='input_1',
    )
    X = inputs

    X = keras.layers.Conv1D(48, (5,))(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Conv1D(64, (5,))(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Conv1D(96, (3,))(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.LSTM(128, return_sequences=True)(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.LSTM(128, return_sequences=False)(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Dense(512)(X)
    X = keras.layers.Dropout(dropout)(X)
    X = keras.layers.Dense(n_classes, activation='softmax')(X)

    outputs = X

    model = keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', top_3_accuracy],
    )

    return model


def get_features(df):
    return np.stack(df.drawing, 0)


def get_targets(df):
    return np_utils.to_categorical(df.word.values)


CV_RANDOM_STATE = 42
DRAWING_MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 1
MODEL_DIR = 'model'


all_df = load_df_from_hd5()

all_df = drop_columns(all_df, ['timestamp', 'key_id'])
all_df = count_strokes(all_df)
all_df = format_drawing(all_df, maxlen=DRAWING_MAX_LENGTH)
all_df.loc[:, 'countrycode'], _ = label_encode(all_df.countrycode)
all_df.loc[:, 'recognized'], _ = label_encode(all_df.recognized)
all_df.loc[:, 'word'], word_encoder = label_encode(all_df.word)

train_X, test_X, train_y, test_y = train_test_split(
    get_features(all_df),
    get_targets(all_df),
    test_size=0.1,
    random_state=CV_RANDOM_STATE,
)

with tf.Session() as sess:

    model = get_RNN(
        n_dims=train_X.shape[-1],
        n_classes=train_y.shape[-1],
    )
    model.summary()
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=MODEL_DIR,
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": train_X},
        y=train_y,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False,
    )
    estimator.train(
        input_fn=train_input_fn,
        steps=int(train_X.shape[0]/BATCH_SIZE*EPOCHS),
    )
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input_1": test_X},
        y=test_y,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False,
    )
    scores = estimator.evaluate(
        input_fn=test_input_fn,
    )
    print('score:', scores)
