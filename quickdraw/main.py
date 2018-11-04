import numpy as np
import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from utils import load_df_from_hd5
from utils import top_3_accuracy
from rnn import RNN


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


CV_RANDOM_STATE = 42
TRAIN_SIZE = None
VALIDATION_SPLIT = 0.5
if TRAIN_SIZE is not None:
    TEST_SIZE = int(np.ceil(TRAIN_SIZE*VALIDATION_SPLIT))
else:
    TEST_SIZE = VALIDATION_SPLIT

DRAWING_MAX_LENGTH = 64
BATCH_SIZE = 1024
EPOCHS = 1


all_df = load_df_from_hd5()

all_df = drop_columns(all_df, ['timestamp', 'key_id'])
all_df = count_strokes(all_df)
all_df = format_drawing(all_df, maxlen=DRAWING_MAX_LENGTH)
all_df.loc[:, 'countrycode'], _ = label_encode(all_df.countrycode)
all_df.loc[:, 'recognized'], _ = label_encode(all_df.recognized)
all_df.loc[:, 'word'], word_encoder = label_encode(all_df.word)

n_dims = 3
n_classes = all_df.word.nunique()

K.clear_session()
with tf.Session() as sess:
    K.set_session(sess)

    rnn = RNN(
        n_dims=n_dims,
        n_classes=n_classes,
    )

    train_df, valid_df = train_test_split(
        all_df,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=CV_RANDOM_STATE,
    )

    model = rnn.get_model()
    model.summary()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', top_3_accuracy],
    )
    hist = model.fit_generator(
        rnn.get_generator(train_df, batch_size=BATCH_SIZE),
        steps_per_epoch=np.ceil(len(train_df)/BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=rnn.get_generator(valid_df, batch_size=BATCH_SIZE),
        validation_steps=np.ceil(len(valid_df)/BATCH_SIZE),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_categorical_accuracy',
                factor=0.5,
                patience=5,
                min_delta=0.005,
                mode='max',
                cooldown=3,
                verbose=1,
            ),
        ],
    )
