from utils import top_3_accuracy

import numpy as np
from keras.utils import np_utils
from tensorflow import keras


class RNN:
    def __init__(
        self,
        n_dims=3,
        dropout=0.3,
        n_classes=340,
        optimizer='adam',
    ):
        self.n_dims = n_dims
        self.dropout = dropout
        self.n_classes = n_classes
        self.optimizer = optimizer

    def get_model(self):
        inputs = keras.layers.Input(
            shape=(None, self.n_dims,),
        )
        X = inputs

        X = keras.layers.Conv1D(48, (5,))(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.Conv1D(64, (5,))(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.Conv1D(96, (3,))(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.LSTM(128, return_sequences=True)(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.LSTM(128, return_sequences=False)(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.Dense(512)(X)
        X = keras.layers.Dropout(self.dropout)(X)
        X = keras.layers.Dense(self.n_classes, activation='softmax')(X)

        outputs = X

        model = keras.models.Model(inputs, outputs)

        self.model = model

        return model

    def get_features(self, df):
        return np.stack(df.drawing, 0)

    def get_targets(self, df):
        return np_utils.to_categorical(
            df.word.values,
            num_classes=self.n_classes,
        )

    def get_generator(self, df, batch_size=1):
        is_training = True if 'word' in df.columns else False
        while True:
            offset = 0
            while len(df) >= offset:
                new_df = df.iloc[offset:offset+batch_size, :]
                offset += batch_size
                X = self.get_features(new_df)
                if is_training:
                    y = self.get_targets(new_df)
                    yield X, y
                else:
                    yield X

    def get_score(self, true_y, pred_y):
        return top_3_accuracy(true_y, pred_y)
