import sys
import ast
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2


def generate_image(
    strokes,
    img_size=128,
    line_width=3,
    alpha=1.0,
):
    max_x = max_y = 0
    min_x = min_y = sys.maxsize
    for t, stroke in enumerate(strokes):
        for i in range(len(stroke[0])):
            x, y = stroke[0][i], stroke[1][i]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

    base_size = max(max_x - min_x, max_y - min_y)
    ratio = float(img_size) / float(base_size)

    line_width = int(max(line_width / ratio, 1))
    base_size = int(base_size + line_width)

    img = np.zeros((base_size, base_size), np.uint8)

    if max_x - min_x > max_y - min_y:
        x_origin = int(line_width / 2)
        y_origin = int((line_width + (max_x-min_x) - (max_y-min_y)) / 2)
    else:
        x_origin = int((line_width + (max_y-min_y) - (max_x-min_x)) / 2)
        y_origin = int(line_width / 2)

    for t, stroke in enumerate(strokes):
        color = 255 * (1 - 2*np.arctan(alpha * t)/np.pi)
        if color < 1.0:
            break
        for i in range(len(stroke[0]) - 1):
            cv2.line(
                img,
                (
                    int(x_origin + stroke[0][i]),
                    int(y_origin + stroke[1][i]),
                ),
                (
                    int(x_origin + stroke[0][i+1]),
                    int(y_origin + stroke[1][i+1]),
                ),
                color,
                line_width,
            )

    if base_size != img_size:
        return cv2.resize(img, (img_size, img_size))

    return img


class ConvNet:
    def __init__(
        self,
        word_encoder,
        countrycode_encoder,
        img_size=64,
        line_width=3,
        alpha=2.5e-1,
    ):
        self.n_classes = len(word_encoder.classes_)
        self.n_countrycodes = len(countrycode_encoder.classes_)

        self.word_encoder = word_encoder
        self.countrycode_encoder = countrycode_encoder

        self.img_size = img_size
        self.line_width = line_width
        self.alpha = alpha

    def get_model(self):
        # TODO(kkohtaka): Use 'imagenet' weights as an initial weights
        mobile_net = keras.applications.MobileNet(
            include_top=False,
            input_shape=(self.img_size, self.img_size, 1),
            alpha=1.,
            # weights="imagenet",
            weights=None,
        )
        # for layer in mobile_net.layers[:-4]:
        #     layer.trainable = False
        # for layer in mobile_net.layers:
        #     print(layer, layer.trainable)

        X_drawing = keras.layers.Flatten(
            name='flatten_mobilenet',
        )(mobile_net.output)

        input_countrycode = keras.layers.Input(
            shape=(self.n_countrycodes,),
            name='input_countrycode'
        )
        X_countrycode = input_countrycode

        input_strokecount = keras.layers.Input(
            shape=(1,),
            name='input_strokecount'
        )
        X_strokecount = input_strokecount

        X = keras.layers.Concatenate(
            name='concatenate',
        )([
            X_drawing,
            X_countrycode,
            X_strokecount,
        ])
        X = keras.layers.Dense(512)(X)
        X = keras.layers.Dropout(0.5, name='dropout_last')(X)
        X = keras.layers.Dense(self.n_classes, activation='softmax')(X)

        model = keras.models.Model(
            inputs=(
                mobile_net.input,
                input_countrycode,
                input_strokecount,
            ),
            outputs=(
                X,
            ),
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'categorical_accuracy',
            ],
        )

        self.model = model
        return model

    def to_categorical(self, encoder, values):
        return keras.utils.to_categorical(
            encoder.transform(values),
            num_classes=len(encoder.classes_),
        )

    def get_features(self, df):
        df.loc[:, 'drawing'] = df.drawing.apply(ast.literal_eval)
        df.countrycode.fillna('OTHER', inplace=True)
        image = np.zeros((len(df), self.img_size, self.img_size, 1))
        strokecount = np.zeros((len(df), 1))
        for idx, strokes in enumerate(df.drawing.values):
            image[idx, :, :, 0] = self.generate_image(strokes)
            strokecount[idx, :] = np.asarray([len(strokes)])
        X = {
            'input_1': keras.applications.mobilenet.preprocess_input(
                image,
            ).astype(np.float32),
            'input_countrycode': self.to_categorical(
                self.countrycode_encoder,
                df.countrycode.values,
            ),
            'input_strokecount': strokecount,
        }
        return X

    def get_targets(self, df):
        return self.to_categorical(self.word_encoder, df.word.values)

    def get_generator(
        self,
        csvs,
        is_training=True,
        batch_size=32,
    ):
        while True:
            for csv in sorted(csvs):
                for df in pd.read_csv(csv, chunksize=batch_size):
                    X = self.get_features(df)
                    if not is_training:
                        yield X
                    else:
                        y = self.get_targets(df)
                        yield X, y

    def get_callbacks(self):
        return [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_categorical_accuracy',
                factor=0.5,
                patience=5,
                min_delta=0.005,
                mode='max',
                cooldown=3,
                verbose=1,
            ),
        ]

    def generate_image(self, strokes):
        return generate_image(
            strokes,
            self.img_size,
            self.line_width,
            self.alpha,
        )
