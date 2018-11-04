from utils import top_3_accuracy

import sys
import ast
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2


class ConvNet:
    def __init__(
        self,
        n_classes,
        img_size=64,
        line_width=6,
        alpha=2.5e-1,
    ):
        self.img_size = img_size
        self.n_classes = n_classes

    def get_model(self):
        model = keras.applications.MobileNet(
            input_shape=(self.img_size, self.img_size, 1),
            alpha=1.,
            weights=None,
            classes=self.num_classes,
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', top_3_accuracy],
        )

        self.model = model
        return model

    def get_features(self, df):
        df.loc[:, 'drawing'] = df.drawing.apply(ast.literal_eval)
        X = np.zeros((len(df), self.img_size, self.img_size, 1))
        for idx, strokes in enumerate(df.drawing.values):
            X[idx, :, :, 0] = self.generate_image(strokes)
        X = keras.applications.mobilenet.preprocess_input(X).astype(np.float32)
        return X

    def get_targets(self, df):
        return keras.utils.to_categorical(
            df.word,
            num_classes=self.num_classes,
        )

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

    def get_score(self, true_y, pred_y):
        return top_3_accuracy(true_y, pred_y)

    def generate_image(self, strokes):
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
        ratio = float(self.size) / float(base_size)

        line_width = int(self.line_width / ratio)
        base_size += line_width

        img = np.zeros((base_size, base_size), np.uint8)

        if max_x - min_x > max_y - min_y:
            x_origin = int(line_width / 2)
            y_origin = int((line_width + (max_x-min_x) - (max_y-min_y)) / 2)
        else:
            x_origin = int((line_width + (max_y-min_y) - (max_x-min_x)) / 2)
            y_origin = int(line_width / 2)

        for t, stroke in enumerate(strokes):
            color = 255 * (1 - self.alpha * np.arctan(t)) / np.pi
            for i in range(len(stroke[0]) - 1):
                cv2.line(
                    img,
                    (x_origin + stroke[0][i],   y_origin + stroke[1][i],),
                    (x_origin + stroke[0][i+1], y_origin + stroke[1][i+1],),
                    color,
                    line_width,
                )

        if base_size != self.size:
            return cv2.resize(img, (self.size, self.size))

        return img
