#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import *
import os
import argparse
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.backend import tensorflow_backend as backend
import numpy as np

IMG_ROWS, IMG_COLS = 224, 224
CHANNELS = 3
RESTORE_FROM = "resnet50_classification_out.h5"
NUM_CLASSES = len(CLASSES)

def get_arguments():
    parser = argparse.ArgumentParser(description="ResNet50 Network")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--target-image", type=str,
                        help="Target Image")
    return parser.parse_args()

def main():
    args = get_arguments()
    filename = args.target_image
    print('input:', filename)

    # ResNet50の構築
    input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    fc = Sequential()
    fc.add(Flatten(input_shape=resnet50.output_shape[1:]))
    fc.add(Dense(256, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(NUM_CLASSES, activation='softmax'))
    model = Model(input=resnet50.input, output=fc(resnet50.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(RESULT_DIR, args.restore_from))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(IMG_ROWS, IMG_COLS))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理を行う
    x = x / 255.0

    # クラスを予測
    # 入力は1枚の画像なので[0]のみ
    pred = model.predict(x)[0]

    # 予測確率が高いトップ5を出力
    top = 5
    top_indices = pred.argsort()[-top:][::-1]
    result = [(CLASSES[i], pred[i]) for i in top_indices]
    for x in result:
        print(x)
    
    backend.clear_session()

 
if __name__ == '__main__':
    main()
