#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from keras.preprocessing import image
from keras.backend import tensorflow_backend as backend
import numpy as np
import glob
from util import get_config, get_model


def get_arguments():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--config", type=str, required=True,
                        help="Specify the config to use")
    parser.add_argument("--model", type=str, required=True,
                        help="Specify the model to use")
    parser.add_argument("--restore-from", type=str, required=True,
                        help="Where restore model parameters from.")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Target image directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    targetDir = args.target_dir
    classess, result_dir, format = get_config(args.config)
    num_classess = len(classess)

    model, rows, cols, _ = get_model(args.model, num_classess)
    model.load_weights(os.path.join(result_dir, args.restore_from))

    for filename in glob.glob(os.path.join(targetDir, "**/*." + format), recursive=True):
        # 画像を読み込んで4次元テンソルへ変換
        img = image.load_img(filename, target_size=(rows, cols))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理を行う
        x = x / 255.0

        # クラスを予測
        pred = model.predict(x)[0]

        # 予測確率が高いトップ5を出力
        print("input :" + filename)
        print("predict :")
        top = 5
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classess[i], pred[i]) for i in top_indices]
        for x in result:
            print(x)
    
    backend.clear_session()
