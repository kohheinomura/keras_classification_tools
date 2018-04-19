#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import *
import os
import time
import argparse
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

BATCH_SIZE = 32
NUM_EPOCH = 100
OUT_FILE = "resnet50_classification_out"
TRAIN_DATA_DIR = './data/train_images'
VALIDATION_DATA_DIR = './data/val_images'
IMG_ROWS, IMG_COLS = 224, 224
CHANNELS = 3
NUM_CLASSES = len(CLASSES)

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def get_arguments():
    parser = argparse.ArgumentParser(description="ResNet50 Network")
    parser.add_argument("--train-data-dir", type=str, default=TRAIN_DATA_DIR,
                        help="Place where training data stored")
    parser.add_argument("--val-data-dir", type=str, default=VALIDATION_DATA_DIR,
                        help="Place where validation data stored")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-epoch", type=int, default=NUM_EPOCH,
                        help="Number of epoch for training")
    parser.add_argument("--out-file", type=str, default=OUT_FILE,
                        help="Number of epoch for training")
    return parser.parse_args()

def main():
    args = get_arguments()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    # 学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層を構築
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_CLASSES, activation='softmax'))
    model = Model(input=resnet50.input, output=top_model(resnet50.output))

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:170]:
        layer.trainable = True

    # モデルのコンパイル。optimizerとしてはSGDを指定
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # 学習データ取得ジェネレータの作成
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        args.train_data_dir,
        target_size=(IMG_ROWS, IMG_COLS),
        color_mode='rgb',
        classes=CLASSES,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True)

    # 検証データ取得ジェネレータの作成
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        args.val_data_dir,
        target_size=(IMG_ROWS, IMG_COLS),
        color_mode='rgb',
        classes=CLASSES,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True)

    # 学習実行
    history = model.fit_generator(
        train_generator,
        epochs=args.num_epoch,
        validation_data=validation_generator)

    # 学習済みモデルとログをsave
    model.save_weights(os.path.join(RESULT_DIR, args.out_file + '.h5'))
    save_history(history, os.path.join(RESULT_DIR, args.out_file + '.txt'))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
    main()
