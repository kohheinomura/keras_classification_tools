#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.backend import tensorflow_backend as backend
from util import get_config, get_model

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
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument("--config", type=str, required=True,
                        help="Specify the config to use")
    parser.add_argument("--model", type=str,
                        help="Specify the model to use")
    parser.add_argument("--train-data-dir", type=str, required=True,
                        help="Place where training data stored")
    parser.add_argument("--val-data-dir", type=str, required=True,
                        help="Place where validation data stored")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-epoch", type=int, required=True,
                        help="Number of epoch for training")
    parser.add_argument("--out-file", type=str, required=True,
                        help="File which stores model weight")
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    args = get_arguments()
    classess, result_dir, format = get_config(args.config)
    num_classess = len(classess)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 学習済み重みをロード
    model, rows, cols, _ = get_model(args.model, num_classess)

    # 最後のconv層の直前までの層をfreeze
    # for layer in model.layers[:15]:
    #     layer.trainable = True

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
        target_size=(rows, cols),
        color_mode='rgb',
        classes=classess,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True)

    # 検証データ取得ジェネレータの作成
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = test_datagen.flow_from_directory(
        args.val_data_dir,
        target_size=(rows, cols),
        color_mode='rgb',
        classes=classess,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True)

    # 学習実行
    cppath = os.path.join(result_dir, args.out_file + '.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5')
    cp_cb = ModelCheckpoint(filepath=cppath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.n/args.batch_size),
        epochs=args.num_epoch,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.n/args.batch_size),
        callbacks=[cp_cb])

    # 学習済みモデルとログをsave
    # model.save_weights(os.path.join(result_dir, args.out_file + '.h5'))
    save_history(history, os.path.join(result_dir, args.out_file + '.txt'))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    backend.clear_session()
