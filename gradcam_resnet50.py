#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import *
from keras.applications.resnet50 import (ResNet50)
from keras.preprocessing import image
from keras.layers.core import Lambda
import tensorflow as tf
import numpy as np
import cv2
import argparse
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_learning_phase(0) #set learning phase
import os
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense

TARGET_DATA_DIR = './data/val_image'
IMG_ROWS, IMG_COLS = 224, 224
CHANNELS = 3
NUM_CLASSES = len(CLASSES)

def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(input_model, image, category_index, layer_name):

    model = Sequential()
    model.add(input_model)
    # model.summary()
    target_layer = lambda x: target_category_loss(x, category_index, NUM_CLASSES)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    # model.summary()
    loss = K.sum(model.layers[-1].output)
    conv_output =[l for l in model.layers[0].layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    # cv2.imwrite("test0.jpg", cam)
    heatmap = cam / np.max(cam)
    # cv2.imwrite("test1.jpg", heatmap)

    #Return to BGR [0..255] from the preprocessed image
    image = image * 255
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # cv2.imwrite("test2.jpg", cam)
    cam = np.float32(cam) + np.float32(image)
    # cv2.imwrite("test3.jpg", cam)
    cam = 255 * cam / np.max(cam)
    # cv2.imwrite("test4.jpg", cam)
    return np.uint8(cam), heatmap


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x = x / 255.0
    return x, img

def createHeatmapGrid(args, cls, model, out_layer):
    numImages = 4
    images = glob.glob(os.path.join(args.target_data_dir, cls, "*." + IMG_FORMAT))
    # random.shuffle(images)
    fig = plt.figure()
    fig.suptitle('Class - ' + cls, size=20)
    gs = gridspec.GridSpec(numImages, 4)
    gs.update(wspace=0.3, hspace=0.3)
    for gridNum in range(0, numImages * 4, 2):
        targetImage = images[gridNum]
        print("Target Image:" + targetImage)
        preprocessed_input, original = load_image(targetImage)
        start2 = time.time()
        predictions = model.predict(preprocessed_input)[0]
        elapsed_time2 = time.time() - start2
        print ("prediction_time:{0}".format(elapsed_time2) + "[sec]")
        print('Predicted class:' + CLASSES[np.argmax(predictions)])
        top = 5
        top_indices = predictions.argsort()[-top:][::-1]
        result = [(CLASSES[i], predictions[i]) for i in top_indices]
        for x in result:
            print(x)
        # print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

        predicted_class = np.argmax(predictions)
        cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, out_layer)
        # cv2.imwrite("gradcam.jpg", cam)

        plt.subplot(gs[gridNum])
        plt.imshow(original, aspect='auto')
        plt.axis("off")
        plt.subplot(gs[gridNum + 1]).set_title(result[0])
        plt.imshow(cam, aspect='auto')
        plt.axis("off")
    plt.savefig("heatmap-" + cls + ".jpg")

def get_arguments():
    parser = argparse.ArgumentParser(description="ResNet50 Network")
    parser.add_argument("--restore-from", type=str, default="resnet50_classification.h5",
                        help="Where restore model parameters from.")
    parser.add_argument("--target-data-dir", type=str, default=TARGET_DATA_DIR,
                        help="Place where target data stored")
    parser.add_argument("--target-classes", type=str,
                        help="Target classes which to create gradcam image")
    return parser.parse_args()

def main():
    args = get_arguments()
    # create model
    input_tensor = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_CLASSES, activation='softmax'))
    model = Model(input=resnet50.input, output=top_model(resnet50.output))
    model.load_weights(os.path.join(RESULT_DIR, args.restore_from))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    target_classes = args.target_classes.split(',')
    for cls in target_classes:
        createHeatmapGrid(args, cls, model, 'activation_49')

if __name__ == '__main__':
    main()