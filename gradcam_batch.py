#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import glob
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.core import Lambda
import tensorflow as tf
import numpy as np
import cv2
import argparse
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_learning_phase(0) #set learning phase
from util import get_config, get_model

def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(model, image, category_index, layer_name, rows, cols, num_classess):

    if len(model.layers) == 2:
        model.pop()

    # model.summary()
    target_layer = lambda x: target_category_loss(x, category_index, num_classess)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =[l for l in model.layers[0].layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Return to BGR [0..255] from the preprocessed image
    image = image * 255
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.resize(cam, (rows, cols), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def load_image(img_path, rows, cols):
    img = image.load_img(img_path, target_size=(rows, cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x = x / 255.0
    return x, img

def get_arguments():
    parser = argparse.ArgumentParser(description="Gradcam")
    parser.add_argument("--config", type=str, required=True,
                        help="Specify the config to use")
    parser.add_argument("--model", type=str,  required=True,
                        help="Specify the model to use")
    parser.add_argument("--restore-from", type=str, required=True,
                        help="Where restore model parameters from.")
    parser.add_argument("--target-dir", type=str,  required=True, help="Place where target data stored")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    classess, result_dir, format = get_config(args.config)
    num_classess = len(classess)

    # create model
    model, rows, cols, target_layer = get_model(args.model, num_classess)
    model.load_weights(os.path.join(result_dir, args.restore_from))

    start1 = time.time()
    gradcam_model = Sequential()
    gradcam_model.add(model)
    elapsed_time = time.time() - start1
    model.summary()
    print ("loading_time:{0}".format(elapsed_time) + "[sec]")

    prefix = os.path.splitext(args.restore_from)[0]
    for targetImage in glob.glob(os.path.join(args.target_dir, "**/*." + format), recursive=True):
        print("Target Image:" + targetImage)
        preprocessed_input, original = load_image(targetImage, rows, cols)
        start2 = time.time()
        predictions = model.predict(preprocessed_input)[0]
        elapsed_time2 = time.time() - start2
        print ("prediction_time:{0}".format(elapsed_time2) + "[sec]")
        print('Predicted class:' + classess[np.argmax(predictions)])
        top = 5
        top_indices = predictions.argsort()[-top:][::-1]
        result = [(classess[i], predictions[i]) for i in top_indices]
        for x in result:
            print(x)

        predicted_class = np.argmax(predictions)
        cam, heatmap = grad_cam(gradcam_model, preprocessed_input, predicted_class, target_layer, rows, cols, num_classess)
        cv2.imwrite(os.path.join(result_dir, prefix + "-gradcam-" + os.path.split(targetImage)[1]), cam)


