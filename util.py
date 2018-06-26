import configparser
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Sequential, Model


def get_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    base = config["Base"]
    classess = [x.strip() for x in base.get("CLASSES").split(',')]
    result_dir = base.get("RESULT_DIR")
    format = base.get("IMG_FORMAT")

    return classess, result_dir, format


def get_model(model_name, num_classess):
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    if model_name ==  "VGG16":
        rows = 224
        cols = 224
        input_tensor = Input(shape=(rows, cols, 3))
        target_layer = "block5_conv3"
        target_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_name == "ResNet50":
        rows = 224
        cols = 224
        input_tensor = Input(shape=(rows, cols, 3))
        target_layer = "activation_49"
        target_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_name == "InceptionV3":
        rows = 299
        cols = 299
        input_tensor = Input(shape=(rows, cols, 3))
        target_layer = "mixed10"
        target_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=target_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classess, activation='softmax'))
    model = Model(input=target_model.input, output=top_model(target_model.output))
    return model, rows, cols, target_layer