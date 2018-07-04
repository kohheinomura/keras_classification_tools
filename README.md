# Kerasを利用した分類ツール


`ResNet50`,  `VGG16`, `InceptionV3`を用いたKerasベースの分類、推論、Grad-Cam実装です。
必要なソースコードを`git clone`します。

```
$ git clone https://github.com/kohheinomura/keras_classification_tools.git
```

以下では、Kaggleの[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)のデータを用いて利用方法を示します。

### 事前設定

全体にまたがる設定を行うコンフィグ・ファイルを作成します。コンフィグ・ファイル名は任意の名前を指定可能です。以下では、`config_dogcat`とします。

```
$ cd keras_classification_tools
$ vi config_dogcat
```

以下は`config_dogcat`の設定内容です。

```
[Base]
CLASSES = dog,cat
RESULT_DIR = ./results_dogcat
IMG_FORMAT = jpg
```

- CLASSES：カンマ区切りで分類対象のクラス名を指定します。上記サンプルではdog, catの2クラスを指定しています。
- RESULT_DIR：学習結果としてのモデルと、ログ・ファイルの書き出しディレクトリを指定します。またGrad-CAMの出力もこのディレクトリに生成されます。
- IMG_FORMAT：利用する画像のフォーマットを指定します。

### 学習・バリデーション用データの配置

学習用データとバリデーション用データを任意のディレクトリ配下に置きます。以下の例では、`data/train_images`に、学習データを、またバリデーション用データを`data/val_images`以下に配置しています。それそれのディレクトリには、コンフィグ・ファイルで指定したクラス名と同じディレクトリを作成し、各ディレクトリにはクラスに応じた画像を配置します。

例：

```
data/train_images
|-dog
|  |-dog.1.jpg
|  |-dog.2.jpg
|
|-cat
   |-cat.100.jpg

data/val_images
|-dog
|  |-dog.1.jpg
|  |-dog.2.jpg
|
|-cat
   |-cat.100.jpg
```

### 学習

学習を実行します。学習には以下のpythonスクリプトを実行します。

実行例：

```
$ python train.py --config ./config_dogcat --model VGG16 --batch-size 16 --num-epoch 3 --out-file vgg16_dogcat --train-data-dir ./data/train_images --val-data-dir ./data/val_images
```

- --config：コンフィグ・ファイルのパスを指定します。
- --model：利用したいモデルを指定します。当ツールでは、`VGG16`、`ResNet50`、`InceptionV3`が指定可能です。
- --batch-size：学習時のバッチサイズを指定します。
- --num-epoch：学習時のエポック数を指定します。
- --out-file：学習済みモデルの重みファイルのファイル名を指定します。このファイルはコンフィグ・ファイルで指定したRESULT_DIR直下に生成されます。また生成されるファイルは、`.h5`、`.txt`の２種類が生成されます。前者が学習結果としての重みが格納され、後者が学習時のログになります。
- --train-data-dir：学習用データのルート・ディレクトリを指定します。ここで指定したディレクトリ直下にコンフィグ・ファイルで指定したクラス名と同じ名前のディレクトリを作成のうえ画像データを配置します。
- --val-data-dir：validation用データのルート・ディレクトリを指定します。ここで指定したディレクトリ直下にコンフィグ・ファイルで指定したクラス名と同じ名前のディレクトリを作成のうえ画像データを配置します。



### 推論

学習済みモデルを利用して画像の分類を実行します。推論には以下のpythonスクリプトを実行します。

実行例：

```
$ python predict_batch.py --config config_dogcat --model VGG16 --restore-from vgg16_dogcat.h5 --target-dir ./data/val_images
```

- --config：コンフィグ・ファイルのパスを指定します。
- --model：利用したいモデルを指定します。当ツールでは、`VGG16`、`ResNet50`、`InceptionV3`が指定可能です。
- --restore-from：学習済みモデルの重みファイルを指定します。ファイルのパスは、コンフィグ・ファイルで指定したRESULT_DIRからの相対パスになります。
- --target-dir：推論したい画像ファイルがあるディレクトリを指定します。尚、ここで指定したディレクトリは再帰的に走査され、配下のすべての画像が推論対象となります。


### Grad-CAMによるヒートマップ作成

学習済みモデルを利用してヒートマップ画像を生成します。ヒートマップの作成には以下のpythonスクリプトを実行します。スクリプト実行後、コンフィグ・ファイルで指定したRESULT_DIRに以下のファイル名でヒートマップ画像が生成されます。

```
<学習済みモデルの重みファイル名>-gradcam-<画像ファイル名>.jpg
```

実行例：

```
$ python gradcam_batch.py --config config_dogcat --model VGG16 --restore-from vgg16_dogcat.h5 --target-dir ./data/val_images
```

- --config：コンフィグ・ファイルのパスを指定します。
- --model：利用したいモデルを指定します。当ツールでは、`VGG16`、`ResNet50`、`InceptionV3`が指定可能です。
- --restore-from：学習済みモデルの重みファイルを指定します。ファイルのパスは、コンフィグ・ファイルで指定したRESULT_DIRからの相対パスになります。
- --target-dir：ヒートマップ画像を作成したい画像ファイルがあるディレクトリを指定します。尚、ここで指定したディレクトリは再帰的に走査され、配下のすべての画像ファイルのヒートマップ画像が生成されます。

