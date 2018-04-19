ResNet50を用いたKerasベースの分類、推論、Grad-Cam実装です。

### 事前設定

学習、推論を実行する前に`config.py`にて、全体に跨る設定を行います。

```
CLASSES = ['C1','C2','C3','C4','C5']
RESULT_DIR = './results'
IMG_FORMAT = "bmp"
```

- CLASSES：カンマ区切りで分類対象のクラス名を指定します。上記サンプルではC1〜C5の５クラスを指定しています。
- RESULT_DIR：学習結果としてのモデルと、ログ・ファイルの書き出しディレクトリを指定します。
- IMG_FORMAT：利用する画像のフォーマットを指定します。

### ResNet50による分類の学習

ResNet50を利用した学習を実行します。学習には以下のpythonスクリプトを実行します。

実行例：

```
python resnet50_classification.py --train-data-dir data/train_images --val-data-dir data/val_images --batch-size 32 --num-epoch 10 --out-file resnet50_classification_out
```

- --train-data-dir：学習用データのルート・ディレクトリを指定します。ここで指定したディレクトリ直下にconfig.pyで指定したクラス名と同じ名前のディレクトリを作成のうえ画像データを配置します。このオプションを指定しない場合、デフォルトの`data/train_images`が適用されます。
- --val-data-dir：validation用データのルート・ディレクトリを指定します。ここで指定したディレクトリ直下にconfig.pyで指定したクラス名と同じ名前のディレクトリを作成のうえ画像データを配置します。このオプションを指定しない場合、デフォルトの`data/val_images`が適用されます。
- --batch-size：学習時のバッチサイズを指定します。このオプションを指定しない場合、デフォルトの`32`が適用されます。
- --num-epoch：学習時のエポック数を指定します。このオプションを指定しない場合、デフォルトの`100`が適用されます。
- --out-file：学習済みモデルの重みファイルのファイル名を指定します。このファイルはconfig.pyで指定したRESULT_DIR直下に生成されます。また生成されるファイルは、`.h5`、`.txt`の２種類が生成されます。前者が学習結果としての重みが格納され、後者が学習時のログになります。

### 学習済みResNet50による推論

学習済みResNet50を利用して１枚の画像の分類を実行します。推論には以下のpythonスクリプトを実行します。

実行例：

```
python resnet50_predict.py --restore-from resnet50_classification_out.h5 --target-image ./data/val_images/C1/c1_image_001.bmp
```

- --restore-from：学習済みモデルの重みファイルを指定します。ファイルのパスは、config.pyで指定したRESULT_DIRからの相対パスになります。
- --target-image：分類の予測を行いたい画像のパスを指定します。


### Grad-CAMによるヒートマップ作成

学習済みResNet50を利用して指定のクラス毎に各16枚のヒートマップ画像を生成します。ヒートマップの作成には以下のpythonスクリプトを実行します。スクリプト実行後、カレントディレクトリにheatmap-<クラス名>.jpg形式でヒートマップ画像が生成されます。

実行例：

```
python resnet50_gradcam.py --restore-from resnet50_classification_out.h5 --target-data-dir data/val_images --target-classes C1,L3
```

- --restore-from：学習済みモデルの重みファイルを指定します。ファイルのパスは、config.pyで指定したRESULT_DIRからの相対パスになります。
- --target-data-dir：ヒートマップを作成したい各クラスの画像が格納されたルートフォルダを指定します。
- --target-classes：ヒートマップを作成したいクラスをカンマ区切りで指定します。


