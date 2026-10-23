# AI の追加学習（RIFE 4.25 の鍛え直し）

拡張機能が使っている AI（RIFE 4.25）を、苦手な動き（急な拡大・迫る動き、繰り返し模様の上の大きな動き）を多く含む合成映像で追加学習させるための道具です。AI の大きさは変わらないので、学習後も補間の重さは同じです。

学習データはその場で自動生成する合成映像だけで、動画や画像のファイルは使いません（著作権の問題がありません）。

## 必要なもの

- NVIDIA の GPU（RTX 3080 で 1 回あたり数時間の見込み）
- 空きディスク 5 GB 程度（PyTorch のインストール用）

## 手順（Windows）

### 1. Python を入れる（初回だけ）

https://www.python.org/downloads/ から Python 3.12 をダウンロードしてインストールします。最初の画面で「Add python.exe to PATH」にチェックを入れてください。

### 2. このフォルダを用意する

GitHub のリポジトリのページで「Code」→「Download ZIP」を押し、展開します。中の `training` フォルダを使います。

### 3. PyTorch などを入れる（初回だけ、ダウンロード約 3 GB）

コマンドプロンプトを開き、次を 1 行ずつ実行します。

```
py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
py -m pip install numpy pillow
```

GPU が使えるか確認します（`True` と GPU の名前が出れば OK）。

```
py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. 学習する

`training` フォルダに移動して実行します（`cd` の後ろは展開した場所に合わせてください）。

```
cd C:\Users\あなたの名前\Downloads\nurunuru-frames-ai-main\training
py train.py
```

- 最初に元の AI の重み（約 25 MB）を自動でダウンロードします。
- 少しすると「about ... s per step: ... take about ... h」と、全体のおおよその所要時間が表示されます。
- 学習中は GPU をほぼ使い切ります。PC がスリープしないように設定しておいてください。
- 途中で止めても（Ctrl+C や PC の再起動）、同じコマンドをもう一度実行すれば続きから再開します。

### 5. 結果を送る

終わると `out` フォルダに次のファイルができます。

- `rife-v4.25-ft.pkl`（約 25 MB）：鍛え直した AI。学習中にいちばん成績のよかったもの
- `log.txt`：学習の記録

この 2 つを送ってください。拡張機能に入れる前に、学習に使っていない映像で元の AI と比べ、良くなっている場合だけ採用します。

## 中身

| ファイル | 内容 |
|---|---|
| `model.py` | RIFE 4.25 の構造（拡張機能の実装と同じ計算） |
| `synth.py` | 合成映像の生成（背景と物体、さまざまな模様、移動・拡大縮小・回転） |
| `train.py` | 追加学習。元の重みから離れすぎないようにする項つき |

主な設定（`py train.py --help`）：`--steps`（学習の回数、既定 20000）、`--batch`（一度に使う組の数、既定 16。GPU のメモリが足りなければ 8 に）、`--lr`（学習の強さ）、`--keep`（元の重みに留める強さ）、`--real`（自由に使える動画から書き出した連番 PNG のフォルダを、合成映像に混ぜる場合）。
