# 学校行事リールメーカー

学校行事の素材動画から **Instagram リール (9:16 縦型)** を自動生成する Python スクリプトです。

- `input/` フォルダに動画を入れるだけ
- **音量が大きいシーン**（歓声・盛り上がり場面）を自動で優先選択
- 横長動画を **1080×1920 (縦型)** に自動クロップ
- 選んだシーンをつなぎ合わせて `output/reel.mp4` を生成

---

## 必要なもの

| 必要なもの | バージョン目安 |
|-----------|--------------|
| Python    | 3.8 以上      |
| ffmpeg    | 4.x 以上      |

---

## セットアップ手順

### 1. ffmpeg をインストールする

動画を処理するために ffmpeg が必要です。

**Mac (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
1. [ffmpeg 公式サイト](https://ffmpeg.org/download.html) からダウンロード
2. 解凍して `bin/` フォルダを環境変数 PATH に追加する

インストールできたか確認:
```bash
ffmpeg -version
```

> **ヒント:** `imageio-ffmpeg` パッケージ（requirements.txt に含まれる）が ffmpeg のバイナリを
> 自動で用意してくれる場合があります。上記のインストールが難しければまず試してみてください。

---

### 2. Python ライブラリをインストールする

```bash
pip install -r requirements.txt
```

仮想環境 (venv) を使う場合（推奨）:
```bash
# 仮想環境を作成
python -m venv venv

# 有効化 (Mac / Linux)
source venv/bin/activate

# 有効化 (Windows)
venv\Scripts\activate

# ライブラリをインストール
pip install -r requirements.txt
```

---

## 使い方

### 1. 素材動画を `input/` に入れる

```
school-reel-maker/
└── input/
    ├── sports_day_01.mp4
    ├── culture_festival.mov
    └── graduation.mp4
```

対応している拡張子: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`

### 2. スクリプトを実行する

```bash
python make_reel.py
```

実行中は以下のような進捗ログが表示されます:

```
=======================================================
  学校行事リールメーカー
=======================================================

[✓] 3 本の動画を見つけました:
    - sports_day_01.mp4
    - culture_festival.mov
    - graduation.mp4

[1/3] 動画を分析しています...
  分析中: sports_day_01.mp4
    長さ: 45.3秒 / 解像度: 1920×1080
    → 15 セグメントを分析しました
  ...

[2/3] 上位シーンを選択しています... (目標: 60秒)
  → 20 シーンを選択 (合計: 60.0秒)

[3/3] リール動画を生成しています...
  ...
  書き出し中: output/reel.mp4

=======================================================
  [完了] リール動画が生成されました!
  出力ファイル: output/reel.mp4
=======================================================
```

### 3. 生成された動画を確認する

`output/reel.mp4` が生成されます。

---

## カスタマイズ

`make_reel.py` の上部にある **設定エリア** の数値を変えることで動作を調整できます。

```python
# リールの目標時間 (秒) ← ここを変える
TARGET_DURATION = 60

# 動画を何秒ごとに区切って分析するか
SEGMENT_LENGTH = 3

# 出力解像度 (Instagram 推奨サイズ)
OUTPUT_WIDTH  = 1080
OUTPUT_HEIGHT = 1920

# クロスフェードの長さ (0 にするとブツ切り)
CROSSFADE_DURATION = 0.3
```

| 設定変数 | デフォルト | 説明 |
|---------|---------|------|
| `TARGET_DURATION` | `60` | 生成するリールの目標時間 (秒)。最大 90 秒。 |
| `SEGMENT_LENGTH` | `3` | 動画を分割する単位 (秒)。短くすると細かく選べる。 |
| `MIN_CLIP_DURATION` | `2` | これより短い区間は無視する (秒)。 |
| `OUTPUT_WIDTH` | `1080` | 出力幅 (px)。 |
| `OUTPUT_HEIGHT` | `1920` | 出力高さ (px)。 |
| `OUTPUT_FPS` | `30` | 出力フレームレート。 |
| `CROSSFADE_DURATION` | `0.3` | 場面転換のフェード時間 (秒)。 |

---

## ファイル構成

```
school-reel-maker/
├── make_reel.py       # メインスクリプト（ここを実行する）
├── requirements.txt   # インストールが必要な Python ライブラリ一覧
├── README.md          # このファイル
├── .gitignore         # Git で除外するファイルの設定
├── input/             # ← 素材動画をここに入れる
└── output/            # ← reel.mp4 がここに生成される
```

---

## 仕組みの説明

1. **動画の収集**: `input/` フォルダから対応する拡張子の動画をすべて読み込む
2. **セグメント分割**: 各動画を `SEGMENT_LENGTH` 秒ごとの区間に分割する
3. **音量測定**: 各区間の音声データから RMS（二乗平均平方根）を計算する
   → RMS 値が大きいほど盛り上がっているシーン
4. **シーン選択**: 全区間を音量の大きい順に並べ、`TARGET_DURATION` 秒に達するまで選ぶ
5. **縦型変換**: 横長動画は中央を切り抜いて 1080×1920 にクロップ・リサイズする
6. **書き出し**: 選んだシーンをクロスフェードでつなぎ `output/reel.mp4` に出力する

---

## トラブルシューティング

**`No module named 'moviepy'` エラー:**
```bash
pip install -r requirements.txt
```

**`ffmpeg` が見つからないエラー:**
- ffmpeg をインストールして PATH を通してください（セットアップ手順 1 を参照）

**処理が遅い / フリーズしているように見える:**
- 動画の数や長さによっては処理に数分かかります。進捗バーが表示されている間はお待ちください。

**生成された動画の画質が悪い:**
- `make_reel.py` の `bitrate` を `"8000k"` などに増やしてみてください。
- ファイルサイズが大きくなりますが画質が上がります。

**動画が 1 本も選ばれない:**
- 動画が `MIN_CLIP_DURATION`（デフォルト: 2 秒）より短い場合はスキップされます。
- ファイルが破損していないか確認してください。
