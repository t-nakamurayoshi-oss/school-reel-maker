# 学校行事リールメーカー

学校行事の素材動画から **Instagram リール (9:16 縦型)** を自動生成する Python スクリプトです。

- `input/` フォルダに動画を入れるだけ
- **音量が大きいシーン**（歓声・盛り上がり場面）を自動で優先選択
- 横長動画を **1080×1920 (縦型)** に自動クロップ
- 選んだシーンをつなぎ合わせて `output/reel.mp4` を生成
- **ABテスト機能**: 複数パターンを一括生成 → 投稿 → 結果を記録 → 次回提案

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
├── make_reel.py          # シンプル版: 1本のリールを生成
├── make_reel_ab.py       # ABテスト版: 複数バリアントを一括生成
├── recommend_next.py     # レビュー結果をもとに次回設定を提案
├── requirements.txt      # インストールが必要な Python ライブラリ一覧
├── README.md             # このファイル
├── .gitignore            # Git で除外するファイルの設定
│
├── presets/
│   └── variants.yaml     # ABテスト用バリアント設定ファイル
│
├── input/                # ← 素材動画をここに入れる
│
├── output/
│   ├── reel.mp4          # make_reel.py の出力（シンプル版）
│   ├── candidates/       # ABテスト版リールの出力先
│   │   ├── reel_A.mp4
│   │   ├── reel_B.mp4
│   │   └── reel_C.mp4
│   └── manifests/        # 各バリアントの生成条件 (JSON)
│       ├── reel_A.json
│       ├── reel_B.json
│       └── reel_C.json
│
└── experiments/
    └── review.csv        # 投稿後の結果を記録するシート
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

---

## ABテストの回し方

複数パターンのリールを生成・比較することで、どんな編集スタイルが
フォロワーに刺さるかを定量的に把握できます。

### 全体の流れ（PDCAサイクル）

```
[生成] make_reel_ab.py
   ↓ 複数バリアントを出力
[投稿] Instagram に A/B/C を投稿
   ↓ 数日〜1週間後に数値を確認
[記録] experiments/review.csv に結果を記入
   ↓
[提案] recommend_next.py でおすすめ設定を確認
   ↓
[改善] variants.yaml を調整して次のサイクルへ
```

---

### Step 1: バリアントを設定する

`presets/variants.yaml` を開いて、試したいパターンを設定します。

```yaml
variants:
  - name: "A"
    label: "音量重視"
    hook_mode: "loud"    # 最も音量が大きいシーンを冒頭に
    clip_sec: 3          # 1クリップ 3 秒
    target_sec: 60       # 合計 60 秒
    bgm: null
    crossfade_sec: 0.3

  - name: "B"
    label: "動き重視"
    hook_mode: "motion"  # 最も動きが激しいシーンを冒頭に
    clip_sec: 4
    target_sec: 60
    bgm: null
    crossfade_sec: 0.5

  - name: "C"
    label: "テンポ速め"
    hook_mode: "loud"
    clip_sec: 2          # クリップを短くしてテンポアップ
    target_sec: 45
    bgm: null
    crossfade_sec: 0.1
```

| パラメータ | 説明 |
|-----------|------|
| `hook_mode: loud` | 歓声・拍手など音量が最大のシーンを冒頭に配置 |
| `hook_mode: motion` | 走る・踊るなど動きが最大のシーンを冒頭に配置 |
| `clip_sec` | 1クリップの長さ。短いほど軽快なテンポに |
| `target_sec` | リール全体の長さ。Instagram は最大 90 秒 |
| `crossfade_sec` | シーン転換のフェード秒数。0 でブツ切り |
| `bgm` | BGM ファイルのパス（null でなし）|

---

### Step 2: リールを一括生成する

```bash
python make_reel_ab.py
```

`output/candidates/` に `reel_A.mp4`、`reel_B.mp4`、`reel_C.mp4` が生成されます。
同時に `output/manifests/reel_A.json` などに生成条件が記録されます。

---

### Step 3: Instagram に投稿する

3 本を**同じタイミング・同じキャプション**で投稿するのが理想ですが、
難しければ同じ日の異なる時間帯に投稿しても構いません。

> **ポイント**: 投稿のタイミングや曜日が揃っているほど、純粋に「動画の違い」を
> 比較しやすくなります。

---

### Step 4: 結果を review.csv に記録する

投稿の 3〜7 日後に Instagram のインサイト（分析ツール）を確認し、
`experiments/review.csv` に数値を記入します。

```csv
date,variant,filename,views,saves,avg_watch_sec,notes
2024-04-01,A,reel_A.mp4,1500,80,45.2,始業式の動画
2024-04-01,B,reel_B.mp4,1200,55,38.7,始業式の動画
2024-04-01,C,reel_C.mp4,2000,70,28.1,始業式の動画
```

| 列名 | 内容 | Instagram での確認場所 |
|------|------|----------------------|
| `views` | 再生数 | インサイト → リーチ数 |
| `saves` | 保存数（最重要）| インサイト → 保存数 |
| `avg_watch_sec` | 平均視聴時間（秒）| インサイト → 平均再生時間 |
| `notes` | メモ（投稿日・行事名など）| ─ |

---

### Step 5: 次回おすすめ設定を確認する

```bash
python recommend_next.py
```

結果例:
```
┌─ バリアント評価結果 ────────────────────────────────┐
  順位  バリアント  保存数    平均視聴(秒)  再生数    総合スコア  データ数
  ────────────────────────────────────────────────────
  ★1   A           80        45.2          1500      0.823       1件
    2   C           70        28.1          2000      0.612       1件
    3   B           55        38.7          1200      0.401       1件
└─────────────────────────────────────────────────────┘

★ 最もパフォーマンスが良かったバリアント: A
   このバリアントの設定:
     hook_mode    : loud
     clip_sec     : 3 秒
     target_sec   : 60 秒
     crossfade_sec: 0.3 秒
```

---

### Step 6: variants.yaml を改善して次のサイクルへ

表示されたおすすめ設定を参考に、`presets/variants.yaml` を調整します。

```yaml
# バリアント A が良かった → A をベースに微調整
variants:
  - name: "A"
    label: "前回ベスト＋clip_sec短縮"
    hook_mode: "loud"
    clip_sec: 2        # 3 → 2 に短縮して試す
    target_sec: 60
    ...

  - name: "B"
    label: "前回ベスト＋BGMあり"
    hook_mode: "loud"
    clip_sec: 3
    target_sec: 60
    bgm: "assets/bgm.mp3"   # BGM を追加して試す
    ...
```

> **データが蓄積されるほど精度が上がります。**
> 3〜5 回サイクルを回すと「この学校のフォロワーには○○が刺さる」
> という傾向が見えてきます。

---

---

## 重みを調整して編集方針を変える方法

`presets/variants.yaml` の `weights` セクションを編集することで、
カット選定の優先基準をバリアントごとに変えられます。

### スコアの種類と意味

| スコア名 | キー | 内容 | 必要なライブラリ |
|---------|------|------|----------------|
| 音量スコア | `w_audio` | 歓声・拍手など音量の大きさ | なし |
| 動きスコア | `w_motion` | フレーム間の動きの激しさ | なし |
| 笑顔スコア | `w_smile` | 笑顔が映っているフレームの割合 | OpenCV |
| 顔サイズスコア | `w_face_size` | 顔が画面内で大きく映る度合い | OpenCV |
| 文脈スコア | `w_context` | ポジティブワードの含有率 | Whisper |

### 総合スコアの計算式

```
total_score =
    w_audio     × 音量スコア(正規化済み)
  + w_smile     × 笑顔スコア(正規化済み)
  + w_face_size × 顔サイズスコア(正規化済み)
  + w_motion    × 動きスコア(正規化済み)
  + w_context   × 文脈スコア(正規化済み)
```

> 各スコアは全セグメントにわたって 0〜1 に正規化されてから重みが掛かります。
> そのため、異なるスケール（音量 0.01 と動きスコア 20）が公平に比較されます。

### 設定例

```yaml
# 笑顔重視: 笑顔が多く映るシーンを優先
weights:
  w_audio:     0.1
  w_smile:     0.5   ← 笑顔スコアを最大に
  w_face_size: 0.2
  w_motion:    0.1
  w_context:   0.1

# 音量重視: 歓声・盛り上がりを優先
weights:
  w_audio:     0.6   ← 音量スコアを最大に
  w_smile:     0.1
  w_face_size: 0.1
  w_motion:    0.2
  w_context:   0.0

# 動き重視: 激しい動作シーンを優先
weights:
  w_audio:     0.2
  w_smile:     0.1
  w_face_size: 0.1
  w_motion:    0.6   ← 動きスコアを最大に
  w_context:   0.0
```

### OpenCV のインストール（笑顔・顔サイズスコアを有効にする）

```bash
pip install opencv-python-headless
```

インストール後は `w_smile` や `w_face_size` の重みを上げると効果が出ます。
OpenCV がない場合、これらのスコアは自動的に 0.0 として扱われます。

### Whisper のインストール（文脈スコアを有効にする）

```bash
pip install openai-whisper
```

インストール後、`presets/variants.yaml` の以下の設定を変更します:

```yaml
config:
  use_whisper: true        # false → true に変更
  whisper_model: "tiny"   # tiny / base / small / medium から選択
```

> Whisper を有効にすると各セグメントの文字起こしを行うため、
> 処理時間が大幅に増加します（3秒クリップ 1 本あたり数秒〜十数秒）。

### 将来の拡張について

`score_engine.py` は拡張を想定した設計になっています:

- **新しいスコアを追加する**: `score_engine.py` に関数を追加し、
  `compute_all_scores()` に組み込む
- **重みの自動最適化**: `experiments/review.csv` の蓄積データから
  `recommend_next.py` に機械学習ロジックを追加する
- **MediaPipe 対応**: `get_smile_score()` / `get_face_size_score()` の
  内部実装を Haar Cascade から MediaPipe に差し替える（高精度化）

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

**`No module named 'yaml'` エラー:**
```bash
pip install pyyaml
```

**`make_reel_ab.py` で `variants.yaml が見つかりません` と表示される:**
- `presets/variants.yaml` が存在するか確認してください。
- リポジトリに含まれているので、git clone 後は自動で作成されています。

**`recommend_next.py` で「有効なデータがありません」と表示される:**
- `experiments/review.csv` の `views` と `saves` が `0` の行は除外されます。
- 投稿後に実際の数値を記入してから再実行してください。
