#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_engine.py

カット選定に使う「魅力スコア」を計算するエンジン。

各セグメントに以下のスコアを付け、重み付き合算で「総合スコア」を算出します:

    - 音量スコア    (audio)     : 歓声・拍手などの音量の大きさ
    - 動きスコア    (motion)    : フレーム間の変化量（動きの激しさ）
    - 笑顔スコア    (smile)     : 笑顔が映っているフレームの割合     [要 OpenCV]
    - 顔サイズスコア (face_size) : 顔が大きく映っている度合い         [要 OpenCV]
    - 文脈スコア    (context)   : ポジティブワードの含有率            [要 Whisper]

【重みの設定方法】
    presets/variants.yaml の weights セクションで各スコアの重みを設定します。

    例:
        weights:
          w_audio:     0.1   ← 音量スコアの重み
          w_smile:     0.5   ← 笑顔スコアの重み（笑顔重視バリアント）
          w_face_size: 0.2
          w_motion:    0.1
          w_context:   0.1

    重みは合計が 1.0 になるように設定すると分かりやすいですが、
    合計が 1.0 以外でも動作します（スコアの相対的な大小だけが重要）。

【将来の拡張ポイント】
    - 拍手・笑い声分類: 音声スペクトルで「拍手」「笑い声」を識別
    - 自動重み最適化: review.csv の蓄積データから機械学習で最適な重みを学習
    - MediaPipe 連携: より高精度な顔・表情検出への換装
"""

import os
import numpy as np

# ============================================================
# オプション依存ライブラリの読み込み
# インストールしていなくても動くように try/except で囲む
# ============================================================

# ── OpenCV (笑顔・顔サイズ・顔中心クロップに使用) ──
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[score_engine] ℹ OpenCV が見つかりません。"
          " 笑顔・顔サイズスコアは 0.0 になります。")
    print("               インストール: pip install opencv-python-headless")

# ── Whisper (文脈スコアに使用) ──
try:
    import whisper as _whisper_module
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    # Whisper は重い依存なので、なくても警告だけ出してスキップ
    # (variants.yaml の use_whisper: true のときだけ必要)

# ============================================================
# ポジティブワードリスト（文脈スコアで使用）
#
# ここに単語を追加すると加点対象が増えます。
# 学校行事に合わせて自由にカスタマイズしてください。
# ============================================================

POSITIVE_WORDS = [
    # 感謝・喜び
    "ありがとう", "ありがとうございます", "感謝",
    # 頑張り・努力
    "頑張った", "頑張ります", "頑張れ", "全力", "一生懸命",
    # 感情
    "最高", "楽しい", "楽しかった", "嬉しい", "嬉しかった",
    "感動", "素晴らしい", "すごい", "最幸",
    # 達成
    "やった", "できた", "成功", "優勝", "1位", "一位",
    # つながり
    "一緒", "みんな", "仲間", "チーム", "クラス",
    # 笑い
    "笑", "大好き", "好き",
]

# ============================================================
# OpenCV カスケード分類器のキャッシュ
# (一度ロードしたら使い回す: モジュールレベルの変数に保存)
# ============================================================

_face_cascade  = None   # 顔検出用
_smile_cascade = None   # 笑顔検出用


def _get_face_cascade():
    """
    顔検出用 Haar Cascade 分類器を取得する内部関数。
    初回呼び出し時だけファイルをロードし、以後はキャッシュを返す。
    """
    global _face_cascade
    if _face_cascade is None and OPENCV_AVAILABLE:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _get_smile_cascade():
    """
    笑顔検出用 Haar Cascade 分類器を取得する内部関数。
    """
    global _smile_cascade
    if _smile_cascade is None and OPENCV_AVAILABLE:
        path = cv2.data.haarcascades + "haarcascade_smile.xml"
        _smile_cascade = cv2.CascadeClassifier(path)
    return _smile_cascade


# ============================================================
# Whisper モデルのキャッシュ
# ============================================================

_whisper_model = None


def load_whisper_model(model_size="tiny"):
    """
    Whisper モデルをロードしてキャッシュする関数。

    初回実行時はモデルファイルをダウンロードするため時間がかかります。
    2回目以降はキャッシュから返すので即座に返ります。

    引数:
        model_size (str): モデルのサイズ
            "tiny"   → 最速・精度低め (39MB)
            "base"   → やや高精度 (74MB)
            "small"  → 高精度 (244MB)
            "medium" → さらに高精度 (769MB)
            "large"  → 最高精度 (1.5GB) ← RAM 8GB 以上推奨

    戻り値:
        Whisper モデルオブジェクト、または None（Whisper 未インストール時）
    """
    global _whisper_model
    if not WHISPER_AVAILABLE:
        print("  [Whisper] openai-whisper がインストールされていません。")
        print("            pip install openai-whisper でインストールしてください。")
        return None
    if _whisper_model is None:
        print(f"  [Whisper] モデル '{model_size}' をロード中...")
        print("            (初回は数分かかる場合があります)")
        _whisper_model = _whisper_module.load_model(model_size)
        print(f"  [Whisper] ロード完了")
    return _whisper_model


# ============================================================
# 個別スコア計算関数
# ============================================================

def get_audio_score(clip):
    """
    音量スコア (RMS) を計算する関数。

    RMS = Root Mean Square（二乗平均平方根）で音量の大きさを数値化します。
    歓声・拍手・笑い声などの「盛り上がり音」が大きいほど高スコアになります。

    引数:
        clip: MoviePy のクリップオブジェクト

    戻り値:
        float: 音量スコア（一般的に 0.001〜0.1 程度の値）
    """
    if clip.audio is None:
        return 0.0
    try:
        arr = clip.audio.to_soundarray(fps=22050)
        # ステレオの場合はモノラルに変換（チャンネル方向の平均）
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return float(np.sqrt(np.mean(arr ** 2)))
    except Exception:
        return 0.0


def get_motion_score(clip, n_samples=5):
    """
    動きスコア（フレーム差分）を計算する関数。

    連続するフレームを比較して「ピクセル値がどれだけ変化したか」を測定します。
    静止した場面はスコアが低く、激しく動く場面（リレー・演技など）は高スコアです。

    【改善点: 旧バージョンとの違い】
        旧: 3点サンプリング（クリップの 25%・50%・75%）
        新: n_samples 点を均等サンプリング（デフォルト 5 点）
            → より安定したスコアが得られる

    引数:
        clip: MoviePy のクリップ
        n_samples (int): サンプリングするフレーム数（多いほど正確・遅い）

    戻り値:
        float: 動きスコア（ピクセル値の平均差分。一般的に 0〜50 程度）
    """
    try:
        duration = clip.duration
        # クリップを均等分割した時刻でサンプリング
        times = [duration * i / (n_samples + 1) for i in range(1, n_samples + 1)]

        frames = []
        for t in times:
            t = min(t, duration - 0.01)
            frame = clip.get_frame(t)
            # R/G/B の平均でグレースケール化（計算コスト 1/3 に削減）
            gray = frame.astype(float).mean(axis=2)
            frames.append(gray)

        # 連続フレーム間の平均絶対差分
        diffs = [np.abs(frames[i] - frames[i - 1]).mean()
                 for i in range(1, len(frames))]
        return float(np.mean(diffs)) if diffs else 0.0

    except Exception:
        return 0.0


def get_smile_score(clip, n_samples=5):
    """
    笑顔スコアを計算する関数 [要 OpenCV]。

    【アルゴリズム】
        1. n_samples フレームをサンプリング
        2. 各フレームで Haar Cascade により顔を検出
        3. 検出された顔の領域内で笑顔を検出
        4. 笑顔の顔 / 全検出顔 の比率をスコアとして返す

    【精度について】
        Haar Cascade は軽量だが精度に限界があります。
        より高精度にしたい場合は MediaPipe や dlib への換装を検討してください。

    引数:
        clip: MoviePy のクリップ
        n_samples (int): サンプリングするフレーム数

    戻り値:
        float: 笑顔スコア (0.0〜1.0)。OpenCV がない場合は 0.0。
    """
    if not OPENCV_AVAILABLE:
        return 0.0

    face_cascade  = _get_face_cascade()
    smile_cascade = _get_smile_cascade()

    duration = clip.duration
    times = [duration * i / (n_samples + 1) for i in range(1, n_samples + 1)]

    total_faces = 0
    smile_faces = 0

    for t in times:
        try:
            t = min(t, duration - 0.01)
            frame = clip.get_frame(t).astype(np.uint8)
            # OpenCV は BGR を基本とするが、MoviePy は RGB で返す
            # → グレースケール変換には COLOR_RGB2GRAY を使う
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 顔を検出
            # scaleFactor=1.2: 12% ずつ縮小しながらスキャン（小さい顔も検出）
            # minNeighbors=5: 5回以上検出された領域のみ採用（誤検知を減らす）
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
            )

            for (fx, fy, fw, fh) in faces:
                total_faces += 1
                # 顔の領域（ROI: Region of Interest）を切り出して笑顔を検索
                roi_gray = gray[fy:fy + fh, fx:fx + fw]
                smiles = smile_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
                )
                if len(smiles) > 0:
                    smile_faces += 1

        except Exception:
            continue

    if total_faces == 0:
        return 0.0
    return float(smile_faces / total_faces)


def get_face_size_score(clip, n_samples=5):
    """
    顔サイズスコアを計算する関数 [要 OpenCV]。

    フレーム内で「最も大きい顔」の面積を計算し、
    フレーム全体に対する割合をスコアとして返します。

    顔が大きく映っているほど高スコア → 人物がクローズアップされた
    感情的に訴えかけやすいシーンを優先できます。

    引数:
        clip: MoviePy のクリップ
        n_samples (int): サンプリングするフレーム数

    戻り値:
        float: 顔サイズスコア (0.0〜1.0)。
               0.0 = 顔なし、0.1 = 小さく映る、0.3+ = 大きくクローズアップ
    """
    if not OPENCV_AVAILABLE:
        return 0.0

    face_cascade = _get_face_cascade()
    frame_area = clip.w * clip.h  # フレーム全体の面積（ピクセル数）

    duration = clip.duration
    times = [duration * i / (n_samples + 1) for i in range(1, n_samples + 1)]

    size_scores = []

    for t in times:
        try:
            t = min(t, duration - 0.01)
            frame = clip.get_frame(t).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # 最も大きい顔の面積を使う（複数人いる場合は最も目立つ人を基準）
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                face_area = largest_face[2] * largest_face[3]
                size_scores.append(face_area / frame_area)
            else:
                size_scores.append(0.0)

        except Exception:
            size_scores.append(0.0)

    return float(np.mean(size_scores)) if size_scores else 0.0


def get_context_score(clip, whisper_model, positive_words=None):
    """
    文脈スコアを計算する関数 [要 Whisper]。

    音声を Whisper で日本語文字起こしし、POSITIVE_WORDS に含まれる
    単語が何個登場するかを数えてスコア化します。

    「ありがとう」「最高」「頑張った」などポジティブな発言が多い
    シーンほど高スコアになります。

    【処理時間について】
        3秒クリップ 1本 の文字起こしに数秒〜十数秒かかります。
        セグメント数が多い場合は全体処理時間が長くなるため、
        use_whisper: false にして無効化することもできます。

    引数:
        clip: MoviePy のクリップ
        whisper_model: load_whisper_model() で取得したモデル
        positive_words (list): ポジティブワードのリスト（None でデフォルト使用）

    戻り値:
        float: 文脈スコア (0.0〜1.0)。Whisper 未インストールまたは None の場合は 0.0。
    """
    if whisper_model is None:
        return 0.0
    if clip.audio is None:
        return 0.0
    if positive_words is None:
        positive_words = POSITIVE_WORDS

    try:
        # Whisper は 16kHz モノラルの float32 配列を入力として受け付ける
        audio_arr = clip.audio.to_soundarray(fps=16000)
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.mean(axis=1)  # ステレオ → モノラル
        audio_arr = audio_arr.astype(np.float32)

        # 文字起こし実行
        # language="ja" で日本語に固定することで精度が上がる
        # fp16=False は CPU 実行時の警告を抑止するため
        result = whisper_model.transcribe(audio_arr, language="ja", fp16=False)
        text = result.get("text", "")

        # ポジティブワードの一致数をカウント
        hits = sum(1 for w in positive_words if w in text)
        # 一致数 / 総単語数 で 0〜1 に正規化（最大 1.0 にクランプ）
        return min(float(hits) / max(len(positive_words), 1), 1.0)

    except Exception as e:
        return 0.0


# ============================================================
# 顔中心クロップ関連
# ============================================================

def get_face_crop_center(clip, t_sample=None):
    """
    クリップの指定フレームから「最大の顔の中心座標」を返す関数 [要 OpenCV]。

    顔が検出できない場合はフレームの中心座標を返します。
    この座標を crop_to_vertical_face() に渡すことで、
    人物が画面の中央に来るように動画を切り抜けます。

    引数:
        clip: MoviePy のクリップ
        t_sample (float): サンプリングする時刻（秒）。None でクリップ中央。

    戻り値:
        (float, float): (x_center, y_center) のタプル（ピクセル単位）
    """
    # OpenCV がない場合はフレーム中央を返す（通常の中央クロップと同じ動作）
    if not OPENCV_AVAILABLE:
        return clip.w / 2, clip.h / 2

    face_cascade = _get_face_cascade()

    if t_sample is None:
        t_sample = clip.duration / 2  # クリップの中間時点を使う
    t_sample = min(t_sample, clip.duration - 0.01)

    try:
        frame = clip.get_frame(t_sample).astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            # 顔が見つからない → フレーム中央にフォールバック
            return clip.w / 2, clip.h / 2

        # 最も大きい顔の中心座標を計算
        largest = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest
        return float(fx + fw / 2), float(fy + fh / 2)

    except Exception:
        return clip.w / 2, clip.h / 2


def crop_to_vertical_face(clip, width=1080, height=1920):
    """
    顔の位置を基準にして縦型 (9:16) にクロップする関数 [要 OpenCV]。

    【通常の中央クロップとの違い】
        通常版: 常にフレームの幾何学的中央を基準にクロップ
        この関数: 最も大きい顔の中心を基準にクロップ

        → 人物が端に寄っている横長動画でも、顔が切れずに縦型に変換できる

    顔が検出できない場合は通常の中央クロップと同じ動作をします。

    引数:
        clip: MoviePy のクリップ
        width (int) : 出力幅 (px)  デフォルト: 1080
        height (int): 出力高さ (px) デフォルト: 1920

    戻り値:
        クロップ・リサイズ済みのクリップ
    """
    # 顔の中心座標を取得（OpenCV 未インストール時はフレーム中央）
    face_cx, face_cy = get_face_crop_center(clip)

    original_w = clip.w
    original_h = clip.h
    target_ratio = width / height  # 9:16 ≈ 0.5625

    if (original_w / original_h) > target_ratio:
        # ── 横長動画: 左右をカット ──
        new_w = int(original_h * target_ratio)
        # 顔の x 座標を中心に。ただし端からはみ出ないようにクランプする。
        cx = float(np.clip(face_cx, new_w / 2, original_w - new_w / 2))
        clipped = clip.crop(
            x_center=cx,
            y_center=float(original_h / 2),
            width=new_w,
            height=original_h,
        )
    else:
        # ── 縦長・正方形: 上下をカット ──
        new_h = int(original_w / target_ratio)
        cy = float(np.clip(face_cy, new_h / 2, original_h - new_h / 2))
        clipped = clip.crop(
            x_center=float(original_w / 2),
            y_center=cy,
            width=original_w,
            height=new_h,
        )

    return clipped.resize((width, height))


# ============================================================
# 全スコアをまとめて計算する関数
# ============================================================

def compute_all_scores(clip, whisper_model=None):
    """
    1 つのクリップに対して全スコアを計算し、辞書にまとめて返す関数。

    この関数を呼ぶだけで 5 種類のスコアが全て得られます。
    OpenCV や Whisper がインストールされていない場合、
    対応するスコアは自動的に 0.0 になります。

    引数:
        clip: MoviePy のクリップ
        whisper_model: load_whisper_model() の戻り値（None でスキップ）

    戻り値:
        dict: {
            "audio"    : float,  # 音量スコア
            "motion"   : float,  # 動きスコア
            "smile"    : float,  # 笑顔スコア     (0.0 if no OpenCV)
            "face_size": float,  # 顔サイズスコア  (0.0 if no OpenCV)
            "context"  : float,  # 文脈スコア     (0.0 if no Whisper)
        }
    """
    return {
        "audio":     get_audio_score(clip),
        "motion":    get_motion_score(clip),
        "smile":     get_smile_score(clip),
        "face_size": get_face_size_score(clip),
        "context":   get_context_score(clip, whisper_model),
    }


# ============================================================
# スコアの正規化と重み付き合算
# ============================================================

def _normalize_list(values):
    """
    数値リストを 0〜1 の範囲に Min-Max 正規化する内部ヘルパー関数。

    【なぜ正規化が必要か】
        音量スコアは 0.001〜0.1 程度の小さな値。
        動きスコアは 0〜50 程度の大きな値。
        → そのまま重み付けすると、大きな値のスコアが支配的になってしまう。
        → 全て 0〜1 に揃えてから重みを掛けることで、公平な比較ができる。

    全ての値が同じ場合は 0.5 を返す（差がないため順位をつけられない）。
    """
    arr = np.array(values, dtype=float)
    min_v, max_v = arr.min(), arr.max()
    if max_v == min_v:
        return [0.5] * len(values)
    return list((arr - min_v) / (max_v - min_v))


def normalize_all_scores(segments):
    """
    全セグメントのスコアを各指標ごとに 0〜1 に正規化する関数。

    全セグメントのスコアをまとめて見て、各指標の最大・最小を基準に正規化します。
    正規化後のスコアは "norm_scores" キーにセットされます。

    引数:
        segments (list[dict]): 各要素に "scores" キーがある辞書のリスト

    戻り値:
        list[dict]: 各要素に "norm_scores" キーを追加したリスト（in-place 変更）
    """
    if not segments:
        return segments

    score_keys = ["audio", "motion", "smile", "face_size", "context"]

    for key in score_keys:
        # 全セグメントの該当スコアを集める
        raw_values = [s["scores"].get(key, 0.0) for s in segments]
        normalized = _normalize_list(raw_values)

        # 各セグメントに正規化済みスコアをセット
        for seg, norm_val in zip(segments, normalized):
            if "norm_scores" not in seg:
                seg["norm_scores"] = {}
            seg["norm_scores"][key] = norm_val

    return segments


# デフォルトの重み設定
# variants.yaml に weights が書かれていない場合はこれを使う
# （音量スコアのみを使う: 旧バージョンと同じ動作）
DEFAULT_WEIGHTS = {
    "w_audio":     1.0,
    "w_smile":     0.0,
    "w_face_size": 0.0,
    "w_motion":    0.0,
    "w_context":   0.0,
}


def compute_total_score(norm_scores, weights):
    """
    正規化済みスコアに重みを掛けて総合スコアを計算する関数。

    【計算式】
        total_score =
            w_audio     × audio_score
          + w_smile     × smile_score
          + w_face_size × face_size_score
          + w_motion    × motion_score
          + w_context   × context_score

    重みの設定例（variants.yaml）:
        笑顔重視バリアント: w_smile=0.5, w_audio=0.2, ...
        音量重視バリアント: w_audio=0.6, w_smile=0.1, ...
        文脈重視バリアント: w_context=0.5, w_audio=0.2, ...

    引数:
        norm_scores (dict): normalize_all_scores() で計算した正規化スコア
        weights (dict)    : 各スコアの重み（variants.yaml の weights セクション）

    戻り値:
        float: 総合スコア
    """
    return float(
        weights.get("w_audio",     DEFAULT_WEIGHTS["w_audio"])     * norm_scores.get("audio",     0.0)
        + weights.get("w_smile",     DEFAULT_WEIGHTS["w_smile"])     * norm_scores.get("smile",     0.0)
        + weights.get("w_face_size", DEFAULT_WEIGHTS["w_face_size"]) * norm_scores.get("face_size", 0.0)
        + weights.get("w_motion",    DEFAULT_WEIGHTS["w_motion"])    * norm_scores.get("motion",    0.0)
        + weights.get("w_context",   DEFAULT_WEIGHTS["w_context"])   * norm_scores.get("context",   0.0)
    )
