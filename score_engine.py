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
    - 手ブレスコア  (shake)     : 映像の安定性（1.0=安定, 0.0=激しいブレ）
    - 傾きスコア    (tilt)      : 映像の水平度（1.0=水平, 0.0=大きく傾いている）[要 OpenCV]

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
# 手ブレ・傾き補正の閾値設定
# ここを変えると補正が適用される範囲が変わります
# ============================================================

# 手ブレ: ジッター量 (px) の閾値
# raw_jitter がこの値より小さい: 安定（補正不要）
SHAKE_STABLE_THRESHOLD = 2.0
# raw_jitter がこの値より大きい: 重度（補正しない。スコアで自然に除外）
SHAKE_SEVERE_THRESHOLD = 8.0

# 傾き: 角度 (degree) の閾値
# |angle| がこの値より小さい: ほぼ水平（補正不要）
TILT_IGNORE_DEG = 0.5
# |angle| がこの値以下: 軽度（自動回転補正を適用）
# |angle| がこの値より大きい: 重度（補正しない。スコアで自然に除外）
TILT_CORRECT_MAX = 5.0

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
# 手ブレ検出・補正
# ============================================================

def _estimate_frame_shift(frame1_gray, frame2_gray):
    """
    位相相関（Phase Correlation）を使って 2 フレーム間のシフト量を推定する関数。

    【原理】
        フーリエ変換すると、画像の「平行移動」は周波数空間での位相差として現れる。
        位相差だけを取り出して逆変換すると、シフト量の位置に極大値ができる。
        この極大値の座標 = フレーム間のシフト量 (dx, dy)。

    OpenCV 不要。NumPy のみで動作します。

    引数:
        frame1_gray: グレースケールの NumPy 配列（float型、ダウンサンプリング済み）
        frame2_gray: 同上

    戻り値:
        (float, float): (dx, dy) ピクセル単位のシフト量
    """
    f1 = np.fft.fft2(frame1_gray)
    f2 = np.fft.fft2(frame2_gray)

    # クロスパワースペクトル（振幅を 1 に正規化して位相差だけを取り出す）
    cross = f1 * np.conj(f2)
    cross /= (np.abs(cross) + 1e-8)

    # 逆フーリエ変換でシフト量のピークを探す
    corr = np.abs(np.fft.ifft2(cross))
    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    # ラップアラウンド補正（負のシフトも正しく処理する）
    h, w = frame1_gray.shape
    if y > h // 2:
        y -= h
    if x > w // 2:
        x -= w

    return float(x), float(y)


def get_shake_info(clip, n_samples=6):
    """
    手ブレの度合いを検出し、(shake_score, raw_jitter) を返す関数。

    【アルゴリズム】
        1. n_samples フレームをサンプリング
        2. 隣接フレーム間のシフト量 (dx, dy) を位相相関で推定
        3. シフト量の「変化率」（2 階差分）の標準偏差 = ジッター量
        4. ジッターが大きいほど手ブレが激しい

    【パン（意図的な動き）と手ブレの違い】
        パン: シフトが一定方向に続く → 変化率（加速度）は小さい
        手ブレ: シフトが不規則に変化 → 変化率（加速度）が大きい
        → 2 階差分を使うことでパンは無視され、手ブレだけを検出できる

    OpenCV 不要。NumPy のみで動作します。

    引数:
        clip     : MoviePy のクリップ
        n_samples: サンプリングするフレーム数（多いほど精度↑・処理↓）

    戻り値:
        (float, float): (shake_score, raw_jitter)
            shake_score: 0.0〜1.0。1.0 = 安定、0.0 = 激しい手ブレ
            raw_jitter : 生のジッター量（ピクセル単位）。補正閾値との比較に使用。
    """
    duration = clip.duration
    if duration < 0.5:
        return 1.0, 0.0  # 短すぎる → 安定と見なす

    try:
        times = np.linspace(0.05, duration * 0.95, n_samples + 1)

        # フレームを取得して 1/4 解像度にダウンサンプリング（FFT を高速化）
        frames = []
        for t in times:
            frame = clip.get_frame(float(min(t, duration - 0.01)))
            gray = frame.astype(float).mean(axis=2)
            frames.append(gray[::4, ::4])  # 縦横それぞれ 1/4 → 面積 1/16

        # 隣接フレーム間のシフト量を推定
        shifts_x, shifts_y = [], []
        for i in range(1, len(frames)):
            dx, dy = _estimate_frame_shift(frames[i - 1], frames[i])
            # ダウンサンプリング補正（1/4 解像度 → 元の解像度換算）
            shifts_x.append(dx * 4)
            shifts_y.append(dy * 4)

        if len(shifts_x) < 2:
            return 1.0, 0.0

        # ジッター = シフト量の 2 階差分（変化率の変化）の標準偏差
        arr_x = np.array(shifts_x)
        arr_y = np.array(shifts_y)
        jitter_x = np.std(np.diff(arr_x))
        jitter_y = np.std(np.diff(arr_y))
        raw_jitter = float(np.sqrt(jitter_x ** 2 + jitter_y ** 2))

        # スコアに変換（SHAKE_SEVERE_THRESHOLD 以上で 0.0 に）
        shake_score = float(max(0.0, 1.0 - raw_jitter / SHAKE_SEVERE_THRESHOLD))
        return shake_score, raw_jitter

    except Exception:
        return 1.0, 0.0  # 計算失敗 → 安定と見なす


def get_tilt_info(clip, n_samples=3):
    """
    動画の水平傾きを検出し、(tilt_score, angle_deg) を返す関数 [要 OpenCV]。

    【アルゴリズム（Hough 変換）】
        1. 複数フレームをサンプリング
        2. Canny エッジ検出で輪郭を抽出
        3. 確率的 Hough 変換で直線を検出
        4. 水平に近い直線（-45°〜45°）の角度の中央値 = 傾き角

    【傾き角の符号】
        angle > 0: 右上がり（反時計回り方向に傾いている）
        angle < 0: 右下がり（時計回り方向に傾いている）
        angle ≈ 0: 水平

    OpenCV がない場合は (1.0, 0.0)（傾きなし）として扱います。

    引数:
        clip     : MoviePy のクリップ
        n_samples: サンプリングするフレーム数（複数の中央値で外れ値に強くなる）

    戻り値:
        (float, float): (tilt_score, angle_deg)
            tilt_score: 0.0〜1.0。1.0 = 水平、0.0 = 大きく傾いている
            angle_deg : 傾き角（度）。correct_tilt() に渡して補正に使用。
    """
    if not OPENCV_AVAILABLE:
        return 1.0, 0.0

    duration = clip.duration
    sample_times = [duration * p for p in [0.25, 0.5, 0.75]][:n_samples]
    all_angles = []

    for t_sample in sample_times:
        try:
            t_sample = min(t_sample, duration - 0.01)
            frame = clip.get_frame(t_sample).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Canny エッジ検出
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # 確率的 Hough 変換で直線を検出
            # minLineLength: フレーム幅の 1/6 以上の直線だけを採用
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=gray.shape[1] // 6,
                maxLineGap=20,
            )
            if lines is None:
                continue

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 水平に近い直線（±45°以内）だけを使う
                if -45.0 <= angle <= 45.0:
                    all_angles.append(angle)

        except Exception:
            continue

    if not all_angles:
        return 1.0, 0.0  # 直線が見つからない → 水平と見なす

    # 中央値を傾き角として採用（外れ値に強い）
    angle_deg = float(np.median(all_angles))

    # 傾きスコア: 15° で 0.0 になるよう線形に減少
    max_tilt_for_zero = 15.0
    tilt_score = float(max(0.0, 1.0 - abs(angle_deg) / max_tilt_for_zero))

    return tilt_score, angle_deg


def stabilize_clip(clip):
    """
    軽度な手ブレを補正する関数（位相相関ベースのシンプルな安定化）。

    【補正の仕組み】
        1. ~5fps でフレームをサンプリングし、フレーム間のシフトを累積推定
        2. 移動平均で軌跡を平滑化（理想的な「滑らかな軌跡」を作る）
        3. 各フレームで「実際の軌跡 - 平滑軌跡」を補正量として画像をシフト

    【制限事項】
        - 平行移動（上下左右）のブレのみ補正。回転ブレは対象外。
        - 補正量が大きいとフレームの縁に黒いアーティファクトが出ることがある。
        - SHAKE_SEVERE_THRESHOLD を超える激しいブレには効果が薄い。

    引数:
        clip: 補正対象の MoviePy クリップ

    戻り値:
        補正済みクリップ（エラー時は元のクリップをそのまま返す）
    """
    MAX_CORRECTION_PX = 20   # 最大補正量 (px)
    SAMPLE_INTERVAL   = 0.2  # サンプリング間隔 (秒)

    duration = clip.duration
    h, w = clip.h, clip.w

    if duration < 0.5:
        return clip

    try:
        # ── 1. フレームをサンプリングして累積シフトを推定 ──
        times = np.arange(0, duration - 0.05, SAMPLE_INTERVAL)
        if len(times) < 3:
            return clip

        prev_gray = (clip.get_frame(float(times[0]))
                     .astype(float).mean(axis=2)[::4, ::4])
        cum_x, cum_y = [0.0], [0.0]

        for t in times[1:]:
            curr_gray = (clip.get_frame(float(min(t, duration - 0.01)))
                         .astype(float).mean(axis=2)[::4, ::4])
            dx, dy = _estimate_frame_shift(prev_gray, curr_gray)
            dx *= 4; dy *= 4  # ダウンサンプリング補正
            cum_x.append(cum_x[-1] + dx)
            cum_y.append(cum_y[-1] + dy)
            prev_gray = curr_gray

        cum_x = np.array(cum_x)
        cum_y = np.array(cum_y)

        # ── 2. 移動平均で軌跡を平滑化 ──
        window = max(len(times) // 4, 3)
        kernel = np.ones(window) / window
        smooth_x = np.convolve(cum_x, kernel, mode="same")
        smooth_y = np.convolve(cum_y, kernel, mode="same")

        # 補正量 = 平滑軌跡 - 実際の軌跡
        corr_x = smooth_x - cum_x
        corr_y = smooth_y - cum_y

        # ── 3. 各フレームに補正を適用する変換関数 ──
        def stabilize_frame(get_frame, t):
            frame = get_frame(t)
            idx = int(np.searchsorted(times, t, side="right") - 1)
            idx = int(np.clip(idx, 0, len(times) - 1))
            cx = int(np.clip(round(corr_x[idx]), -MAX_CORRECTION_PX, MAX_CORRECTION_PX))
            cy = int(np.clip(round(corr_y[idx]), -MAX_CORRECTION_PX, MAX_CORRECTION_PX))

            if cx == 0 and cy == 0:
                return frame

            result = np.zeros_like(frame)
            src_y1 = max(0, -cy); src_y2 = min(h, h - cy)
            src_x1 = max(0, -cx); src_x2 = min(w, w - cx)
            dst_y1 = max(0, cy);  dst_y2 = min(h, h + cy)
            dst_x1 = max(0, cx);  dst_x2 = min(w, w + cx)

            if src_y1 < src_y2 and src_x1 < src_x2:
                result[dst_y1:dst_y2, dst_x1:dst_x2] = \
                    frame[src_y1:src_y2, src_x1:src_x2]
            return result

        return clip.fl(stabilize_frame)

    except Exception as e:
        print(f"    [警告] 手ブレ補正に失敗しました: {e}")
        return clip


def correct_tilt(clip, angle_deg):
    """
    軽度な傾きを自動補正する関数。

    MoviePy の rotate() を使い、検出した傾き角だけ逆回転させます。
    TILT_IGNORE_DEG 未満の傾きはスキップ、TILT_CORRECT_MAX を超える
    傾きも補正せず（スコアで自然に除外される設計）。

    引数:
        clip      : 補正対象の MoviePy クリップ
        angle_deg : get_tilt_info() で検出した傾き角（度）

    戻り値:
        回転補正済みクリップ（補正不要・不可時は元のクリップ）
    """
    abs_angle = abs(angle_deg)

    if abs_angle < TILT_IGNORE_DEG:
        return clip  # ほぼ水平 → 補正不要

    if abs_angle > TILT_CORRECT_MAX:
        return clip  # 重度 → スコアで除外させる

    try:
        # 傾きと逆方向に回転（expand=False でフレームサイズを維持）
        return clip.rotate(-angle_deg, expand=False)
    except Exception as e:
        print(f"    [警告] 傾き補正に失敗しました: {e}")
        return clip


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
    1 つのクリップに対して全スコアと品質情報を計算して返す関数。

    この関数を呼ぶだけで 7 種類のスコアと品質指標が全て得られます。
    OpenCV や Whisper がインストールされていない場合、
    対応するスコアは自動的に 0.0 / フォールバック値 になります。

    引数:
        clip: MoviePy のクリップ
        whisper_model: load_whisper_model() の戻り値（None でスキップ）

    戻り値:
        (dict, dict): (scores, quality) のタプル

        scores: {
            "audio"    : float,  # 音量スコア
            "motion"   : float,  # 動きスコア
            "smile"    : float,  # 笑顔スコア      (0.0 if no OpenCV)
            "face_size": float,  # 顔サイズスコア   (0.0 if no OpenCV)
            "context"  : float,  # 文脈スコア      (0.0 if no Whisper)
            "shake"    : float,  # 手ブレスコア     (1.0=安定, 0.0=激しいブレ)
            "tilt"     : float,  # 傾きスコア      (1.0=水平, 0.0=大きく傾いている)
        }

        quality: {
            "shake_jitter": float,  # 生のジッター量 (px)
            "tilt_angle"  : float,  # 傾き角 (degree)
        }
    """
    shake_score, shake_jitter = get_shake_info(clip)
    tilt_score,  tilt_angle   = get_tilt_info(clip)

    scores = {
        "audio":     get_audio_score(clip),
        "motion":    get_motion_score(clip),
        "smile":     get_smile_score(clip),
        "face_size": get_face_size_score(clip),
        "context":   get_context_score(clip, whisper_model),
        "shake":     shake_score,
        "tilt":      tilt_score,
    }
    quality = {
        "shake_jitter": round(shake_jitter, 3),
        "tilt_angle":   round(tilt_angle, 3),
    }
    return scores, quality


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

    score_keys = ["audio", "motion", "smile", "face_size", "context", "shake", "tilt"]

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
    "w_shake":     0.0,  # 0.0 = 手ブレスコアを無視（デフォルト）
    "w_tilt":      0.0,  # 0.0 = 傾きスコアを無視（デフォルト）
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
          + w_shake     × shake_score
          + w_tilt      × tilt_score

    重みの設定例（variants.yaml）:
        笑顔重視バリアント : w_smile=0.5, w_audio=0.2, ...
        音量重視バリアント : w_audio=0.6, w_smile=0.1, ...
        文脈重視バリアント : w_context=0.5, w_audio=0.2, ...
        映像品質重視バリアント: w_shake=0.2, w_tilt=0.2, w_audio=0.3, ...

    【shake / tilt スコアの使い方】
        - w_shake > 0 にすると、手ブレの少ない安定した映像が優先されます
        - w_tilt  > 0 にすると、水平に近い映像が優先されます
        - どちらも 0.0 にすると品質スコアは無視されます（デフォルト）

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
        + weights.get("w_shake",     DEFAULT_WEIGHTS["w_shake"])     * norm_scores.get("shake",     0.5)
        + weights.get("w_tilt",      DEFAULT_WEIGHTS["w_tilt"])      * norm_scores.get("tilt",      0.5)
    )
