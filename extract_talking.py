#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_talking.py

定点カメラ映像から「カメラに向かって話しかけている部分」を自動で切り出すスクリプト。

【使い方】
    python extract_talking.py 動画ファイル.mp4
    python extract_talking.py 動画ファイル.mp4 --output output/talking/
    python extract_talking.py 動画ファイル.mp4 --min-sec 3 --padding 1.0
    python extract_talking.py 動画ファイル.mp4 --speech-top 50 --interval 0.3

【仕組み】
    ① フレームサンプリング
        SAMPLE_INTERVAL 秒ごとにフレームを取得する。

    ② 正面顔検出（カメラ向き判定）
        OpenCV の Haar Cascade（frontalface）で正面顔を検出する。
        正面顔が検出される = カメラ方向を向いている、と判定する。
        さらに目の検出（eye cascade）を追加確認として使い、
        横顔や下を向いた顔を誤検出しにくくする。

    ③ 発話判定（音声 RMS）
        ffmpeg で動画の音声全体を PCM として取得し、
        各区間の RMS（音量）を計算する。
        全区間の RMS を並べて「上位 SPEECH_TOP_PERCENT% 以上」を発話とみなす。
        → 絶対値に依存せず、動画ごとに自動で閾値が決まる。

    ④ 口の動き検出（補助判定）
        正面顔が検出された場合、顔領域の下半分（口周辺）のフレーム差分を計算する。
        前後フレームで口の動きがあれば「発話あり」として補強する。
        音声が取れない環境や BGM 有り動画でも機能する。

    ⑤ 判定ロジック
        「カメラを向いて話している」と判定する条件:
            正面顔あり AND (音声発話あり OR 口の動きあり)

    ⑥ 区間マージ
        短いギャップ（GAP_FILL_SEC 以下）をつないで、
        細切れにならないようにまとめる。

    ⑦ クリップ書き出し
        output/talking/ に talking_001.mp4, talking_002.mp4 ... として書き出す。
        結果一覧を talking_segments.csv に保存する。

【必要ライブラリ】
    pip install opencv-python-headless moviepy numpy

【オプション依存】
    ffmpeg がインストール済みであること（MoviePy が内部で使用）
"""

import os
import sys
import csv
import argparse
import subprocess

import numpy as np

# ============================================================
# オプション依存のライブラリ
# ============================================================

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# ============================================================
# 設定エリア: ここを変えると動作をカスタマイズできます
# ============================================================

# フレームのサンプリング間隔 (秒)
# 小さくすると精度が上がるが処理が遅くなる
SAMPLE_INTERVAL = 0.5

# ── 顔検出の設定 ──
# スキャン縮小率: 小さいほど小さい顔も検出されるが処理が遅くなる (1.05〜1.3 推奨)
FACE_SCALE_FACTOR = 1.15
# 検出の厳しさ: 大きいほど誤検知が減るが検出漏れが増える (3〜8 推奨)
FACE_MIN_NEIGHBORS = 5
# フレーム幅に対する最小顔サイズの割合: 顔が小さく映る場合は下げる (0.03〜0.1)
FACE_MIN_SIZE_RATIO = 0.05

# 目の検出を使って「カメラを向いた正面顔」の精度を高めるか
# True = 目が検出された顔だけを「カメラ向き」とみなす（精度優先）
# False = 正面顔検出だけで判定（検出率優先）
USE_EYE_CONFIRMATION = True

# ── 発話判定の設定 ──
# 全区間の RMS を並べたとき「上位何%以上を発話とみなすか」
# 例: 40 → 上位 40% の音量区間を発話とみなす
# 静かな場所なら下げる (30)、ざわざわした場所なら上げる (50〜60)
SPEECH_TOP_PERCENT = 40

# ── 口の動き検出の設定 ──
# 前後フレームの顔下半分ピクセル差分がこの値を超えたら「口が動いている」とみなす
# 小さくすると敏感になる（誤検知増）、大きくすると鈍感になる（検出漏れ増）
MOUTH_DIFF_THRESHOLD = 8.0

# ── セグメント設定 ──
# 最小クリップ長 (秒): これより短い区間はスキップ
MIN_SEGMENT_SEC = 2.0
# セグメント前後のパディング (秒): 発話の冒頭・末尾が切れないよう余白を加える
PADDING_SEC = 0.5
# ギャップフィル (秒): この秒数以下の「話していない」区間は無視してつなぐ
GAP_FILL_SEC = 1.5

# ── 出力設定 ──
OUTPUT_FPS = 30
OUTPUT_BITRATE = "5000k"


# ============================================================
# Step 1: 音声 RMS の計算
# ============================================================

def load_audio_rms(video_path, sample_times, window_sec):
    """
    動画の音声全体を一括取得し、各サンプル時刻の RMS を計算して返す。

    ffmpeg で全音声を PCM として取得し、NumPy でウィンドウごとに切り出して計算する。
    区間ごとに ffmpeg を呼び出すより大幅に高速。

    引数:
        video_path  : 動画ファイルのパス
        sample_times: サンプリング時刻の配列
        window_sec  : 各ウィンドウの長さ (秒)

    戻り値:
        np.ndarray: 各サンプル時刻の RMS 値
    """
    print("  [1/3] 音声を分析しています...")

    sample_rate = 16000  # 16kHz モノラルで取得 (発話検出には十分な品質)

    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                    # 映像トラックを無視
            "-acodec", "pcm_f32le",  # 32bit float PCM
            "-ar", str(sample_rate),  # サンプルレート
            "-ac", "1",               # モノラル
            "-f", "f32le",            # 出力フォーマット
            "-loglevel", "error",     # ffmpeg のログを抑制
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)

        if not result.stdout:
            print("  [情報] 音声トラックが見つかりません。発話判定は口の動きのみで行います。")
            return np.zeros(len(sample_times))

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        samples_per_window = int(sample_rate * window_sec)

        rms_values = []
        for t in sample_times:
            start_idx = int(t * sample_rate)
            end_idx = start_idx + samples_per_window
            window = audio[start_idx:end_idx]
            if len(window) == 0:
                rms_values.append(0.0)
            else:
                rms_values.append(float(np.sqrt(np.mean(window ** 2))))

        return np.array(rms_values)

    except subprocess.TimeoutExpired:
        print("  [警告] 音声分析がタイムアウトしました。音声なしとして処理します。")
        return np.zeros(len(sample_times))
    except Exception as e:
        print(f"  [警告] 音声分析エラー: {e}")
        return np.zeros(len(sample_times))


def compute_speech_threshold(rms_values, top_percent):
    """
    RMS の分布から「発話とみなす閾値」を自動計算する。

    「上位 top_percent% に入る RMS 値の最小値」を閾値とする。
    動画ごとに音量レベルが異なっても自動で適応する。

    例: top_percent=40 → 全区間の上位 40% = 発話とみなす
    """
    if np.all(rms_values == 0):
        return float("inf")  # 全区間が 0 → 発話なし（口の動きだけで判定）

    return float(np.percentile(rms_values, 100 - top_percent))


# ============================================================
# Step 2: 正面顔 + 口の動き検出
# ============================================================

def detect_faces_and_mouth(video_path, sample_times):
    """
    各サンプル時刻で「正面顔があるか」「口が動いているか」を判定する。

    正面顔検出:
        Haar Cascade (frontalface_default) で検出。
        USE_EYE_CONFIRMATION=True の場合、さらに顔領域内で目を検出し、
        目が確認できた顔だけを「カメラ向き」と判定する（精度向上）。

    口の動き検出:
        前のサンプルフレームと現在のフレームで、
        顔の下半分領域（口周辺）のピクセル差分を計算する。
        差分が MOUTH_DIFF_THRESHOLD を超えたら「口が動いている」とみなす。

    戻り値:
        (face_flags, mouth_flags): 各サンプル時刻の bool 配列のタプル
    """
    print("  [2/3] 顔の向きと口の動きを検出しています...")

    # カスケード分類器の読み込み
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [エラー] 動画を開けませんでした: {video_path}")
        return np.zeros(len(sample_times), dtype=bool), np.zeros(len(sample_times), dtype=bool)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_face_px = int(min(frame_w, frame_h) * FACE_MIN_SIZE_RATIO)

    face_flags  = []
    mouth_flags = []
    prev_mouth_gray = None  # 口周辺の前フレーム（差分計算用）

    total = len(sample_times)

    for i, t in enumerate(sample_times):
        if i % 20 == 0:
            print(f"    {i}/{total} フレーム処理中...", end="\r", flush=True)

        # 指定時刻にシーク
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            face_flags.append(False)
            mouth_flags.append(False)
            prev_mouth_gray = None
            continue

        # 処理速度向上のため 1/2 サイズにダウンサンプリング
        scale = 0.5
        small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ── 正面顔検出 ──
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(int(min_face_px * scale), int(min_face_px * scale)),
        )

        has_face = False
        best_face = None  # 最大の顔（口の動き検出に使う）

        for (fx, fy, fw, fh) in faces:
            if USE_EYE_CONFIRMATION:
                # 顔の上半分で目を検出（目が見えない = 横向き or 俯き の可能性が高い）
                upper_roi = gray[fy : fy + fh // 2, fx : fx + fw]
                eyes = eye_cascade.detectMultiScale(
                    upper_roi,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(int(fw * 0.15), int(fh * 0.08)),
                )
                if len(eyes) == 0:
                    continue  # 目が見えない顔はスキップ

            has_face = True
            # 最大面積の顔を best_face として記録
            if best_face is None or fw * fh > best_face[2] * best_face[3]:
                best_face = (fx, fy, fw, fh)

        face_flags.append(has_face)

        # ── 口の動き検出 ──
        mouth_moving = False
        if best_face is not None:
            fx, fy, fw, fh = best_face
            # 顔の下半分（口周辺）を切り出す
            mouth_y1 = fy + fh // 2
            mouth_y2 = fy + fh
            mouth_roi = gray[mouth_y1:mouth_y2, fx : fx + fw]

            if prev_mouth_gray is not None and mouth_roi.shape == prev_mouth_gray.shape:
                # 前フレームとの差分で口の動きを計算
                diff = np.abs(mouth_roi.astype(float) - prev_mouth_gray.astype(float))
                mean_diff = float(diff.mean())
                mouth_moving = mean_diff > MOUTH_DIFF_THRESHOLD

            prev_mouth_gray = mouth_roi.copy()
        else:
            prev_mouth_gray = None

        mouth_flags.append(mouth_moving)

    cap.release()
    print(f"    {total}/{total} フレーム完了          ")

    return np.array(face_flags), np.array(mouth_flags)


# ============================================================
# Step 3: 発話シーンの区間マージ
# ============================================================

def find_talking_segments(
    sample_times, face_flags, rms_values, mouth_flags,
    speech_threshold, gap_fill_sec, min_sec, padding_sec, total_duration,
):
    """
    「正面顔あり + (発話あり OR 口の動きあり)」の区間を検出し、
    マージ・フィルタリングしてセグメントリストを返す。

    判定ロジック:
        talking = face_detected AND (rms >= speech_threshold OR mouth_moving)

    audio が取れない場合 (rms = 0 全区間) は face + 口の動きだけで判定。

    引数:
        sample_times     : サンプリング時刻の配列
        face_flags       : 各時刻に正面顔があるか (bool 配列)
        rms_values       : 各時刻の音声 RMS (float 配列)
        mouth_flags      : 各時刻に口が動いているか (bool 配列)
        speech_threshold : 発話とみなす RMS の閾値
        gap_fill_sec     : つなぐギャップの最大長 (秒)
        min_sec          : 最小クリップ長 (秒)
        padding_sec      : 前後パディング (秒)
        total_duration   : 動画の長さ (秒)

    戻り値:
        list[dict]: {"start", "end", "duration"} のリスト
    """
    # 音声が完全に取れない場合は口の動きだけで発話を判定
    audio_available = not np.all(rms_values == 0)

    if audio_available:
        speech_flags = rms_values >= speech_threshold
        # 発話判定: 音声あり OR 口の動きあり
        talking_flags = face_flags & (speech_flags | mouth_flags)
    else:
        # 音声なし: 顔あり AND 口の動きあり
        talking_flags = face_flags & mouth_flags

    if not np.any(talking_flags):
        return []

    # ── 連続する True 区間をひとまとめにする ──
    raw_segments = []
    in_seg = False
    seg_start = None

    for t, is_talking in zip(sample_times, talking_flags):
        if is_talking and not in_seg:
            seg_start = float(t)
            in_seg = True
        elif not is_talking and in_seg:
            raw_segments.append([seg_start, float(t)])
            in_seg = False

    if in_seg:
        raw_segments.append([seg_start, float(sample_times[-1])])

    if not raw_segments:
        return []

    # ── ギャップフィル: 短いギャップをつなぐ ──
    merged = [raw_segments[0]]
    for start, end in raw_segments[1:]:
        gap = start - merged[-1][1]
        if gap <= gap_fill_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # ── パディング追加 + 最小長フィルタ ──
    result = []
    for start, end in merged:
        padded_start = max(0.0, start - padding_sec)
        padded_end   = min(total_duration, end + padding_sec)
        duration = padded_end - padded_start
        if duration >= min_sec:
            result.append({
                "start":    round(padded_start, 3),
                "end":      round(padded_end,   3),
                "duration": round(duration,     3),
            })

    return result


# ============================================================
# Step 4: クリップの書き出し
# ============================================================

def export_clips(video_path, segments, output_dir):
    """
    検出されたセグメントを個別の MP4 クリップとして書き出す。

    出力ファイル名: talking_001.mp4, talking_002.mp4, ...

    戻り値:
        list[str]: 書き出したファイルのパスリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    print(f"\n  [3/3] クリップを書き出しています... ({len(segments)} 件)")

    video = VideoFileClip(video_path)

    for i, seg in enumerate(segments, start=1):
        clip_name = f"talking_{i:03d}.mp4"
        out_path  = os.path.join(output_dir, clip_name)

        print(f"    [{i:3d}/{len(segments)}] {clip_name}"
              f"  {seg['start']:.1f}s 〜 {seg['end']:.1f}s"
              f"  ({seg['duration']:.1f}秒)")

        clip = video.subclip(seg["start"], seg["end"])
        clip.write_videofile(
            out_path,
            fps=OUTPUT_FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate=OUTPUT_BITRATE,
            threads=2,
            logger=None,  # MoviePy の詳細ログを抑制
        )
        clip.close()
        output_paths.append(out_path)

    video.close()
    return output_paths


# ============================================================
# Step 5: サマリ CSV の保存
# ============================================================

def save_summary_csv(segments, output_paths, output_dir, video_path):
    """
    切り出し結果を CSV ファイルに保存する。

    保存内容: クリップ名, 元動画名, 開始時刻, 終了時刻, 長さ
    """
    csv_path = os.path.join(output_dir, "talking_segments.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "source", "start_sec", "end_sec", "duration_sec"])
        for seg, out_path in zip(segments, output_paths):
            writer.writerow([
                os.path.basename(out_path),
                os.path.basename(video_path),
                seg["start"],
                seg["end"],
                seg["duration"],
            ])

    print(f"  サマリ CSV: {csv_path}")
    return csv_path


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="定点カメラ映像からカメラに向かって話しているシーンを切り出す",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python extract_talking.py input/event.mp4
  python extract_talking.py input/event.mp4 --output output/talking/
  python extract_talking.py input/event.mp4 --min-sec 3 --padding 1.0
  python extract_talking.py input/event.mp4 --speech-top 50 --no-eye-check
        """,
    )
    parser.add_argument("video", help="入力動画ファイルのパス")
    parser.add_argument(
        "--output", "-o",
        default=os.path.join("output", "talking"),
        help="出力フォルダ (デフォルト: output/talking/)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float, default=SAMPLE_INTERVAL,
        help=f"サンプリング間隔 秒 (デフォルト: {SAMPLE_INTERVAL})",
    )
    parser.add_argument(
        "--min-sec", "-m",
        type=float, default=MIN_SEGMENT_SEC,
        help=f"最小クリップ長 秒 (デフォルト: {MIN_SEGMENT_SEC})",
    )
    parser.add_argument(
        "--padding",
        type=float, default=PADDING_SEC,
        help=f"前後パディング 秒 (デフォルト: {PADDING_SEC})",
    )
    parser.add_argument(
        "--gap-fill",
        type=float, default=GAP_FILL_SEC,
        help=f"ギャップフィル 秒 (デフォルト: {GAP_FILL_SEC})",
    )
    parser.add_argument(
        "--speech-top",
        type=float, default=SPEECH_TOP_PERCENT,
        help=f"上位何%%の音量を発話とみなすか (デフォルト: {SPEECH_TOP_PERCENT})",
    )
    parser.add_argument(
        "--no-eye-check",
        action="store_true",
        help="目の確認をスキップして顔検出のみで判定する (検出率優先)",
    )
    args = parser.parse_args()

    # ── ライブラリチェック ──
    if not OPENCV_AVAILABLE:
        print("[エラー] OpenCV が必要です。")
        print("         pip install opencv-python-headless")
        sys.exit(1)

    if not MOVIEPY_AVAILABLE:
        print("[エラー] MoviePy が必要です。")
        print("         pip install moviepy")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"[エラー] 動画ファイルが見つかりません: {args.video}")
        sys.exit(1)

    # コマンドライン引数でグローバル設定を上書き
    global USE_EYE_CONFIRMATION
    if args.no_eye_check:
        USE_EYE_CONFIRMATION = False

    print("=" * 60)
    print("  カメラ向き発話シーン 切り出しツール")
    print("=" * 60)
    print(f"  入力      : {args.video}")
    print(f"  出力      : {args.output}/")
    print(f"  サンプリング間隔: {args.interval}秒")
    print(f"  発話判定  : 上位 {args.speech_top:.0f}% の音量 + 口の動き")
    print(f"  目の確認  : {'あり（精度優先）' if USE_EYE_CONFIRMATION else 'なし（検出率優先）'}")
    print(f"  最小クリップ長 : {args.min_sec}秒")
    print(f"  前後パディング : {args.padding}秒")
    print()

    # ── 動画情報を取得 ──
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[エラー] 動画を開けませんでした: {args.video}")
        sys.exit(1)

    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps_orig
    cap.release()

    print(f"  動画の長さ  : {duration:.1f} 秒 ({duration / 60:.1f} 分)")
    sample_times = np.arange(0, duration, args.interval)
    print(f"  サンプル数  : {len(sample_times)} フレーム")
    print()

    # ── Step 1: 音声 RMS の計算 ──
    rms_values = load_audio_rms(args.video, sample_times, args.interval)
    speech_threshold = compute_speech_threshold(rms_values, args.speech_top)
    audio_available = not np.all(rms_values == 0)

    if audio_available:
        speech_count = int(np.sum(rms_values >= speech_threshold))
        print(f"  発話閾値 (RMS) : {speech_threshold:.5f}  "
              f"({speech_count}/{len(rms_values)} 区間が音声発話)")
    else:
        print("  音声なし → 顔検出 + 口の動きで判定します")

    # ── Step 2: 顔検出 + 口の動き検出 ──
    face_flags, mouth_flags = detect_faces_and_mouth(args.video, sample_times)
    face_count  = int(np.sum(face_flags))
    mouth_count = int(np.sum(mouth_flags))
    print(f"  正面顔あり : {face_count}/{len(face_flags)} 区間")
    print(f"  口の動きあり: {mouth_count}/{len(mouth_flags)} 区間")

    # ── Step 3: 発話セグメントの検出 ──
    print("\n  発話シーンを検出しています...")
    segments = find_talking_segments(
        sample_times, face_flags, rms_values, mouth_flags,
        speech_threshold, args.gap_fill, args.min_sec, args.padding, duration,
    )

    if not segments:
        print("\n[!] 「カメラを向いて話しているシーン」が検出されませんでした。")
        print("\n    試してみてください:")
        print("    - --speech-top の値を上げる (例: --speech-top 60)")
        print("      → 「上位 60% の音量」を発話とみなす（広く拾う）")
        print("    - --no-eye-check を追加する")
        print("      → 目の確認をスキップして顔検出のみで判定する")
        print("    - --min-sec の値を下げる (例: --min-sec 1.0)")
        print("      → 短い発話区間も切り出す")
        print("    - FACE_MIN_NEIGHBORS を下げる (スクリプト内 設定エリア)")
        print("      → 顔が小さく映っている場合に有効")
        sys.exit(0)

    total_sec = sum(s["duration"] for s in segments)
    print(f"\n  {len(segments)} 件の発話シーンを検出しました (合計 {total_sec:.1f}秒):")
    for i, seg in enumerate(segments, 1):
        print(f"    {i:3d}. {seg['start']:7.1f}s 〜 {seg['end']:7.1f}s"
              f"  ({seg['duration']:.1f}秒)")

    # ── Step 4: クリップ書き出し ──
    output_paths = export_clips(args.video, segments, args.output)

    # ── Step 5: サマリ CSV 保存 ──
    save_summary_csv(segments, output_paths, args.output, args.video)

    # ── 完了 ──
    print("\n" + "=" * 60)
    print("  [完了] 切り出しが終わりました!")
    print(f"  出力フォルダ : {args.output}/")
    print(f"  生成クリップ : {len(output_paths)} 本")
    print(f"  合計時間     : {total_sec:.1f}秒 ({total_sec / 60:.1f}分)")
    print("=" * 60)


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    main()
