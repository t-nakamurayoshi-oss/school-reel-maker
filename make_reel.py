#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_reel.py

学校行事の素材動画から Instagram リール動画 (9:16) を自動生成するスクリプトです。

【使い方】
    1. input/ フォルダに動画ファイル (.mp4, .mov など) を入れる
    2. ターミナルで以下を実行する:
           python make_reel.py
    3. output/reel.mp4 が生成される

【仕組み】
    - 動画を数秒ごとのセグメント（区間）に分割する
    - 各セグメントの音量 (RMS) を計算する
    - 音量が大きいセグメント = 盛り上がりシーンを優先して選ぶ
    - 横長の動画は中央を切り抜いて縦 (1080×1920) に変換する
    - 選んだシーンをつなぎ合わせて reel.mp4 を出力する
"""

import os       # ファイルパス操作に使う標準ライブラリ
import sys      # スクリプト終了 (sys.exit) に使う標準ライブラリ
import glob     # ワイルドカードでファイルを検索する標準ライブラリ

import numpy as np                        # 音量計算の数値処理に使う
from moviepy.editor import (
    VideoFileClip,          # 動画ファイルを読み込むクラス
    concatenate_videoclips, # 複数のクリップをつなぐ関数
)

# ============================================================
# ★ 設定エリア: ここの数値を変えると動作をカスタマイズできます
# ============================================================

# 入力フォルダ（素材動画を入れる場所）
INPUT_DIR = "input"

# 出力フォルダと出力ファイル名
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reel.mp4")

# リールの目標時間 (秒)
# Instagram リールの最大は 90 秒。60 秒前後がおすすめ。
TARGET_DURATION = 60

# 動画を何秒ごとに区切って分析するか (秒)
# 短くすると細かく選べるが処理が遅くなる
SEGMENT_LENGTH = 3

# これより短いクリップは無視する (秒)
MIN_CLIP_DURATION = 2

# 出力動画の解像度 (Instagram リール推奨: 1080×1920 = 9:16 縦型)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# 出力動画のフレームレート (fps)
OUTPUT_FPS = 30

# クロスフェード（シーン切り替え時のなめらかな溶け込み）の長さ (秒)
# 0 にするとブツ切りになる
CROSSFADE_DURATION = 0.3

# 対応している動画の拡張子
VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"]

# ============================================================
# 関数定義エリア
# ============================================================

def get_audio_rms(clip):
    """
    動画クリップの音量 (RMS 値) を計算して返す関数。

    RMS (Root Mean Square: 二乗平均平方根) は音の大きさを表す指標で、
    値が大きいほど音が大きい（＝盛り上がっている）シーンを意味します。

    引数:
        clip: MoviePy の VideoFileClip またはサブクリップオブジェクト

    戻り値:
        float: RMS 値。音声がない場合は 0.0 を返す。
    """
    # 音声トラックがない動画は音量 0 として扱う
    if clip.audio is None:
        return 0.0

    try:
        # 音声データを NumPy 配列として取得する
        # fps=22050 は「1 秒あたりのサンプル数」（CD 音質の半分程度）
        audio_array = clip.audio.to_soundarray(fps=22050)

        # ステレオ（左右 2 チャンネル）の場合は平均を取ってモノラルにする
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # RMS を計算: 各サンプルを二乗 → 平均 → 平方根
        rms = np.sqrt(np.mean(audio_array ** 2))
        return float(rms)

    except Exception as e:
        # 音声の読み込みに失敗した場合（破損ファイルなど）
        print(f"    [警告] 音声の読み込みに失敗しました: {e}")
        return 0.0


def crop_to_vertical(clip, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT):
    """
    動画クリップを縦型 (9:16) にクロップしてリサイズする関数。

    横長 (16:9) の動画を縦型にするとき、左右を切り捨てて中央部分だけ残します。
    スマホで横向き動画を縦に持ったとき、左右がはみ出るイメージです。

    引数:
        clip : MoviePy のクリップオブジェクト
        width : 出力幅 (px)  ← デフォルト: 1080
        height: 出力高さ (px) ← デフォルト: 1920

    戻り値:
        クロップ・リサイズ済みのクリップ
    """
    original_w = clip.w   # 元の動画の幅
    original_h = clip.h   # 元の動画の高さ

    # 目標の縦横比 (9:16 ≈ 0.5625)
    target_ratio = width / height
    # 元の動画の縦横比
    original_ratio = original_w / original_h

    if original_ratio > target_ratio:
        # ── 横長動画の場合: 左右をカットする ──
        # 高さをそのまま保ちつつ、幅を目標比率に合わせて計算
        new_w = int(original_h * target_ratio)
        # x_center / y_center で「どこを中心に切り抜くか」を指定
        clipped = clip.crop(
            x_center=original_w / 2,  # 水平方向は真ん中
            y_center=original_h / 2,  # 垂直方向も真ん中
            width=new_w,
            height=original_h,
        )
    else:
        # ── 縦長・正方形動画の場合: 上下をカットする ──
        new_h = int(original_w / target_ratio)
        clipped = clip.crop(
            x_center=original_w / 2,
            y_center=original_h / 2,
            width=original_w,
            height=new_h,
        )

    # 最終的に目標解像度 (1080×1920) にリサイズして返す
    return clipped.resize((width, height))


def analyze_video(video_path):
    """
    1 本の動画ファイルを分析し、各セグメントの音量情報を返す関数。

    動画を SEGMENT_LENGTH 秒ずつに区切り、それぞれの区間の音量を計測します。

    引数:
        video_path (str): 動画ファイルのパス

    戻り値:
        list[dict]: セグメント情報のリスト。各要素は以下のキーを持つ辞書:
            - "rms"     : 音量 (float)
            - "start"   : 開始時間 (float, 秒)
            - "end"     : 終了時間 (float, 秒)
            - "duration": 長さ (float, 秒)
            - "path"    : 動画ファイルのパス (str)
        音量の大きい順に並び替えて返す。
    """
    print(f"\n  分析中: {os.path.basename(video_path)}")

    segments = []  # 分析結果を入れるリスト

    try:
        # 動画ファイルを読み込む
        video = VideoFileClip(video_path)
        duration = video.duration  # 動画の長さ（秒）

        print(f"    長さ: {duration:.1f}秒 / 解像度: {video.w}×{video.h}")

        # 動画が短すぎる場合はスキップ
        if duration < MIN_CLIP_DURATION:
            print(f"    [スキップ] 動画が短すぎます ({duration:.1f}秒 < {MIN_CLIP_DURATION}秒)")
            video.close()
            return segments

        # ── 動画を SEGMENT_LENGTH 秒ごとに分割して音量を測定 ──
        start_time = 0.0

        while start_time < duration - MIN_CLIP_DURATION:
            # セグメントの終了時間（動画の末尾を超えないように clamp する）
            end_time = min(start_time + SEGMENT_LENGTH, duration)
            segment_duration = end_time - start_time

            # 短すぎるセグメントは分析しない
            if segment_duration < MIN_CLIP_DURATION:
                break

            # 区間を切り出す（subclip は元の video を参照するだけ。コピーはしない）
            segment_clip = video.subclip(start_time, end_time)

            # 音量を測定
            rms = get_audio_rms(segment_clip)

            # 結果をリストに追加
            segments.append({
                "rms":      rms,
                "start":    start_time,
                "end":      end_time,
                "duration": segment_duration,
                "path":     video_path,
            })

            # 参照を解放してメモリを節約
            segment_clip.close()

            # 次のセグメントへ
            start_time += SEGMENT_LENGTH

        video.close()
        print(f"    → {len(segments)} セグメントを分析しました")

    except Exception as e:
        print(f"    [エラー] 動画の読み込みに失敗しました: {e}")

    # 音量の大きい順（降順）に並び替えて返す
    segments.sort(key=lambda x: x["rms"], reverse=True)
    return segments


def select_segments(all_segments, target_duration=TARGET_DURATION):
    """
    全セグメントの中から、目標時間に収まるように上位のものを選ぶ関数。

    引数:
        all_segments (list[dict]): 音量の大きい順に並んだ全セグメントのリスト
        target_duration (float)  : 目標の合計時間 (秒)

    戻り値:
        (list[dict], float): 選ばれたセグメントのリストと合計時間のタプル
    """
    selected = []
    total_duration = 0.0

    for seg in all_segments:
        # 目標時間に達したら選択を終了
        if total_duration >= target_duration:
            break

        selected.append(seg)
        total_duration += seg["duration"]

    return selected, total_duration


# ============================================================
# メイン処理
# ============================================================

def make_reel():
    """
    リール動画を生成するメイン処理。

    処理の流れ:
        [ステップ 1] input/ フォルダから動画ファイルを収集
        [ステップ 2] 各動画を分析して音量データを収集
        [ステップ 3] 音量の大きいセグメントを選択
        [ステップ 4] 縦型 (9:16) にクロップ & リサイズ
        [ステップ 5] クリップをつなぎ合わせて reel.mp4 に書き出す
    """
    print("=" * 55)
    print("  学校行事リールメーカー")
    print("=" * 55)

    # ──────────────────────────────────────────
    # ステップ 1: フォルダの確認 & 動画ファイルの収集
    # ──────────────────────────────────────────

    # input フォルダがなければ作成してユーザーに知らせる
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"\n[!] '{INPUT_DIR}/' フォルダを作成しました。")
        print(f"    素材動画をここに入れてから再実行してください。")
        sys.exit(0)

    # output フォルダがなければ作成する
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[✓] '{OUTPUT_DIR}/' フォルダを作成しました。")

    # input/ フォルダ内の動画ファイルを検索する
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        # 小文字の拡張子と大文字の拡張子の両方を検索（例: .mp4 と .MP4）
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))

    # 重複を取り除いてアルファベット順に並べる
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"\n[!] '{INPUT_DIR}/' フォルダに動画が見つかりません。")
        print(f"    対応形式: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(0)

    print(f"\n[✓] {len(video_files)} 本の動画を見つけました:")
    for f in video_files:
        print(f"    - {os.path.basename(f)}")

    # ──────────────────────────────────────────
    # ステップ 2: 全動画を分析して音量データを収集
    # ──────────────────────────────────────────

    print(f"\n[1/3] 動画を分析しています...")

    all_segments = []
    for video_path in video_files:
        segs = analyze_video(video_path)
        all_segments.extend(segs)

    if not all_segments:
        print("\n[エラー] 有効なセグメントが見つかりませんでした。")
        print("         動画ファイルが壊れていないか確認してください。")
        sys.exit(1)

    # 全セグメントをまとめて音量の大きい順に並び替える
    all_segments.sort(key=lambda x: x["rms"], reverse=True)

    # 上位 5 件のシーンを表示（ログ確認用）
    print(f"\n  合計 {len(all_segments)} セグメントが見つかりました。")
    print(f"  音量トップ 5 シーン:")
    for i, seg in enumerate(all_segments[:5]):
        fname = os.path.basename(seg["path"])
        print(f"    {i+1}. {fname}  {seg['start']:.1f}s〜{seg['end']:.1f}s"
              f"  (RMS: {seg['rms']:.4f})")

    # ──────────────────────────────────────────
    # ステップ 3: セグメントを選択
    # ──────────────────────────────────────────

    print(f"\n[2/3] 上位シーンを選択しています... (目標: {TARGET_DURATION}秒)")
    selected, total_sec = select_segments(all_segments)
    print(f"  → {len(selected)} シーンを選択 (合計: {total_sec:.1f}秒)")

    # ──────────────────────────────────────────
    # ステップ 4 & 5: クリップを組み立てて書き出す
    # ──────────────────────────────────────────

    print(f"\n[3/3] リール動画を生成しています...")

    # 開いた VideoFileClip を追跡するリスト（後で close するため）
    open_videos = []
    # 最終的につなぐクリップを格納するリスト
    clips = []

    for i, seg in enumerate(selected):
        print(f"  クリップ {i+1:2d}/{len(selected)}: "
              f"{os.path.basename(seg['path'])}"
              f"  {seg['start']:.1f}s〜{seg['end']:.1f}s")

        # 動画ファイルを開く（後で close するために追跡する）
        video = VideoFileClip(seg["path"])
        open_videos.append(video)

        # 指定した区間を切り出す
        clip = video.subclip(seg["start"], seg["end"])

        # 縦型 (1080×1920) にクロップ & リサイズ
        clip_vertical = crop_to_vertical(clip)

        clips.append(clip_vertical)

    if not clips:
        print("\n[エラー] 有効なクリップの生成に失敗しました。")
        sys.exit(1)

    # ── クリップをつなぎ合わせる ──
    print("\n  クリップを結合しています...")

    if len(clips) > 1 and CROSSFADE_DURATION > 0:
        # クロスフェード（場面転換のなめらかな溶け込み）を適用する場合
        # 最初のクリップはそのまま使い、2 番目以降に crossfadein を適用する
        clips_with_fade = [clips[0]]
        for clip in clips[1:]:
            clips_with_fade.append(clip.crossfadein(CROSSFADE_DURATION))

        # padding を負にすることで、クリップが CROSSFADE_DURATION 秒分重なってフェードする
        final_video = concatenate_videoclips(
            clips_with_fade,
            padding=-CROSSFADE_DURATION,
            method="compose",
        )
    else:
        # クロスフェードなし（ブツ切り）
        final_video = concatenate_videoclips(clips, method="compose")

    # ── 動画ファイルとして書き出す ──
    print(f"\n  書き出し中: {OUTPUT_FILE}")
    print("  (動画の長さや画質によって数分かかることがあります...)\n")

    final_video.write_videofile(
        OUTPUT_FILE,
        fps=OUTPUT_FPS,
        codec="libx264",       # 広く使われている H.264 映像コーデック
        audio_codec="aac",     # 標準的な AAC 音声コーデック
        bitrate="5000k",       # 映像ビットレート (高いほど高画質・ファイルが大きい)
        threads=4,             # 並列処理スレッド数
        logger="bar",          # ターミナルに進捗バーを表示
    )

    # ── 後処理: 開いたファイルを閉じてメモリを解放 ──
    final_video.close()
    for v in open_videos:
        try:
            v.close()
        except Exception:
            pass

    # 完了メッセージ
    print("\n" + "=" * 55)
    print("  [完了] リール動画が生成されました!")
    print(f"  出力ファイル: {OUTPUT_FILE}")
    print("=" * 55)


# ============================================================
# エントリーポイント
# このスクリプトを直接実行したときだけ make_reel() が動く。
# 他のファイルから import しても自動実行はされない。
# ============================================================

if __name__ == "__main__":
    make_reel()
