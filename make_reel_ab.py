#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_reel_ab.py

ABテスト用に複数バリアントのリール動画を自動生成するスクリプト。

【使い方】
    1. input/ フォルダに動画ファイルを入れる
    2. presets/variants.yaml でバリアントを設定する
    3. python make_reel_ab.py を実行する
    4. output/candidates/reel_A.mp4, reel_B.mp4, ... が生成される
    5. experiments/review.csv に投稿後の数値を記入する
    6. python recommend_next.py で次回おすすめ設定を確認する

【make_reel.py との主な違い】
    - 複数バリアントを一括生成
    - 音量・動き・笑顔・顔サイズ・文脈の 5 種類のスコアを計算
    - variants.yaml の weights でバリアントごとに重みを変えてカット選定
    - 顔検出で人物が画面中央に来るようにクロップ（OpenCV が必要）
    - 生成条件を JSON マニフェストに記録
"""

import os
import sys
import glob
import json
import datetime

import numpy as np
import yaml
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
)

# ── make_reel.py から共通ユーティリティを import ──
from make_reel import (
    crop_to_vertical,   # 中央クロップ版（OpenCV なし環境でのフォールバック）
    VIDEO_EXTENSIONS,
    MIN_CLIP_DURATION,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
)

# ── score_engine から全スコア計算・正規化・合算関数を import ──
from score_engine import (
    OPENCV_AVAILABLE,
    WHISPER_AVAILABLE,
    load_whisper_model,
    compute_all_scores,
    normalize_all_scores,
    compute_total_score,
    crop_to_vertical_face,
    DEFAULT_WEIGHTS,
)

# ============================================================
# パス設定
# ============================================================

INPUT_DIR       = "input"
VARIANTS_FILE   = os.path.join("presets", "variants.yaml")
CANDIDATES_DIR  = os.path.join("output", "candidates")
MANIFESTS_DIR   = os.path.join("output", "manifests")
EXPERIMENTS_DIR = "experiments"
REVIEW_CSV      = os.path.join(EXPERIMENTS_DIR, "review.csv")


# ============================================================
# variants.yaml 読み込み
# ============================================================

def load_config_and_variants():
    """
    presets/variants.yaml を読み込み、グローバル設定とバリアントリストを返す関数。

    戻り値:
        (dict, list[dict]): (config設定辞書, バリアント設定のリスト)
    """
    if not os.path.exists(VARIANTS_FILE):
        print(f"[エラー] {VARIANTS_FILE} が見つかりません。")
        sys.exit(1)

    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config   = data.get("config", {})
    variants = data.get("variants", [])

    if not variants:
        print("[エラー] variants.yaml にバリアントが定義されていません。")
        sys.exit(1)

    return config, variants


# ============================================================
# 動画分析（全スコアを計算）
# ============================================================

def analyze_all_videos(video_files, clip_sec, whisper_model=None):
    """
    全動画ファイルを分析し、各セグメントの 5 種類のスコアを計算する関数。

    【スコア計算の流れ】
        1. 動画を clip_sec 秒ごとに分割
        2. 各セグメントで compute_all_scores() を呼び、5スコアを取得
        3. 全セグメントを返す（正規化はこの後で一括処理）

    引数:
        video_files (list[str]): 動画ファイルのパスリスト
        clip_sec (float)       : 1セグメントの長さ（秒）
        whisper_model          : Whisper モデル（None でコンテキストスコアをスキップ）

    戻り値:
        list[dict]: セグメント情報のリスト。各要素:
            - path, start, end, duration : 動画・時間情報
            - scores (dict)              : 5種類の生スコア
    """
    all_segments = []

    for video_path in video_files:
        print(f"\n  分析中: {os.path.basename(video_path)}")

        try:
            video    = VideoFileClip(video_path)
            duration = video.duration

            print(f"    長さ: {duration:.1f}秒 / 解像度: {video.w}×{video.h}")

            if duration < MIN_CLIP_DURATION:
                print(f"    [スキップ] 動画が短すぎます")
                video.close()
                continue

            start_time = 0.0
            count = 0

            while start_time < duration - MIN_CLIP_DURATION:
                end_time = min(start_time + clip_sec, duration)
                seg_dur  = end_time - start_time

                if seg_dur < MIN_CLIP_DURATION:
                    break

                seg_clip = video.subclip(start_time, end_time)

                # 5種類のスコアを一括計算
                scores = compute_all_scores(seg_clip, whisper_model)

                all_segments.append({
                    "path":     video_path,
                    "start":    start_time,
                    "end":      end_time,
                    "duration": seg_dur,
                    "scores":   scores,  # {"audio": ..., "motion": ..., ...}
                })

                seg_clip.close()
                start_time += clip_sec
                count += 1

            video.close()
            print(f"    → {count} セグメントを分析しました")

        except Exception as e:
            print(f"    [エラー] {os.path.basename(video_path)}: {e}")
            continue

    return all_segments


# ============================================================
# フック（冒頭シーン）選択
# ============================================================

def select_hook_segment(segments, hook_mode):
    """
    バリアントの hook_mode に応じて冒頭シーンを 1 つ選ぶ関数。

    hook_mode の意味:
        loud      → 音量が最大のシーン（歓声・拍手を冒頭に）
        motion    → 動きが最大のシーン（激しい動作を冒頭に）
        face      → 顔サイズスコアが最大のシーン（クローズアップを冒頭に）[要 OpenCV]
        smile     → 笑顔スコアが最大のシーン（笑顔を冒頭に）              [要 OpenCV]

    引数:
        segments (list[dict]): 全セグメントのリスト（"scores" キーが必要）
        hook_mode (str)      : フックの選び方

    戻り値:
        dict: フックとして使うセグメント（1件）
    """
    if not segments:
        return None

    if hook_mode == "loud":
        return max(segments, key=lambda s: s["scores"]["audio"])

    elif hook_mode == "motion":
        return max(segments, key=lambda s: s["scores"]["motion"])

    elif hook_mode == "face":
        if not OPENCV_AVAILABLE:
            print("    [情報] face モード: OpenCV がないため loud にフォールバック")
            return max(segments, key=lambda s: s["scores"]["audio"])
        return max(segments, key=lambda s: s["scores"]["face_size"])

    elif hook_mode == "smile":
        if not OPENCV_AVAILABLE:
            print("    [情報] smile モード: OpenCV がないため loud にフォールバック")
            return max(segments, key=lambda s: s["scores"]["audio"])
        return max(segments, key=lambda s: s["scores"]["smile"])

    else:
        print(f"    [警告] 未知の hook_mode: '{hook_mode}'。loud にフォールバックします。")
        return max(segments, key=lambda s: s["scores"]["audio"])


# ============================================================
# セグメント選択（フック + 本編）
# ============================================================

def select_segments_for_variant(all_segments, hook_segment, target_sec, weights):
    """
    フックセグメントを先頭に配置し、残りを総合スコア順で埋める関数。

    【選び方】
        1. フックセグメント（hook_mode で選んだシーン）を先頭に固定
        2. 残りを「総合スコア（重み付き）」の高い順に追加
        3. 合計時間が target_sec に達したら終了

    引数:
        all_segments (list[dict]): 全セグメント（"norm_scores" キーが必要）
        hook_segment (dict)      : 冒頭に配置するセグメント
        target_sec (float)       : 目標の合計時間（秒）
        weights (dict)           : 各スコアの重み（variants.yaml の weights）

    戻り値:
        (list[dict], float): 選ばれたセグメントリストと合計時間のタプル
    """
    selected  = []
    total_sec = 0.0

    # ── 1. フックセグメントを先頭に追加 ──
    if hook_segment:
        hook_copy = dict(hook_segment)
        hook_copy["role"] = "hook"
        selected.append(hook_copy)
        total_sec += hook_segment["duration"]

    # ── 2. 各セグメントの総合スコアを計算して降順ソート ──
    # ここでバリアントごとに異なる重みが適用される（ABテストの核心部分）
    scored = sorted(
        all_segments,
        key=lambda s: compute_total_score(s.get("norm_scores", {}), weights),
        reverse=True,
    )

    # ── 3. フック以外のセグメントを総合スコア順に追加 ──
    for seg in scored:
        if total_sec >= target_sec:
            break

        # フックと同じ区間は重複して追加しない
        is_hook = (
            hook_segment is not None
            and seg["path"]  == hook_segment["path"]
            and seg["start"] == hook_segment["start"]
        )
        if is_hook:
            continue

        seg_copy = dict(seg)
        seg_copy["role"] = "body"
        selected.append(seg_copy)
        total_sec += seg["duration"]

    return selected, total_sec


# ============================================================
# 動画書き出し（顔中心クロップ + BGM オプション付き）
# ============================================================

def build_and_export(selected_segments, output_path, crossfade_sec, bgm_path=None):
    """
    選択されたセグメントをつなぎ合わせて動画ファイルに書き出す関数。

    【クロップの動作】
        OpenCV がインストールされている場合:
            → crop_to_vertical_face() を使い、顔の中心にクロップ（人物中心クロップ）
        OpenCV がない場合:
            → crop_to_vertical() を使い、フレームの幾何学的中央にクロップ

    引数:
        selected_segments (list[dict]): 使用するセグメントリスト（先頭がフック）
        output_path (str)             : 出力ファイルのパス
        crossfade_sec (float)         : クロスフェードの長さ（秒）
        bgm_path (str or None)        : BGM ファイルのパス（None = なし）
    """
    open_videos = []
    clips = []

    for i, seg in enumerate(selected_segments):
        role_tag = "★フック" if seg.get("role") == "hook" else f" 本編{i:02d}"
        # スコア情報も表示してどんなシーンか分かるようにする
        s = seg.get("scores", {})
        score_str = (
            f"audio={s.get('audio', 0):.3f} "
            f"motion={s.get('motion', 0):.1f} "
            f"smile={s.get('smile', 0):.2f}"
        )
        print(f"  [{role_tag}] {os.path.basename(seg['path'])}"
              f"  {seg['start']:.1f}s〜{seg['end']:.1f}s  ({score_str})")

        video = VideoFileClip(seg["path"])
        open_videos.append(video)
        clip  = video.subclip(seg["start"], seg["end"])

        # ── クロップ: OpenCV があれば顔中心、なければフレーム中央 ──
        if OPENCV_AVAILABLE:
            clip_v = crop_to_vertical_face(clip, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        else:
            clip_v = crop_to_vertical(clip)

        clips.append(clip_v)

    if not clips:
        print("  [エラー] クリップが空です。このバリアントをスキップします。")
        return

    # ── クリップを結合 ──
    if len(clips) > 1 and crossfade_sec > 0:
        faded = [clips[0]]
        for c in clips[1:]:
            faded.append(c.crossfadein(crossfade_sec))
        final = concatenate_videoclips(faded, padding=-crossfade_sec, method="compose")
    else:
        final = concatenate_videoclips(clips, method="compose")

    # ── BGM を追加（オプション）──
    if bgm_path:
        if os.path.exists(bgm_path):
            print(f"  BGM を追加中: {bgm_path}")
            try:
                from moviepy.editor import CompositeAudioClip
                from moviepy.audio.fx.all import audio_loop

                bgm = AudioFileClip(bgm_path)
                # 動画の長さに BGM を合わせる（短い場合はループ）
                if bgm.duration < final.duration:
                    repeat = int(final.duration / bgm.duration) + 1
                    bgm = audio_loop(bgm, nloops=repeat)
                bgm = bgm.subclip(0, final.duration).volumex(0.3)  # 音量 30%
                if final.audio is not None:
                    final = final.set_audio(CompositeAudioClip([final.audio, bgm]))
                else:
                    final = final.set_audio(bgm)
            except Exception as e:
                print(f"  [警告] BGM の追加に失敗: {e}")
        else:
            print(f"  [警告] BGM ファイルが見つかりません: {bgm_path}")

    # ── 書き出し ──
    print(f"\n  書き出し中: {output_path}")
    final.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        threads=4,
        logger="bar",
    )

    # ── 後処理 ──
    final.close()
    for v in open_videos:
        try:
            v.close()
        except Exception:
            pass


# ============================================================
# マニフェスト保存
# ============================================================

def save_manifest(variant, selected_segments, total_sec, manifest_path):
    """
    バリアントの生成条件を JSON ファイルに保存する関数。

    保存される内容:
        - バリアント名・ラベル・生成日時
        - パラメータ（hook_mode, clip_sec, weights など）
        - 使用したクリップ一覧（ファイル名・時刻・全スコア）
        - 合計時間
    """
    manifest = {
        "variant":      variant["name"],
        "label":        variant.get("label", ""),
        "generated_at": datetime.datetime.now().isoformat(),
        "params": {
            "hook_mode":     variant.get("hook_mode", "loud"),
            "clip_sec":      variant.get("clip_sec", 3),
            "target_sec":    variant.get("target_sec", 60),
            "bgm":           variant.get("bgm", None),
            "crossfade_sec": variant.get("crossfade_sec", 0.3),
            # weights も保存して「どの重みで作ったか」を記録
            "weights":       variant.get("weights", DEFAULT_WEIGHTS),
        },
        "clips": [
            {
                "role":      seg.get("role", "body"),
                "source":    os.path.basename(seg["path"]),
                "start_sec": round(seg["start"], 2),
                "end_sec":   round(seg["end"], 2),
                # 全スコアを記録（あとで分析できるように）
                "scores": {
                    k: round(v, 4)
                    for k, v in seg.get("scores", {}).items()
                },
                "norm_scores": {
                    k: round(v, 4)
                    for k, v in seg.get("norm_scores", {}).items()
                },
            }
            for seg in selected_segments
        ],
        "total_duration_sec": round(total_sec, 2),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  マニフェスト保存: {manifest_path}")


# ============================================================
# review.csv の初期化
# ============================================================

def init_review_csv(variants):
    """
    experiments/review.csv を初期化する関数。
    ファイルが存在しない場合にのみ作成します（既存データは保護）。
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    if not os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "w", encoding="utf-8") as f:
            f.write("date,variant,filename,views,saves,avg_watch_sec,notes\n")
            for v in variants:
                name = v["name"]
                f.write(f"YYYY-MM-DD,{name},reel_{name}.mp4,0,0,0.0,投稿前\n")
        print(f"\n[✓] レビューシートを作成しました: {REVIEW_CSV}")
    else:
        print(f"\n[✓] レビューシートは既に存在します: {REVIEW_CSV}")


# ============================================================
# メイン処理
# ============================================================

def make_reel_ab():
    """
    ABテスト用リール生成のメイン処理。

    処理の流れ:
        [準備]    フォルダ確認・動画収集・variants.yaml 読み込み
        [設定]    Whisper ロード（use_whisper: true の場合）
        [分析]    全動画を分析（5種類のスコアを計算）
        [正規化]  全セグメントのスコアを 0〜1 に正規化
        [生成]    各バリアントでリールを生成・書き出し
        [記録]    マニフェスト JSON + review.csv を保存
    """
    print("=" * 60)
    print("  学校行事リールメーカー ABテスト版 (スコアリングエンジン搭載)")
    print("=" * 60)

    # ── フォルダ確認 ──
    for d in [CANDIDATES_DIR, MANIFESTS_DIR, EXPERIMENTS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"\n[!] '{INPUT_DIR}/' を作成しました。素材動画を入れて再実行してください。")
        sys.exit(0)

    # ── 動画ファイル収集 ──
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"\n[!] '{INPUT_DIR}/' に動画が見つかりません。")
        sys.exit(0)

    print(f"\n[✓] {len(video_files)} 本の素材動画を見つけました")

    # ── variants.yaml 読み込み ──
    config, variants = load_config_and_variants()
    print(f"[✓] {len(variants)} 個のバリアントを読み込みました:")
    for v in variants:
        w = v.get("weights", DEFAULT_WEIGHTS)
        print(f"    [{v['name']}] {v.get('label', '')}  "
              f"hook={v.get('hook_mode','loud')}  "
              f"clip={v.get('clip_sec',3)}秒  "
              f"target={v.get('target_sec',60)}秒")
        print(f"           weights: audio={w.get('w_audio',1.0):.1f}  "
              f"smile={w.get('w_smile',0.0):.1f}  "
              f"face_size={w.get('w_face_size',0.0):.1f}  "
              f"motion={w.get('w_motion',0.0):.1f}  "
              f"context={w.get('w_context',0.0):.1f}")

    # ── スコアエンジンの状態を表示 ──
    print(f"\n  スコアエンジンの状態:")
    print(f"    OpenCV (笑顔・顔サイズ): {'✓ 利用可能' if OPENCV_AVAILABLE else '✗ 未インストール (pip install opencv-python-headless)'}")
    print(f"    Whisper (文脈スコア)   : {'✓ 利用可能' if WHISPER_AVAILABLE else '✗ 未インストール (pip install openai-whisper)'}")

    # ── Whisper モデルのロード（use_whisper: true の場合のみ）──
    use_whisper      = config.get("use_whisper", False)
    whisper_size     = config.get("whisper_model", "tiny")
    whisper_model    = None

    if use_whisper:
        if WHISPER_AVAILABLE:
            print(f"\n  Whisper モデルをロードします（モデルサイズ: {whisper_size}）")
            whisper_model = load_whisper_model(whisper_size)
        else:
            print("\n  [警告] use_whisper: true ですが openai-whisper がインストールされていません。")
            print("         文脈スコアは 0.0 になります。")
    else:
        print(f"\n  [情報] Whisper は無効です (use_whisper: false)。")
        print(f"         有効にするには presets/variants.yaml の config セクションで")
        print(f"         use_whisper: true に変更してください。")

    # ── 全動画を分析 ──
    min_clip_sec = min(v.get("clip_sec", 3) for v in variants)
    print(f"\n[分析] 全動画を分析しています (セグメント長: {min_clip_sec}秒)")
    if OPENCV_AVAILABLE:
        print("       笑顔・顔サイズスコアも計算します（処理に時間がかかります）")

    all_segments = analyze_all_videos(video_files, clip_sec=min_clip_sec,
                                      whisper_model=whisper_model)

    if not all_segments:
        print("\n[エラー] 有効なセグメントが見つかりませんでした。")
        sys.exit(1)

    print(f"\n  合計 {len(all_segments)} セグメントが見つかりました")

    # ── スコアを正規化（全セグメントにわたって 0〜1 に揃える）──
    print("  スコアを正規化しています...")
    normalize_all_scores(all_segments)

    # 参考: 各スコアのトップ3を表示
    for score_key, label in [("audio", "音量"), ("motion", "動き"),
                               ("smile", "笑顔"), ("face_size", "顔サイズ")]:
        top = sorted(all_segments, key=lambda s: s["scores"].get(score_key, 0),
                     reverse=True)[:3]
        if any(s["scores"].get(score_key, 0) > 0 for s in top):
            print(f"\n  {label}スコアトップ3:")
            for i, s in enumerate(top):
                print(f"    {i+1}. {os.path.basename(s['path'])}"
                      f"  {s['start']:.1f}s〜{s['end']:.1f}s"
                      f"  raw={s['scores'].get(score_key,0):.4f}"
                      f"  norm={s['norm_scores'].get(score_key,0):.3f}")

    # ── 各バリアントのリールを生成 ──
    print(f"\n{'='*60}")
    print(f"  各バリアントのリールを生成します")
    print(f"{'='*60}")

    for idx, variant in enumerate(variants):
        name          = variant["name"]
        hook_mode     = variant.get("hook_mode", "loud")
        target_sec    = variant.get("target_sec", 60)
        crossfade_sec = variant.get("crossfade_sec", 0.3)
        bgm_path      = variant.get("bgm", None)
        weights       = variant.get("weights", DEFAULT_WEIGHTS)

        output_path   = os.path.join(CANDIDATES_DIR, f"reel_{name}.mp4")
        manifest_path = os.path.join(MANIFESTS_DIR,  f"reel_{name}.json")

        print(f"\n[{idx+1}/{len(variants)}] バリアント {name}: {variant.get('label', '')}")

        # フックセグメントを選択
        hook_seg = select_hook_segment(all_segments, hook_mode)
        if hook_seg:
            print(f"  フック: {os.path.basename(hook_seg['path'])}"
                  f"  {hook_seg['start']:.1f}s〜{hook_seg['end']:.1f}s"
                  f"  (hook_mode={hook_mode})")

        # 本編セグメントを総合スコア順で選択
        selected, total = select_segments_for_variant(
            all_segments, hook_seg, target_sec, weights
        )
        print(f"  → {len(selected)} シーンを選択 (合計: {total:.1f}秒)")

        # 動画を書き出す
        build_and_export(selected, output_path, crossfade_sec, bgm_path)

        # マニフェストを保存
        save_manifest(variant, selected, total, manifest_path)

        print(f"  [完了] {output_path}")

    # ── review.csv を初期化 ──
    init_review_csv(variants)

    # ── 完了メッセージ ──
    print("\n" + "=" * 60)
    print("  [完了] ABテスト用リールの生成が終わりました!")
    print(f"\n  生成されたファイル:")
    for v in variants:
        print(f"    output/candidates/reel_{v['name']}.mp4")
    print(f"\n  次のステップ:")
    print(f"    1. output/candidates/ の動画を Instagram に投稿する")
    print(f"    2. 投稿後に experiments/review.csv へ数値を記入する")
    print(f"    3. python recommend_next.py でおすすめ設定を確認する")
    print("=" * 60)


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    make_reel_ab()
