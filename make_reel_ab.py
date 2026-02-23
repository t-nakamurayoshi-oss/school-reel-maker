#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_reel_ab.py

ABテスト用に複数バリアントのリール動画を自動生成するスクリプト。

presets/variants.yaml に定義されたパラメータで複数のリールを生成し、
output/candidates/ に保存します。

【使い方】
    1. input/ フォルダに動画ファイルを入れる
    2. presets/variants.yaml でバリアントを設定する（デフォルトでA/B/Cの3本）
    3. python make_reel_ab.py を実行する
    4. output/candidates/reel_A.mp4, reel_B.mp4, ... が生成される
    5. 各動画の生成条件が output/manifests/reel_A.json などに保存される
    6. 3本を投稿して results を experiments/review.csv に記入する
    7. python recommend_next.py で次回おすすめ設定を確認する

【make_reel.py との違い】
    - 1本ではなく複数のバリアントを一括生成
    - 音量スコアに加えて「動きスコア」も計算
    - 冒頭のフックシーンをバリアントごとに異なる方法で選択
    - 生成条件をマニフェスト (JSON) に記録して再現・比較を可能にする
"""

import os           # ファイルパスの操作に使う
import sys          # スクリプト終了 (sys.exit) に使う
import glob         # ワイルドカードでファイルを検索する
import json         # マニフェストを JSON 形式で保存する
import datetime     # 生成日時の記録に使う

import numpy as np          # 数値計算（音量・動きスコア）に使う
import yaml                 # variants.yaml を読み込む (pip install pyyaml)
from moviepy.editor import (
    VideoFileClip,          # 動画ファイルを読み込むクラス
    concatenate_videoclips, # 複数クリップをつなぐ関数
    AudioFileClip,          # BGM ファイルを読み込むクラス
)

# make_reel.py から共通のユーティリティ関数を import する
# （コードの重複を避けるため、基本機能は make_reel.py に集約している）
from make_reel import (
    get_audio_rms,      # 音量 (RMS) を計算する関数
    crop_to_vertical,   # 縦型 (9:16) にクロップする関数
    VIDEO_EXTENSIONS,   # 対応する動画拡張子のリスト
    MIN_CLIP_DURATION,  # 最小クリップ長（秒）
    OUTPUT_WIDTH,       # 出力幅 (1080)
    OUTPUT_HEIGHT,      # 出力高さ (1920)
    OUTPUT_FPS,         # 出力フレームレート (30)
)

# ============================================================
# パス設定
# ここを変えると入出力先を変更できます
# ============================================================

INPUT_DIR       = "input"                               # 素材動画フォルダ
VARIANTS_FILE   = os.path.join("presets", "variants.yaml")  # バリアント設定
CANDIDATES_DIR  = os.path.join("output", "candidates") # リール出力先
MANIFESTS_DIR   = os.path.join("output", "manifests")  # マニフェスト出力先
EXPERIMENTS_DIR = "experiments"                         # 実験データフォルダ
REVIEW_CSV      = os.path.join(EXPERIMENTS_DIR, "review.csv")  # レビューシート

# ============================================================
# variants.yaml 読み込み
# ============================================================

def load_variants():
    """
    presets/variants.yaml を読み込んでバリアントのリストを返す関数。

    variants.yaml の例:
        variants:
          - name: "A"
            hook_mode: "loud"
            clip_sec: 3
            ...

    戻り値:
        list[dict]: バリアント設定の辞書リスト
    """
    if not os.path.exists(VARIANTS_FILE):
        print(f"[エラー] {VARIANTS_FILE} が見つかりません。")
        print(f"         presets/variants.yaml を確認してください。")
        sys.exit(1)

    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    variants = data.get("variants", [])
    if not variants:
        print("[エラー] variants.yaml にバリアントが定義されていません。")
        sys.exit(1)

    return variants


# ============================================================
# 動画分析: 音量スコア + 動きスコア
# ============================================================

def get_motion_score(clip):
    """
    動画クリップの「動きの大きさ」を数値化する関数。

    フレーム間のピクセル差分を計算することで、シーンにどれだけ
    動きがあるかを判定します。

    【原理】
        - 連続する2枚のフレームを引き算する
        - 差が大きい = ピクセルが多く変化した = 動きが大きい

    例:
        - 静止している黒板 → スコア低（変化が少ない）
        - 運動会のリレー   → スコア高（走る動きで多くのピクセルが変化）

    引数:
        clip: MoviePy のクリップオブジェクト

    戻り値:
        float: 動きスコア（大きいほど動きが激しい）
    """
    try:
        duration = clip.duration

        # クリップの 25%・50%・75% の時点のフレームをサンプリングする
        # （全フレームを取得すると処理が重いため、3点に絞る）
        sample_times = [duration * 0.25, duration * 0.5, duration * 0.75]

        # 各時刻のフレームを NumPy 配列として取得してグレースケールに変換
        # get_frame(t) は (高さ, 幅, 3チャンネル) の配列を返す
        frames = []
        for t in sample_times:
            t = min(t, duration - 0.01)  # 動画末尾を超えないように補正
            frame_rgb = clip.get_frame(t)
            # R, G, B の平均を取ってグレースケール（1チャンネル）にする
            # こうすると計算量が 1/3 になる
            frame_gray = frame_rgb.mean(axis=2)
            frames.append(frame_gray.astype(float))

        # 連続するフレーム間の平均絶対差分を計算
        diffs = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i] - frames[i - 1])
            diffs.append(diff.mean())

        return float(np.mean(diffs)) if diffs else 0.0

    except Exception as e:
        # フレーム取得に失敗した場合（破損ファイルなど）
        print(f"    [警告] 動きスコアの計算に失敗: {e}")
        return 0.0


def analyze_all_videos(video_files, clip_sec):
    """
    全動画ファイルを分析し、全セグメントのスコア情報を返す関数。

    各セグメントについて以下を計算します:
        - 音量スコア (rms): 大きいほど盛り上がっている
        - 動きスコア (motion): 大きいほど活発なシーン

    引数:
        video_files (list[str]): 動画ファイルのパスリスト
        clip_sec (float)       : 1セグメントの長さ（秒）

    戻り値:
        list[dict]: 全セグメントの情報リスト。各要素:
            - path    : 動画ファイルのパス
            - start   : 開始時間（秒）
            - end     : 終了時間（秒）
            - duration: 長さ（秒）
            - rms     : 音量スコア
            - motion  : 動きスコア
    """
    all_segments = []

    for video_path in video_files:
        print(f"\n  分析中: {os.path.basename(video_path)}")

        try:
            video = VideoFileClip(video_path)
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
                seg_dur = end_time - start_time

                if seg_dur < MIN_CLIP_DURATION:
                    break

                # 区間を切り出す
                seg_clip = video.subclip(start_time, end_time)

                # 音量スコアを計算
                rms = get_audio_rms(seg_clip)
                # 動きスコアを計算
                motion = get_motion_score(seg_clip)

                all_segments.append({
                    "path":     video_path,
                    "start":    start_time,
                    "end":      end_time,
                    "duration": seg_dur,
                    "rms":      rms,
                    "motion":   motion,
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
    バリアントの hook_mode に応じて冒頭シーンを1つ選ぶ関数。

    フック（hook）とは動画の最初の数秒で視聴者を引き込むシーンのこと。
    Instagram では最初の数秒が「続きを見るかどうか」の判断に直結します。

    hook_mode の意味:
        loud   → 歓声・盛り上がりを冒頭に → 音量が最大のセグメントを選択
        motion → 激しい動きを冒頭に      → 動きスコアが最大のセグメントを選択
        face   → 顔が大きく映るシーンを冒頭に → 将来実装予定（現在は loud にフォールバック）

    引数:
        segments (list[dict]): 全セグメントのリスト
        hook_mode (str)      : フックの選び方

    戻り値:
        dict: フックとして使うセグメント（1件）
    """
    if not segments:
        return None

    if hook_mode == "loud":
        # 音量（RMS）が最大のセグメントを選ぶ
        return max(segments, key=lambda s: s["rms"])

    elif hook_mode == "motion":
        # 動きスコアが最大のセグメントを選ぶ
        return max(segments, key=lambda s: s["motion"])

    elif hook_mode == "face":
        # 顔検出は将来の拡張として設計。
        # 実装するには OpenCV の Haar Cascade や MediaPipe を使う。
        # TODO: 顔サイズ・位置スコアを計算してここで返す
        print("    [情報] face モードは未実装のため、loud にフォールバックします。")
        return max(segments, key=lambda s: s["rms"])

    else:
        print(f"    [警告] 未知の hook_mode: '{hook_mode}'。loud にフォールバックします。")
        return max(segments, key=lambda s: s["rms"])


# ============================================================
# セグメント選択（フック + 本編）
# ============================================================

def select_segments_for_variant(all_segments, hook_segment, target_sec):
    """
    フックセグメントを先頭に配置し、残りを音量スコア順で埋める関数。

    【組み立て方】
        1. フックセグメント（hook_mode で選んだ最良のシーン）を先頭に
        2. 残りの時間を音量の大きい順に補充する
        3. 合計が target_sec 以上になったら終了

    引数:
        all_segments (list[dict]): 全セグメントのリスト
        hook_segment (dict)      : 冒頭に配置するセグメント
        target_sec (float)       : 目標の合計時間（秒）

    戻り値:
        (list[dict], float): 選ばれたセグメントリストと合計時間のタプル
    """
    selected = []
    total_sec = 0.0

    # ── 1. フックセグメントを先頭に追加 ──
    if hook_segment:
        hook_copy = dict(hook_segment)
        hook_copy["role"] = "hook"  # フックであることをマーキング
        selected.append(hook_copy)
        total_sec += hook_segment["duration"]

    # ── 2. 残りを音量の大きい順に追加 ──
    # 音量順に並べたコピーを作成（元のリストは変えない）
    body_candidates = sorted(all_segments, key=lambda s: s["rms"], reverse=True)

    for seg in body_candidates:
        if total_sec >= target_sec:
            break

        # フックセグメントと同じ区間は二重に入れない
        is_same_as_hook = (
            hook_segment is not None
            and seg["path"] == hook_segment["path"]
            and seg["start"] == hook_segment["start"]
        )
        if is_same_as_hook:
            continue

        seg_copy = dict(seg)
        seg_copy["role"] = "body"  # 本編部分をマーキング
        selected.append(seg_copy)
        total_sec += seg["duration"]

    return selected, total_sec


# ============================================================
# 動画書き出し（BGM オプション付き）
# ============================================================

def build_and_export(selected_segments, output_path, crossfade_sec, bgm_path=None):
    """
    選択されたセグメントをつなぎ合わせて動画ファイルに書き出す関数。

    引数:
        selected_segments (list[dict]): 使用するセグメントのリスト（先頭がフック）
        output_path (str)             : 出力ファイルのパス（例: output/candidates/reel_A.mp4）
        crossfade_sec (float)         : クロスフェードの長さ（秒）
        bgm_path (str or None)        : BGM ファイルのパス。None の場合は元音声のまま。
    """
    open_videos = []  # 後でクローズするために追跡するリスト
    clips = []

    for i, seg in enumerate(selected_segments):
        role_tag = "★フック" if seg.get("role") == "hook" else f" 本編{i:02d}"
        fname = os.path.basename(seg["path"])
        print(f"  [{role_tag}] {fname}  {seg['start']:.1f}s〜{seg['end']:.1f}s")

        # 動画を開いて区間を切り出す
        video = VideoFileClip(seg["path"])
        open_videos.append(video)
        clip = video.subclip(seg["start"], seg["end"])
        clip_v = crop_to_vertical(clip)
        clips.append(clip_v)

    if not clips:
        print("  [エラー] 有効なクリップがありません。このバリアントをスキップします。")
        return

    # ── クリップを結合（クロスフェードあり or ブツ切り）──
    if len(clips) > 1 and crossfade_sec > 0:
        # 2枚目以降に crossfadein を適用してなめらかにつなぐ
        faded_clips = [clips[0]]
        for c in clips[1:]:
            faded_clips.append(c.crossfadein(crossfade_sec))
        final = concatenate_videoclips(
            faded_clips, padding=-crossfade_sec, method="compose"
        )
    else:
        final = concatenate_videoclips(clips, method="compose")

    # ── BGM を追加（bgm_path が指定されている場合）──
    if bgm_path:
        if os.path.exists(bgm_path):
            print(f"  BGM を追加中: {bgm_path}")
            try:
                from moviepy.editor import CompositeAudioClip

                bgm = AudioFileClip(bgm_path)
                # 動画の長さに合わせて BGM をループまたはトリム
                if bgm.duration < final.duration:
                    # BGM が短い場合: ループ再生（単純に繰り返す）
                    repeat_count = int(final.duration / bgm.duration) + 1
                    from moviepy.audio.fx.all import audio_loop
                    bgm = audio_loop(bgm, nloops=repeat_count)
                bgm = bgm.subclip(0, final.duration)

                # BGM を 30% に下げて元音声と合成（元音声の方が聞こえるように）
                bgm = bgm.volumex(0.3)
                if final.audio is not None:
                    mixed_audio = CompositeAudioClip([final.audio, bgm])
                else:
                    mixed_audio = bgm
                final = final.set_audio(mixed_audio)

            except Exception as e:
                print(f"  [警告] BGM の追加に失敗しました: {e}")
        else:
            print(f"  [警告] BGM ファイルが見つかりません: {bgm_path}")

    # ── 動画ファイルとして書き出す ──
    print(f"\n  書き出し中: {output_path}")
    print("  (処理に数分かかる場合があります...)\n")

    final.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        threads=4,
        logger="bar",
    )

    # ── 後処理: 開いたオブジェクトを全てクローズ ──
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
    バリアントの生成条件をJSONファイルに保存する関数。

    マニフェストとは「この動画はどんな設定で作られたか」の記録です。
    experiments/review.csv と組み合わせることで、
    「どの設定が良かったか」を後から分析できます。

    保存される内容:
        - バリアント名・説明ラベル
        - 生成日時
        - パラメータ（hook_mode, clip_sec, target_sec, ...）
        - 使用したクリップ一覧（ファイル名・開始終了時間・スコア）
        - 合計時間

    引数:
        variant (dict)               : variants.yaml のバリアント設定
        selected_segments (list[dict]): 使用したセグメントのリスト
        total_sec (float)            : 合計時間（秒）
        manifest_path (str)          : 保存先のパス
    """
    manifest = {
        # ── 基本情報 ──
        "variant":      variant["name"],
        "label":        variant.get("label", ""),
        "generated_at": datetime.datetime.now().isoformat(),

        # ── 生成パラメータ（variants.yaml の設定をそのまま記録）──
        "params": {
            "hook_mode":     variant.get("hook_mode", "loud"),
            "clip_sec":      variant.get("clip_sec", 3),
            "target_sec":    variant.get("target_sec", 60),
            "bgm":           variant.get("bgm", None),
            "crossfade_sec": variant.get("crossfade_sec", 0.3),
        },

        # ── 使用したクリップの一覧 ──
        # これを見ると「どの動画の何秒のシーンを使ったか」が分かる
        "clips": [
            {
                "role":      seg.get("role", "body"),        # hook / body
                "source":    os.path.basename(seg["path"]),  # ファイル名
                "start_sec": round(seg["start"], 2),
                "end_sec":   round(seg["end"], 2),
                "rms":       round(seg["rms"], 4),           # 音量スコア
                "motion":    round(seg.get("motion", 0.0), 4),  # 動きスコア
            }
            for seg in selected_segments
        ],

        # ── 合計時間 ──
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

    ファイルが存在しない場合にだけヘッダーとサンプル行を書き込みます。
    すでに存在する場合は何もしません（既存データを消さない）。

    CSV の列:
        date         : 投稿日（YYYY-MM-DD）
        variant      : バリアント名（A, B, C, ...）
        filename     : ファイル名（reel_A.mp4）
        views        : 再生数
        saves        : 保存数（最重要指標）
        avg_watch_sec: 平均視聴時間（秒）
        notes        : メモ（投稿タイミング・キャプションなど）

    引数:
        variants (list[dict]): バリアントのリスト（ファイル名生成に使用）
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    if not os.path.exists(REVIEW_CSV):
        with open(REVIEW_CSV, "w", encoding="utf-8") as f:
            # ヘッダー行
            f.write("date,variant,filename,views,saves,avg_watch_sec,notes\n")
            # 各バリアントのサンプル行（まだ投稿していない状態）
            for v in variants:
                name = v["name"]
                f.write(f"YYYY-MM-DD,{name},reel_{name}.mp4,0,0,0.0,投稿前\n")

        print(f"\n[✓] レビューシートを作成しました: {REVIEW_CSV}")
        print("    投稿後にこのCSVファイルに結果を記入してください。")
    else:
        print(f"\n[✓] レビューシートは既に存在します: {REVIEW_CSV}")


# ============================================================
# メイン処理
# ============================================================

def make_reel_ab():
    """
    ABテスト用リール生成のメイン処理。

    処理の流れ:
        [準備]      フォルダの確認・作成
        [収集]      input/ から動画ファイルを収集
        [設定読込]  presets/variants.yaml を読み込む
        [分析]      全動画を分析（音量・動きスコア）
        [生成]      各バリアントでリールを生成・書き出し
        [記録]      マニフェスト (JSON) を保存
        [後処理]    experiments/review.csv を初期化
    """
    print("=" * 60)
    print("  学校行事リールメーカー ABテスト版")
    print("=" * 60)

    # ── フォルダを確認・作成 ──
    for d in [CANDIDATES_DIR, MANIFESTS_DIR, EXPERIMENTS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"\n[!] '{INPUT_DIR}/' フォルダを作成しました。")
        print(f"    素材動画をここに入れてから再実行してください。")
        sys.exit(0)

    # ── 動画ファイルを収集 ──
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"\n[!] '{INPUT_DIR}/' フォルダに動画が見つかりません。")
        sys.exit(0)

    print(f"\n[✓] {len(video_files)} 本の素材動画を見つけました")

    # ── variants.yaml を読み込む ──
    variants = load_variants()
    print(f"[✓] {len(variants)} 個のバリアントを読み込みました:")
    for v in variants:
        print(f"    [{v['name']}] {v.get('label', '')}  "
              f"hook={v.get('hook_mode','loud')}  "
              f"clip={v.get('clip_sec',3)}秒  "
              f"target={v.get('target_sec',60)}秒")

    # ── 全動画を分析 ──
    # バリアントごとに clip_sec が異なる場合は、最も短い clip_sec で分析する。
    # こうすることで全バリアントに対応できる。
    min_clip_sec = min(v.get("clip_sec", 3) for v in variants)

    print(f"\n[分析] 全動画を分析しています (セグメント長: {min_clip_sec}秒)")
    print("      (動画の本数・長さによっては数分かかります)")

    all_segments = analyze_all_videos(video_files, clip_sec=min_clip_sec)

    if not all_segments:
        print("\n[エラー] 有効なセグメントが見つかりませんでした。")
        print("         動画ファイルが壊れていないか確認してください。")
        sys.exit(1)

    print(f"\n  合計 {len(all_segments)} セグメントが見つかりました")

    # 参考: 音量・動きのトップ3を表示
    top_rms    = sorted(all_segments, key=lambda s: s["rms"],    reverse=True)[:3]
    top_motion = sorted(all_segments, key=lambda s: s["motion"], reverse=True)[:3]

    print("\n  音量トップ3:")
    for i, s in enumerate(top_rms):
        print(f"    {i+1}. {os.path.basename(s['path'])}  "
              f"{s['start']:.1f}s〜{s['end']:.1f}s  RMS={s['rms']:.4f}")

    print("  動きスコアトップ3:")
    for i, s in enumerate(top_motion):
        print(f"    {i+1}. {os.path.basename(s['path'])}  "
              f"{s['start']:.1f}s〜{s['end']:.1f}s  motion={s['motion']:.2f}")

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

        output_path   = os.path.join(CANDIDATES_DIR, f"reel_{name}.mp4")
        manifest_path = os.path.join(MANIFESTS_DIR,  f"reel_{name}.json")

        print(f"\n[{idx+1}/{len(variants)}] バリアント {name}: {variant.get('label', '')}")
        print(f"  hook_mode={hook_mode}  target={target_sec}秒  crossfade={crossfade_sec}秒")

        # フックセグメントを選択（冒頭に配置するシーン）
        hook_seg = select_hook_segment(all_segments, hook_mode)
        if hook_seg:
            print(f"  フック候補: {os.path.basename(hook_seg['path'])}  "
                  f"{hook_seg['start']:.1f}s〜{hook_seg['end']:.1f}s")

        # 本編セグメントを選択（フック + 音量順の本編）
        selected, total = select_segments_for_variant(
            all_segments, hook_seg, target_sec
        )
        print(f"  → {len(selected)} シーンを選択 (合計: {total:.1f}秒)")

        # 動画を組み立てて書き出す
        build_and_export(selected, output_path, crossfade_sec, bgm_path)

        # マニフェストを保存
        save_manifest(variant, selected, total, manifest_path)

        print(f"  [完了] {output_path}")

    # ── review.csv を初期化（初回のみ作成）──
    init_review_csv(variants)

    # ── 完了メッセージ ──
    print("\n" + "=" * 60)
    print("  [完了] ABテスト用リールの生成が終わりました!")
    print(f"\n  生成されたファイル:")
    for v in variants:
        print(f"    output/candidates/reel_{v['name']}.mp4")
    print(f"\n  次のステップ:")
    print(f"    1. output/candidates/ の動画を Instagram に投稿する")
    print(f"    2. 数日後に投稿の数値を experiments/review.csv に記入する")
    print(f"       (views, saves, avg_watch_sec の列に数値を入力)")
    print(f"    3. python recommend_next.py で次回おすすめ設定を確認する")
    print("=" * 60)


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    make_reel_ab()
