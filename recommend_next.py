#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_next.py

experiments/review.csv の結果をもとに、次回リール生成のおすすめ設定を提案する
簡易レコメンドスクリプト。

【使い方】
    1. make_reel_ab.py で複数バリアントのリールを生成する
    2. 各リールを Instagram に投稿する
    3. 数日後に投稿の数値を experiments/review.csv に記入する
    4. python recommend_next.py を実行する
    5. 「次回おすすめ設定」が表示される

【review.csv の記入例】
    date,variant,filename,views,saves,avg_watch_sec,notes
    2024-04-01,A,reel_A.mp4,1500,80,45.2,始業式の動画
    2024-04-01,B,reel_B.mp4,1200,55,38.7,始業式の動画
    2024-04-01,C,reel_C.mp4,2000,70,28.1,始業式の動画

【評価指標の優先順位】
    1位: 保存数 (saves)      ← 最重要。保存 = ファイルとして価値を認めた証拠
    2位: 平均視聴時間 (avg_watch_sec) ← 最後まで見てもらえたか
    3位: 再生数 (views)     ← 露出の大きさ

【将来の拡張アイデア】
    - review.csv の蓄積データから自動で重みを最適化（機械学習）
    - 季節・行事種別ごとにベストパラメータを記憶
    - Bayesian Optimization による高効率なA/Bテスト
"""

import os
import csv
import json
import sys

# ============================================================
# パス設定
# ============================================================

REVIEW_CSV    = os.path.join("experiments", "review.csv")
MANIFESTS_DIR = os.path.join("output", "manifests")

# ============================================================
# 評価スコアの重み設定
#
# ここを変えると評価の優先順位が変わります。
# 3つの重みの合計を 1.0 にすると分かりやすいです。
#
# 例: 「とにかく保存数を上げたい」場合は WEIGHT_SAVES を上げる
#     WEIGHT_SAVES = 0.7, WEIGHT_AVG_WATCH = 0.2, WEIGHT_VIEWS = 0.1
# ============================================================

WEIGHT_SAVES      = 0.5   # 保存数の重み（最重要）
WEIGHT_AVG_WATCH  = 0.3   # 平均視聴時間の重み
WEIGHT_VIEWS      = 0.2   # 再生数の重み


# ============================================================
# データ読み込み
# ============================================================

def load_review_data():
    """
    experiments/review.csv を読み込んでデータのリストを返す関数。

    戻り値:
        list[dict]: CSV の各行を辞書として格納したリスト
                    例: [{"date": "2024-04-01", "variant": "A", ...}, ...]
    """
    if not os.path.exists(REVIEW_CSV):
        print(f"[エラー] {REVIEW_CSV} が見つかりません。")
        print(f"         まず make_reel_ab.py を実行してください。")
        sys.exit(1)

    rows = []
    with open(REVIEW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"[エラー] {REVIEW_CSV} にデータがありません。")
        sys.exit(1)

    return rows


def load_manifest(variant_name):
    """
    指定したバリアントのマニフェスト JSON を読み込む関数。

    マニフェストには「どんな設定でリールを作ったか」が記録されています。
    ベストバリアントの設定を読み取り、次回への提案に使います。

    引数:
        variant_name (str): バリアント名（例: "A", "B"）

    戻り値:
        dict: マニフェストの内容。ファイルが存在しない場合は None。
    """
    path = os.path.join(MANIFESTS_DIR, f"reel_{variant_name}.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# スコア計算
# ============================================================

def normalize(values):
    """
    数値のリストを 0〜1 の範囲に正規化する関数（Min-Max 正規化）。

    【なぜ正規化するのか？】
        views（再生数）は数千〜数万の大きな値。
        saves（保存数）は数十〜数百の小さな値。
        そのまま足すと views の影響が大きくなりすぎるため、
        全指標を 0〜1 の同じスケールに揃えてから重み付けします。

    例:
        [100, 200, 300] → [0.0, 0.5, 1.0]
        [50, 50, 50]    → [0.5, 0.5, 0.5]  （全て同じ場合は 0.5 固定）

    引数:
        values (list[float]): 正規化したい数値のリスト

    戻り値:
        list[float]: 0〜1 に正規化されたリスト
    """
    if not values:
        return []

    min_v = min(values)
    max_v = max(values)

    if max_v == min_v:
        # 全ての値が同じ場合: 差がないので全て 0.5 にする
        return [0.5] * len(values)

    # Min-Max 正規化: (値 - 最小値) / (最大値 - 最小値)
    return [(v - min_v) / (max_v - min_v) for v in values]


def parse_valid_rows(rows):
    """
    CSV データから有効な行（投稿済みで数値が入っている行）だけを抽出する関数。

    「投稿前」の行（views=0 かつ saves=0）は除外します。

    引数:
        rows (list[dict]): CSV の全行

    戻り値:
        list[dict]: 有効な行のみ。各行に数値フィールド (_saves, _avg_w, _views) を追加。
    """
    valid = []
    for row in rows:
        try:
            saves = float(row.get("saves", 0) or 0)
            avg_w = float(row.get("avg_watch_sec", 0) or 0)
            views = float(row.get("views", 0) or 0)

            # views も saves も 0 の行 = まだ投稿していないのでスキップ
            if saves == 0 and views == 0:
                continue

            valid.append({
                **row,
                "_saves": saves,
                "_avg_w": avg_w,
                "_views": views,
            })
        except (ValueError, TypeError):
            # 数値に変換できない行（記入ミスなど）はスキップ
            continue

    return valid


def compute_scores(valid_rows):
    """
    有効な行に対して総合スコアを計算する関数。

    【スコア計算の流れ】
        1. 保存数・平均視聴時間・再生数をそれぞれ 0〜1 に正規化
        2. 重み（WEIGHT_*）を掛けて足し合わせる
        3. 各行に "_score" キーでスコアを追加する

    引数:
        valid_rows (list[dict]): 有効な行のリスト（parse_valid_rows の戻り値）

    戻り値:
        list[dict]: "_score" キーを追加した行のリスト
    """
    saves_norm = normalize([r["_saves"] for r in valid_rows])
    avg_w_norm = normalize([r["_avg_w"] for r in valid_rows])
    views_norm = normalize([r["_views"] for r in valid_rows])

    for i, row in enumerate(valid_rows):
        row["_score"] = (
            WEIGHT_SAVES     * saves_norm[i]
            + WEIGHT_AVG_WATCH * avg_w_norm[i]
            + WEIGHT_VIEWS     * views_norm[i]
        )

    return valid_rows


def aggregate_by_variant(scored_rows):
    """
    同じバリアントの行が複数ある場合に平均スコアで集計する関数。

    複数回の投稿データがある場合、バリアントごとの平均をとることで
    より信頼性の高い比較ができます。

    引数:
        scored_rows (list[dict]): スコア計算済みの行リスト

    戻り値:
        dict: {バリアント名: {"avg_score": ..., "avg_saves": ..., ...}} の辞書
    """
    # バリアント名ごとにデータをまとめる
    variant_data = {}
    for row in scored_rows:
        v = row.get("variant", "?")
        if v not in variant_data:
            variant_data[v] = {"scores": [], "saves": [], "avg_w": [], "views": [], "rows": []}
        variant_data[v]["scores"].append(row["_score"])
        variant_data[v]["saves"].append(row["_saves"])
        variant_data[v]["avg_w"].append(row["_avg_w"])
        variant_data[v]["views"].append(row["_views"])
        variant_data[v]["rows"].append(row)

    # 各バリアントの平均値を計算
    aggregated = {}
    for v, data in variant_data.items():
        n = len(data["scores"])
        aggregated[v] = {
            "variant":   v,
            "avg_score": sum(data["scores"]) / n,
            "avg_saves": sum(data["saves"])  / n,
            "avg_avg_w": sum(data["avg_w"])  / n,
            "avg_views": sum(data["views"])  / n,
            "n_records": n,  # 何回分のデータか
        }

    return aggregated


# ============================================================
# 表示・提案
# ============================================================

def print_ranking_table(aggregated):
    """
    バリアントの評価結果を表形式で表示する関数。

    引数:
        aggregated (dict): aggregate_by_variant() の戻り値
    """
    # スコア降順に並べる
    ranked = sorted(aggregated.values(), key=lambda x: x["avg_score"], reverse=True)

    print("\n┌─ バリアント評価結果 " + "─" * 40 + "┐")
    print(f"  {'順位':<4}{'バリアント':<10}{'保存数':<10}{'平均視聴(秒)':<14}{'再生数':<10}{'総合スコア':<12}{'データ数'}")
    print("  " + "─" * 64)

    for rank, r in enumerate(ranked, start=1):
        medal = "★" if rank == 1 else ("  " if rank > 1 else "  ")
        print(
            f"  {medal}{rank:<3}"
            f"{r['variant']:<10}"
            f"{r['avg_saves']:<10.0f}"
            f"{r['avg_avg_w']:<14.1f}"
            f"{r['avg_views']:<10.0f}"
            f"{r['avg_score']:<12.3f}"
            f"{r['n_records']}件"
        )

    print("└" + "─" * 65 + "┘")

    return ranked


def print_recommendation(ranked, aggregated):
    """
    ベストバリアントに基づいて次回のおすすめ設定を表示する関数。

    引数:
        ranked (list[dict]): スコア順に並んだバリアント情報
        aggregated (dict)  : aggregate_by_variant() の戻り値
    """
    if not ranked:
        return

    best = ranked[0]
    best_variant = best["variant"]

    print(f"\n★ 最もパフォーマンスが良かったバリアント: {best_variant}")
    print(f"   平均保存数:     {best['avg_saves']:.0f}")
    print(f"   平均視聴時間:   {best['avg_avg_w']:.1f}秒")
    print(f"   平均再生数:     {best['avg_views']:.0f}")
    print(f"   総合スコア:     {best['avg_score']:.3f}")

    # マニフェストからパラメータを読み取って表示
    manifest = load_manifest(best_variant)
    if manifest:
        params = manifest.get("params", {})
        print(f"\n  このバリアントの設定（次回も参考にしてください）:")
        print(f"    hook_mode    : {params.get('hook_mode', '不明')}")
        print(f"    clip_sec     : {params.get('clip_sec', '不明')} 秒")
        print(f"    target_sec   : {params.get('target_sec', '不明')} 秒")
        print(f"    crossfade_sec: {params.get('crossfade_sec', '不明')} 秒")
        bgm = params.get("bgm")
        if bgm:
            print(f"    bgm          : {bgm}")

    # ── 差分分析 ──
    print(f"\n─── 改善のヒント {'─'*43}")

    if len(ranked) >= 2:
        second = ranked[1]
        score_gap = best["avg_score"] - second["avg_score"]

        if score_gap > 0.25:
            # 大差でリードしている場合
            print(f"  バリアント {best_variant} が大差でリードしています。")
            print(f"  次回は {best_variant} の設定をベースに")
            print(f"  clip_sec や crossfade_sec を少し調整してみましょう。")
        elif score_gap > 0.05:
            # 差が小さい場合
            print(f"  バリアント {best_variant} と {second['variant']} の差は僅差です。")
            print(f"  両方の良いところを組み合わせた新バリアントを試してみましょう。")
            print(f"  例: {best_variant} の hook_mode + {second['variant']} の clip_sec")
        else:
            # ほぼ同スコアの場合
            print(f"  全バリアントのスコアがほぼ同じです。")
            print(f"  投稿のタイミング（時間帯・曜日）やキャプションを変えて")
            print(f"  より多くのデータを集めてから再分析してください。")
    else:
        print(f"  データが1バリアントしかありません。")
        print(f"  複数のバリアントを比較するとより正確な提案ができます。")

    # ── 各指標の最良バリアントを表示 ──
    best_saves = max(aggregated.values(), key=lambda x: x["avg_saves"])
    best_watch = max(aggregated.values(), key=lambda x: x["avg_avg_w"])
    best_views = max(aggregated.values(), key=lambda x: x["avg_views"])

    print(f"\n  指標別ベストバリアント:")
    print(f"    保存数が最多     : バリアント {best_saves['variant']} ({best_saves['avg_saves']:.0f}件)")
    print(f"    視聴時間が最長   : バリアント {best_watch['variant']} ({best_watch['avg_avg_w']:.1f}秒)")
    print(f"    再生数が最多     : バリアント {best_views['variant']} ({best_views['avg_views']:.0f}回)")


def print_score_formula():
    """
    スコア計算式を表示する関数。透明性のために式を明示する。
    """
    print(f"\n─── スコア計算式 {'─'*44}")
    print(f"  総合スコア = 保存数×{WEIGHT_SAVES}"
          f" + 平均視聴時間×{WEIGHT_AVG_WATCH}"
          f" + 再生数×{WEIGHT_VIEWS}")
    print(f"  （各指標は 0〜1 に正規化した後に重み付けして合計）")
    print(f"\n  重みを変えるには recommend_next.py の WEIGHT_* を編集してください。")
    print(f"  例: 「保存数をもっと重視したい」場合")
    print(f"       WEIGHT_SAVES = 0.7  # 上げる")
    print(f"       WEIGHT_AVG_WATCH = 0.2  # 少し下げる")
    print(f"       WEIGHT_VIEWS = 0.1  # 下げる")


# ============================================================
# メイン処理
# ============================================================

def recommend():
    """
    レコメンドのメイン処理。

    処理の流れ:
        1. review.csv を読み込む
        2. 投稿済みの有効な行を抽出する
        3. 各行のスコアを計算する
        4. バリアントごとに集計する
        5. 評価テーブルを表示する
        6. 次回おすすめ設定を提案する
    """
    print("=" * 60)
    print("  リールメーカー: 次回おすすめ設定レコメンダー")
    print("=" * 60)

    # ── データ読み込み ──
    rows = load_review_data()
    print(f"\n[✓] {REVIEW_CSV} から {len(rows)} 行のデータを読み込みました")

    # ── 有効な行だけ抽出（投稿済みのもの）──
    valid_rows = parse_valid_rows(rows)

    if not valid_rows:
        print("\n[!] 投稿済みのデータが見つかりません。")
        print(f"    {REVIEW_CSV} に投稿後の数値を記入してください。")
        print("\n    記入例（CSVをテキストエディタで開いて編集）:")
        print("    date,variant,filename,views,saves,avg_watch_sec,notes")
        print("    2024-04-01,A,reel_A.mp4,1500,80,45.2,始業式の動画")
        print("    2024-04-01,B,reel_B.mp4,1200,55,38.7,始業式の動画")
        print("    2024-04-01,C,reel_C.mp4,2000,70,28.1,始業式の動画")
        return

    print(f"  うち投稿済み（有効データ）: {len(valid_rows)} 行")

    # ── スコア計算 ──
    scored_rows = compute_scores(valid_rows)

    # ── バリアントごとに集計 ──
    aggregated = aggregate_by_variant(scored_rows)

    # ── 結果テーブルを表示 ──
    ranked = print_ranking_table(aggregated)

    # ── 次回おすすめ設定を表示 ──
    print_recommendation(ranked, aggregated)

    # ── スコア計算式を表示 ──
    print_score_formula()

    print("\n" + "=" * 60)


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    recommend()
