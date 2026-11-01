"""提出用動画を組み立てる。

runs/<run>/video_*.mp4（Recorder の録画: 左 俯瞰 960x540 / 右上 手首 320x180 / 右 ASCII キャプション）から
映像部分だけを切り出し、日本語の字幕（タスク・ステップ一覧・VLM の判断）を PIL で描き直して 1 本にする。
ステップ番号は録画内の "step N:" 文字（ASCII）をテンプレート照合で読む。

  python scripts/make_submission_video.py            # → ../video/submission.mp4
  python scripts/make_submission_video.py --out /path/to/out.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402
from perception.camera import Camera  # noqa: E402
from perception.vlm import draw_grid  # noqa: E402

CFG = load_config()
MASK = Camera(dict(CFG["camera"], backend="mock"))   # 録画と同じ黒塗り（blur_cells）を VLM 入力画像のサムネイルにも適用
GRID = (CFG["vlm"].get("grid_cols", 10), CFG["vlm"].get("grid_rows", 7))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 10

# 動画に入れる実行（順番どおり）。speed: 再生倍率（2 なら 2 倍速と表示）
SEGMENTS = [
    dict(run="20261102_020059_run", task="青い消しゴムをカレンダーの上に乗せて", label="タスク 1（成功）", speed=1,
         note="教示範囲の外（カレンダー）へ極座標 IK モデルで到達。接触検知で置いて退避、俯瞰で確認"),
    dict(run="20261102_021619_run", task="青い消しゴムをコップに入れて", label="タスク 2（成功）", speed=1,
         note="陶器のコップ: 俯瞰で把持物を追跡して口の真上へ → 口の 4cm 上まで低トルクで降下 → 開放",
         wrist_mask=[(15.0, 22.0, 220, 0, 320, 80)]),   # 手首映像の右上に本棚が映る区間を黒塗り (t0, t1, x0, y0, x1, y1) 320x180 座標
    dict(run="20261102_020528_run", task="青い消しゴムをコップに入れて（改良前）", label="失敗例", speed=2,
         before="改良前: 俯瞰画像での位置合わせをせず、18cm の運搬高さからそのまま開放していた（運搬姿勢では FK の誤差で先端が約 6cm ずれる）",
         before_short="改良前: 俯瞰での位置合わせなしに 18cm からそのまま開放（運搬姿勢の FK 誤差で約 6cm ずれる）",
         note="結果: 消しゴムが縁に当たって弾かれ、拾い直しに行く途中で人の手を認識して停止。改良後（タスク 2）は俯瞰で把持物を追跡して口の真上へ合わせ、口の 4cm 上まで降りてから開放"),
]

ACTION_JA = {"goto_cell": "セルへ移動", "move": "手首画像で微調整", "shift": "俯瞰画像で微調整", "rotate": "手首を回転",
             "grasp": "掴む（接触検知）", "hover": "6cm へ降りる", "place": "置く（低トルク）", "drop": "容器に入れる",
             "done": "完了判定", "home": "退避"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def draw_text(img: np.ndarray, xy, text: str, size: int, color=(255, 255, 255), bold=False, max_w: int | None = None):
    """BGR ndarray に日本語テキストを描く（折り返しつき）。戻り値: 次の行の y"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil); f = font(size, bold)
    x, y = xy
    lines = []
    if max_w:
        # 英数字の連なり（18cm, goto_cell など）は途中で折り返さない: トークン単位で詰める
        import re as _re
        tokens = _re.findall(r"[A-Za-z0-9_/.\-]+|.", text)
        cur = ""
        for tok in tokens:
            if cur and tok not in "、。）」" and d.textlength(cur + tok, font=f) > max_w:   # 行頭の句読点・閉じ括弧は避ける
                carry = ""
                while cur and cur[-1] in "（「":   # 行末の開き括弧は次の行へ送る
                    carry = cur[-1] + carry; cur = cur[:-1]
                lines.append(cur.rstrip()); cur = carry + tok.lstrip()
            else:
                cur += tok
        lines.append(cur)
    else:
        lines = [text]
    for line in lines:
        d.text((x, y), line, font=f, fill=color); y += int(size * 1.35)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return y


def card(lines: list[tuple[str, int, bool]], sec: float, sub: str = ""):
    """タイトルカード。lines: (text, size, bold)"""
    img = np.zeros((H, W, 3), np.uint8); img[:] = (30, 30, 30)
    y = 130
    for text, size, bold in lines:
        y = draw_text(img, (80, y), text, size, (255, 255, 255), bold, max_w=W - 160) + int(size * 0.35)
    if sub:
        draw_text(img, (80, H - 120), sub, 26, (180, 180, 180), max_w=W - 160)
    return [img] * int(sec * FPS)


# --- 録画からステップ番号を読む（Recorder が (968,300) に "step N: ..." を scale 0.55 で描いている）
def _templates():
    t = {}
    for n in range(40):
        c = np.zeros((40, 140), np.uint8)
        cv2.putText(c, f"step {n}:", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
        t[n] = c
    return t


TPL = _templates()


def read_step(frame: np.ndarray) -> int | None:
    roi = frame[270:310, 960:1100]
    yellow = ((roi[:, :, 1] > 150) & (roi[:, :, 2] > 150) & (roi[:, :, 0] < 120)).astype(np.uint8) * 255
    if yellow.sum() == 0:
        return None
    best, bn = -1.0, None
    for n, t in TPL.items():
        r = cv2.matchTemplate(yellow, t, cv2.TM_CCOEFF_NORMED)
        if r.max() > best:
            best, bn = r.max(), n
    return bn if best > 0.5 else None


def load_steps(run_dir: Path):
    steps = []
    for f in sorted(glob.glob(str(run_dir / "*agent*.json"))):
        j = json.load(open(f)); r = j if "action" in j else j.get("response", j)
        img0, img1 = f[:-5] + "_0.jpg", f[:-5] + "_1.jpg"
        a = r.get("action", {}) or {}
        name = str(a.get("name", "?"))
        detail = ""
        if name == "goto_cell":
            detail = f" {a.get('cell', '')}"
        elif name in ("move", "shift"):
            detail = f" {a.get('direction', '')}/{a.get('size', '')}"
        elif name == "rotate":
            detail = f" {a.get('degrees', '')}°"
        elif name == "done":
            detail = " 成功" if a.get("success") else " 失敗"
        steps.append(dict(name=name, label=ACTION_JA.get(name, name) + detail, reasoning=str(r.get("reasoning", "")),
                          reason=str(a.get("reason", "")), img0=img0 if os.path.exists(img0) else None,
                          img1=img1 if os.path.exists(img1) else None))
    return steps


def compose(frame: np.ndarray, seg: dict, steps: list[dict], step_i: int | None, t_sec: float, speed: int):
    # 失敗例 1 の録画は黒塗り対応前（Recorder がカメラを直接読んでいた）なので、ここで全セグメントに同じ黒塗りを適用する
    over = light_grid(MASK.mask_cells(frame[90:630, 0:960]), *GRID)    # VLM に渡すのと同じ分割の格子（A1〜J7）を控えめに重ねる
    wrist = frame[90:270, 960:1280].copy()
    for t0, t1, x0, y0, x1, y1 in seg.get("wrist_mask", []):   # 手首映像の映り込み対策（区間・矩形指定の黒塗り）
        if t0 <= t_sec <= t1:
            wrist[y0:y1, x0:x1] = 0
    img = np.zeros((H, W, 3), np.uint8); img[:] = (20, 20, 20)
    img[70:610, 0:960] = over
    img[70:250, 968:1288 - 8] = wrist[:, :312] if wrist.shape[1] > 312 else wrist
    draw_text(img, (12, 8), f"{seg['label']}  「{seg['task']}」", 28, (255, 255, 255), True)
    draw_text(img, (968, 44), "手首カメラ", 18, (200, 200, 200))
    if seg.get("before"):
        draw_text(img, (12, 44), seg.get("before_short", seg["before"]), 18, (255, 200, 200), max_w=940)
    else:
        draw_text(img, (12, 44), "俯瞰カメラ + VLM 用の格子（作業領域外は映り込み対策で黒塗り）", 18, (200, 200, 200))
    # ステップ一覧
    y = 258
    draw_text(img, (968, y), "GPT-6 Astra が選んだコマンド", 18, (200, 200, 200)); y += 27
    for i, st in enumerate(steps):
        cur = (step_i is not None and i == step_i)
        done = (step_i is not None and i < step_i)
        col = (80, 255, 255) if cur else ((160, 160, 160) if done else (110, 110, 110))
        mark = "▶ " if cur else ("✓ " if done else "・")
        draw_text(img, (968, y), f"{mark}{i}: {st['label']}", 18, col, cur); y += 23
        if y > 455:
            break
    # このステップで VLM が見た俯瞰画像（格子・target 十字・reachable 枠つき。黒塗りは録画と同じ）
    if step_i is not None and step_i < len(steps) and steps[step_i].get("img0"):
        th = cv2.imread(steps[step_i]["img0"])
        if th is not None:
            th = cv2.resize(MASK.mask_cells(th), (312, 175))
            img[492:667, 968:1280] = th
            draw_text(img, (968, 468), "Astra への入力画像（俯瞰）", 16, (200, 200, 200))
    # 判断理由
    if step_i is not None and step_i < len(steps):
        st = steps[step_i]
        txt = st["reason"] if st["name"] == "done" and st["reason"] else st["reasoning"]
        draw_text(img, (12, 622), "GPT-6 Astra の出力（reasoning）: " + txt, 20, (180, 255, 180), max_w=940)
    sp = f"{speed}倍速" if speed != 1 else "等速"
    draw_text(img, (968, 680), f"{sp}   {t_sec:5.1f} s", 20, (200, 200, 200))
    return img


def light_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """映像用の控えめな格子（VLM 入力と同じ分割。線は細く、ラベルはセル左上に小さく）"""
    out = img.copy(); h, w = out.shape[:2]; cw, ch = w / cols, h / rows
    for c in range(cols + 1):
        x = min(int(c * cw), w - 1); cv2.line(out, (x, 0), (x, h), (0, 200, 200), 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = min(int(r * ch), h - 1); cv2.line(out, (0, y), (w, y), (0, 200, 200), 1, cv2.LINE_AA)
    for c in range(cols):
        for r in range(rows):
            label = f"{chr(ord('A') + c)}{r + 1}"
            cv2.putText(out, label, (int(c * cw) + 4, int(r * ch) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (int(c * cw) + 4, int(r * ch) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 230), 1, cv2.LINE_AA)
    return out


def box(d: ImageDraw.ImageDraw, xy, text: str, size: int, fill=(45, 45, 60), outline=(120, 160, 255), bold=False, max_w=None):
    """枠付きテキスト。xy=(x0,y0,x1,y1)"""
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=10, fill=fill, outline=outline, width=2)
    f = font(size, bold)
    lines = []
    cur = ""
    for ch in text:
        if ch == "\n" or d.textlength(cur + ch, font=f) > (max_w or (x1 - x0 - 20)):
            lines.append(cur); cur = "" if ch == "\n" else ch
        else:
            cur += ch
    lines.append(cur)
    th = len(lines) * int(size * 1.3)
    y = (y0 + y1) // 2 - th // 2
    for line in lines:
        tw = d.textlength(line, font=f)
        d.text(((x0 + x1) // 2 - tw / 2, y), line, font=f, fill=(255, 255, 255)); y += int(size * 1.3)


def arrow(d: ImageDraw.ImageDraw, p, q, color=(200, 200, 200)):
    d.line([p, q], fill=color, width=3)
    import math
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    for a in (ang + 2.6, ang - 2.6):
        d.line([q, (q[0] + 14 * math.cos(a), q[1] + 14 * math.sin(a))], fill=color, width=3)


def pipeline_card(sec: float):
    """推論パイプラインの図解"""
    pil = Image.new("RGB", (W, H), (30, 30, 30)); d = ImageDraw.Draw(pil)
    d.text((80, 40), "推論パイプライン（1 ステップ分）", font=font(34, True), fill=(255, 255, 255))
    # 入力 3 つ
    box(d, (60, 120, 360, 190), "指示文\n「青い消しゴムをコップに入れて」", 20)
    box(d, (60, 205, 360, 275), "俯瞰画像\n格子 A1〜J7 + 目標点の十字 + 到達範囲の枠", 20)
    box(d, (60, 290, 360, 360), "手首画像\n指が閉じる位置の十字", 20)
    # VLM
    box(d, (430, 150, 760, 330), "GPT-6 Astra\n（OpenAI Responses API）\n画像と履歴を見て\n次の 1 手を JSON で返す\n{\"action\": {\"name\": \"grasp\"}}", 21, fill=(60, 40, 80), outline=(230, 150, 255), bold=True)
    for y in (155, 240, 325):
        arrow(d, (360, y), (430, y))
    # コマンド
    box(d, (830, 120, 1220, 230), "事前定義コマンド（primitives）\ngoto_cell / move / shift / rotate\ngrasp / hover / place / drop / done", 20)
    arrow(d, (760, 200), (830, 175))
    box(d, (830, 250, 1220, 360), "コマンド内で自動化\n低トルク降下・接触検知・把持確認\n容器の口への閉ループ位置合わせ・退避", 20)
    arrow(d, (1025, 230), (1025, 250))
    # IK → 実機
    box(d, (830, 400, 1220, 480), "画素 → 関節角\n教示 9 点の補間 + 極座標 IK + 平面 FK", 20)
    arrow(d, (1025, 360), (1025, 400))
    box(d, (830, 510, 1220, 580), "SO-ARM101 実機（LeRobot）\nトルク制限・接触検知で安全に駆動", 20, fill=(40, 60, 45), outline=(150, 230, 170))
    arrow(d, (1025, 480), (1025, 510))
    # フィードバック
    box(d, (430, 420, 760, 500), "実行結果の文字列 + 新しい 2 画像\n（履歴として次のステップへ）", 20, fill=(45, 45, 45), outline=(160, 160, 160))
    arrow(d, (830, 545), (760, 470))
    arrow(d, (595, 420), (595, 330))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    y = draw_text(img, (60, 612), "物体の認識: VLM の格子読み（粗い位置）+ 手首画像の十字（最終位置合わせ）。OWLv2（ローカル GPU）は位置特定の補助。", 20, (190, 190, 190), max_w=1160)
    draw_text(img, (60, y + 4), "映り込み対策: この動画の俯瞰映像は作業領域外のセルを黒塗りしている（VLM・検出器への入力は無加工の生画像）。", 20, (190, 190, 190), max_w=1160)
    return [img] * int(sec * FPS)


def core_idea_card(sec: float):
    """コアアイデア 2 つ: (1) VLM に全部やらせず事前定義の動きを選ばせる (2) 3 次元の作業を 2 次元の格子で表現する"""
    img = np.zeros((H, W, 3), np.uint8); img[:] = (30, 30, 30)
    draw_text(img, (60, 28), "コアアイデア", 36, (255, 255, 255), True)

    def column(x0, x1, color, title, title_color, groups):
        cv2.rectangle(img, (x0, 90), (x1, 610), color, -1)
        w = x1 - x0 - 40
        y = draw_text(img, (x0 + 20, 100), title, 24, title_color, True, max_w=w)
        for lead, bullets in groups:
            if lead:
                y = draw_text(img, (x0 + 20, y + 10), lead, 21, (255, 255, 180), True, max_w=w)
            for b in bullets:   # ぶら下げインデント（折り返し行も本文の頭に揃える）
                draw_text(img, (x0 + 34, y - 1), "・", 19, (235, 235, 235))
                y = draw_text(img, (x0 + 54, y - 1), b, 19, (235, 235, 235), max_w=w - 34)

    column(50, 630, (80, 60, 100), "1. VLM に全てをやらせず、事前定義の動き（プリミティブ）を選ばせる", (255, 220, 255), [
        ("【課題】", ["VLM は「何を・どこへ」の意味理解と画像の見比べは得意だが、距離や高さの数値は苦手"]),
        ("【対処方法】", ["動作を表すプリミティブなコマンドだけを用意（goto_cell / move / rotate / grasp / hover / place / drop / done）",
                       "VLM の仕事は「2 枚の画像を見て次の一手を選ぶ」ことだけに限定",
                       "降下・開閉・接触検知・把持の確認・容器への位置合わせ・退避は、コマンドの中で自動化"]),
        ("【効果】", ["手順書を「十字が対象に重なっていれば grasp」のように画像で判断できる規則として書ける",
                   "人の手を見て VLM が自ら停止した（失敗例）"]),
    ])
    column(650, 1230, (50, 80, 70), "2. 3 次元の作業を 2 次元の格子で表現する", (200, 255, 220), [
        ("【課題】", ["カメラ画像に奥行きはなく、VLM は座標や距離を数値で正確に出せない"]),
        ("【対処方法】", ["俯瞰画像に A1〜J7 の格子を描き、VLM は「セルとセル内の位置」で場所を答える（set-of-marks）",
                       "画素と関節角は机上 9 点の教示で対応づけ、範囲外は台座からの角度と距離で計算",
                       "高さは測らず、低トルクで降りて接触で止まる。最後の位置合わせは手首画像の十字で行う"]),
        ("【効果】", ["VLM は数値を出さずに済み、「どのセルか」「十字が重なっているか」だけで動かせる",
                   "深度センサなしで、置く・容器に入れるまで実行できた"]),
    ])
    draw_text(img, (670, 572), "※ 2. は Claude（Claude Code）との議論で得たアイデア", 18, (200, 255, 220), max_w=540)
    return [img] * int(sec * FPS)


SUMMARY = {
    "工夫した点": [
        "VLM に「移動・掴む・置く・落とす」などの事前定義コマンドだけを選ばせ、上下動・開閉・接触検知はコマンド内で自動化",
        "陶器のコップ対策: 低トルクでの降下と接触検知、容器には降りずに口の上で開放、俯瞰画像での閉ループ位置合わせ",
        "置いた後は自動で退避し、アームが映っていない画像で VLM が結果を検証（失敗なら掴み直す）",
    ],
    "課題と展望": [
        "課題: 今回のゴールは「必ず失敗していたタスクを成功するまで持っていく」こと。定量的な成功率は測っていない（各タスク成功 1 回の記録）",
        "展望: 同じ指示を繰り返して成功率と失敗要因を定量化、教示点の追加と自動校正、未知の物体・指示での汎化の検証",
    ],
}


def summary_card(sec: float):
    """工夫した点と課題・展望を 1 枚に"""
    img = np.zeros((H, W, 3), np.uint8); img[:] = (30, 30, 30)
    y = 40
    for title, bullets in SUMMARY.items():
        y = draw_text(img, (60, y), title, 30, (255, 255, 255), True) + 2
        for b in bullets:
            draw_text(img, (70, y), "・", 21, (230, 230, 230))
            y = draw_text(img, (92, y), b, 21, (230, 230, 230), max_w=W - 160)
        y += 18
    return [img] * int(sec * FPS)


def recognition_card(sec: float, run: str = "20261102_021619_run", idx: str = "001_agent00"):
    """物体をどう認識しているか: 実際に VLM へ渡した 2 画像を並べて説明"""
    run_dir = ROOT / "runs" / run
    over = cv2.imread(str(run_dir / f"{idx}_0.jpg")); wrist = cv2.imread(str(run_dir / f"{idx}_1.jpg"))
    img = np.zeros((H, W, 3), np.uint8); img[:] = (30, 30, 30)
    draw_text(img, (40, 24), "物体の認識: GPT-6 Astra に渡している実際の画像（タスク 2 の 1 手目）", 30, (255, 255, 255), True)
    if over is not None:
        img[90:450, 40:680] = cv2.resize(MASK.mask_cells(over), (640, 360))
    if wrist is not None:
        img[90:450, 720:1240] = cv2.resize(wrist, (520, 360))[:, :520]
    draw_text(img, (40, 456), "俯瞰: 全セルにラベルを付けた格子（set-of-marks）。VLM は「D4 の左下」のようにセルとセル内位置で物体の場所を答える。", 19, (220, 220, 220), max_w=640)
    draw_text(img, (40, 510), "赤い十字 = 現在の目標点（グリッパの真下）、緑枠 = 到達範囲。黒塗りは録画のみ（VLM には生画像）。", 19, (220, 220, 220), max_w=640)
    draw_text(img, (720, 456), "手首: グリッパに固定したカメラ。赤い十字 = 降りたとき指が閉じる位置。", 19, (220, 220, 220), max_w=520)
    draw_text(img, (720, 510), "VLM は「十字が対象の輪郭の内側にあれば grasp」「外なら move で寄せる」と判断する。細長い物は rotate で長辺を上下方向に。", 19, (220, 220, 220), max_w=520)
    draw_text(img, (40, 600), "把持の成否も手首画像を VLM に見せて判定し、置いた後はアームを退避させてから俯瞰画像で結果を検証する。", 20, (180, 255, 180), max_w=1200)
    return [img] * int(sec * FPS)


def segment_frames(seg: dict):
    run_dir = ROOT / "runs" / seg["run"]
    mp4 = sorted(run_dir.glob("video_*.mp4"))[-1]
    steps = load_steps(run_dir)
    cap = cv2.VideoCapture(str(mp4))
    out = []; i = 0; last_step = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % seg["speed"] == 0:
            s = read_step(fr)
            if s is not None:
                last_step = s
            out.append(compose(fr, seg, steps, last_step, i / FPS, seg["speed"]))
        i += 1
    cap.release()
    return out, steps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT.parent / "video" / "submission.mp4"))
    ap.add_argument("--op", default=str(ROOT.parent / "video" / "assets" / "PAI応用1_最終課題_OP.mp4"), help="先頭に付ける講座のオープニング動画")
    ap.add_argument("--ed", default=str(ROOT.parent / "video" / "assets" / "PAI応用1_最終課題_ED.mp4"), help="末尾に付けるエンディング動画")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".raw.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = []
    frames += card([("Physical AI 応用1 最終課題", 34, False),
                    ("自然言語指示によるロボットアームのゼロショット操作", 44, True),
                    ("使用 AI: OpenAI GPT-6 Astra（VLM エージェント、Responses API）", 30, True),
                    ("実機: SO-ARM101 + 俯瞰カメラ + 手首カメラ。VLM が画像を見て 1 手ずつ事前定義コマンドを選ぶ computer use 方式", 24, False),
                    ("映り込み対策: 俯瞰映像の作業領域外を黒塗りしています（VLM への入力は無加工）", 24, False)], 8,
                   sub="タスク 1・2 は等速、失敗例は 2 倍速で再生しています")
    frames += pipeline_card(10)
    frames += core_idea_card(18)
    frames += recognition_card(9)
    for seg in SEGMENTS:
        print("[segment]", seg["run"])
        lines = [(seg["label"], 30, False), (f"「{seg['task']}」", 44, True)]
        if seg.get("before"):
            lines.append((seg["before"], 26, True))
        lines.append((seg["note"], 26, False))
        frames += card(lines, 6.0 if seg.get("before") else 3.5, sub=("2 倍速" if seg["speed"] != 1 else "等速"))
        fs, steps = segment_frames(seg)
        frames += fs
        frames += [fs[-1]] * (FPS * 1)
    frames += summary_card(16)
    for f in frames:
        wr.write(f)
    wr.release()
    # H.264 に変換（再生互換性）。講座の OP / ED 動画（1920x1080, 29.97fps）があれば 1280x720・30fps に揃えて前後に結合する。
    # ffmpeg は imageio-ffmpeg 同梱のものを使う
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        op, ed = (Path(args.op) if args.op and Path(args.op).exists() else None), (Path(args.ed) if args.ed and Path(args.ed).exists() else None)
        inputs, parts, k = [], [], 0
        for clip in (op, tmp, ed):
            if clip is None:
                continue
            inputs += ["-i", str(clip)]
            parts.append(f"[{k}:v]scale={W}:{H}:flags=lanczos,fps=30,setsar=1,format=yuv420p[v{k}]"); k += 1
        fc = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[v]"
        subprocess.run([exe, "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
        os.remove(tmp)
        print(f"[concat] OP={op.name if op else '-'} / 本編 / ED={ed.name if ed else '-'}")
    except Exception as e:  # noqa: BLE001
        print("ffmpeg 変換失敗（mp4v のまま）:", e); os.replace(tmp, out)
    print(f"[done] {out}  本編 {len(frames)} frames = {len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()
