"""方眼紙なしの教示: 俯瞰カメラの画素座標 (u, v) を作業座標として GridIK を作る。

各教示点で
  1) トルク OFF のアームを手で動かし、グリッパ先端（閉じた状態）をテーブルに軽く当てて Enter
     → 俯瞰画像が出るので先端をクリック（down 姿勢 + 画素座標を記録）
  2) そのまま 5〜8cm 真上に持ち上げて Enter（up 姿勢を記録）
を繰り返す。作業範囲を囲うように 6〜12 点ほど（四隅 + 中間）取るとよい。

出力: calib/grid_points.json（x=u, y=v の画素）と calib/homography.npy（単位行列）
      → 既存パイプライン (locate → Workspace → GridIK) がそのまま動く。

使い方: python scripts/teach_pixels.py            # 新規（既存があれば追記）
        python scripts/teach_pixels.py --reset    # 作り直し
        python scripts/teach_pixels.py --auto-up 12   # up 姿勢を手で作らず shoulder_lift を -12° して自動記録
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, show  # noqa: E402
from perception.camera import Camera  # noqa: E402
from robot.so101 import JOINTS, make_robot  # noqa: E402


def click_point(img: np.ndarray, title: str, prev: list[tuple[float, float]]) -> tuple[float, float] | None:
    disp = img.copy()
    for i, (u, v) in enumerate(prev):
        cv2.circle(disp, (int(u), int(v)), 5, (0, 200, 0), -1)
        cv2.putText(disp, str(i + 1), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    cv2.putText(disp, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    pt: list = []

    def cb(ev, u, v, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt[:] = [(u, v)]
            d = disp.copy(); cv2.drawMarker(d, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2); show("teach", d)

    show("teach", disp); cv2.setMouseCallback("teach", cb)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and pt:  # Enter で確定
            cv2.destroyWindow("teach"); show._seen.discard("teach"); return pt[0]
        if k == ord("q"):
            cv2.destroyWindow("teach"); show._seen.discard("teach"); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--auto-up", type=float, default=None, help="shoulder_lift をこの角度 [deg] 減らして up を自動生成")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="指定番号(1始まり)の点を削除して終了")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.mock:
        cfg["robot"]["backend"] = "mock"; cfg["camera"]["backend"] = "mock"
    out = Path(cfg["ik"]["grid_file"])
    data = {"points": [], "units": "px"}
    if out.exists() and not args.reset:
        data = json.load(open(out)); print(f"既存 {len(data['points'])} 点に追記")
    if args.drop:
        for n in sorted(args.drop, reverse=True):
            if 1 <= n <= len(data["points"]):
                r = data["points"].pop(n - 1); print(f"点 {n} を削除: px=({int(r['x'])},{int(r['y'])})")
        json.dump(data, open(out, "w"), indent=2); print(f"残り {len(data['points'])} 点"); return
    hfile = Path(cfg["workspace"]["homography_file"]); hfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(hfile, np.eye(3))  # 画素 = 作業座標

    robot = make_robot(cfg["robot"]); robot.connect()
    cam = Camera(cfg["camera"]); cam.open()
    robot.set_torque(False)
    print("トルク OFF。手でアームを動かしてください。")
    print("※ 画像ウィンドウは、このターミナルで Enter を押した後に開きます（ターミナルの裏に出ることがあります）")
    try:
        while True:
            n = len(data["points"])
            s = input(f"\n[点 {n + 1}] 先端をテーブルに当てたら、ここで Enter → 画像ウィンドウが開きます（q で終了）> ")
            if s.strip() == "q":
                break
            down = {k: robot.joints()[k] for k in JOINTS}
            img = cam.grab()
            uv = click_point(img, f"point {n + 1}: click gripper tip, Enter=OK, q=skip", [(p["x"], p["y"]) for p in data["points"]])
            if uv is None:
                print("  スキップ"); continue
            if args.auto_up is not None:
                up = dict(down); up["shoulder_lift"] -= args.auto_up
                robot.set_torque(True); robot.move_to(up, sec=1.0); robot.set_torque(False)
            else:
                while True:
                    input("  そのまま真上に持ち上げて Enter > ")
                    up = {k: robot.joints()[k] for k in JOINTS}
                    moved = max(abs(up[k] - down[k]) for k in JOINTS)
                    if moved >= 5.0:
                        break
                    print(f"  ⚠ down からほとんど動いていません（最大 {moved:.1f}°）。持ち上げてからもう一度 Enter")
            # 既存点との整合チェック: 関節が近いのに画素が遠い（または逆）なら警告
            for i, q in enumerate(data["points"], 1):
                dj = max(abs(q["down"][k] - down[k]) for k in ("shoulder_pan", "shoulder_lift", "elbow_flex"))
                dp = ((q["x"] - uv[0]) ** 2 + (q["y"] - uv[1]) ** 2) ** 0.5
                if dj < 8 and dp > 150:
                    print(f"  ⚠ 点 {i} と関節角がほぼ同じ({dj:.1f}°)なのに画素が {dp:.0f}px 離れています。クリック位置を確認してください")
            data["points"].append({"x": float(uv[0]), "y": float(uv[1]), "down": down, "up": up})
            json.dump(data, open(out, "w"), indent=2)
            print(f"  記録: px={tuple(int(c) for c in uv)}  down={ {k: round(v, 1) for k, v in down.items()} }")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{len(data['points'])} 点を {out} に保存（単位: px）。homography は単位行列。")
        cv2.destroyAllWindows(); cam.close()
        robot.set_torque(True); robot.disconnect()


if __name__ == "__main__":
    main()
