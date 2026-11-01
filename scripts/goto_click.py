"""教示精度テスト: 俯瞰画像をクリックした場所にアームを降ろす（VLM 不要）。

  python scripts/goto_click.py
  画像をクリック → Enter で carry(18cm) → hover(6cm) → down → hover → home の順に動く。q で終了。
  --hover-only: down（机面）まで降りず hover で止める（教示範囲外の精度確認用。凸包の外は極座標 IK モデル）。
  教示点の凸包を緑で表示。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, show  # noqa: E402
from perception.camera import Camera  # noqa: E402
from perception.workspace import Workspace  # noqa: E402
from robot.kinematics import make_ik  # noqa: E402
from robot.so101 import make_robot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hover-only", action="store_true"); args = ap.parse_args()
    cfg = load_config()
    robot = make_robot(cfg["robot"]); robot.connect()
    cam = Camera(cfg["camera"]); cam.open()
    ik = make_ik(cfg["ik"]); ws = Workspace(cfg["workspace"]["homography_file"])
    hull = cv2.convexHull(ik.pts.astype(np.float32)).astype(int) if hasattr(ik, "pts") else None
    robot.home(sec=2.5)
    print("画像をクリック → Enter で移動、q で終了")
    try:
        while True:
            img = cam.grab(); disp = img.copy()
            if hull is not None:
                cv2.polylines(disp, [hull], True, (0, 255, 0), 2)
            pt: list = []

            def cb(ev, u, v, *_):
                if ev == cv2.EVENT_LBUTTONDOWN:
                    pt[:] = [(u, v)]; d = disp.copy()
                    cv2.drawMarker(d, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2); show("goto", d)

            show("goto", disp); cv2.setMouseCallback("goto", cb)
            while True:
                k = cv2.waitKey(30) & 0xFF
                if k == ord("q"):
                    return
                if k == 13 and pt:
                    break
            u, v = pt[0]; x, y = ws.pixel_to_xy(u, v)
            print(f"\nclick px=({u},{v}) -> xy=({x:.1f},{y:.1f})")
            robot.gripper(open_=False, sec=0.6)
            try:
                q_carry, q_hover = ik.solve(x, y, "carry"), ik.solve(x, y, "hover")
                q_down = ik.solve(x, y, "down")
            except Exception as e:  # noqa: BLE001
                print(f"  到達不可: {e}"); continue
            print("  down joints:", {k: round(v, 1) for k, v in q_down.items()})
            robot.move_to(q_carry, sec=2.5)
            robot.move_to(q_hover, sec=1.5)
            if not args.hover_only:
                robot.move_to(q_down, sec=1.5)
            after = cam.grab()
            cv2.drawMarker(after, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(after, "red = clicked target. any key to continue", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            show("goto", after); cv2.waitKey(0)
            robot.move_to(q_hover, sec=1.5)
            robot.move_to(q_carry, sec=1.5)
            robot.home(sec=2.5)
    finally:
        cv2.destroyAllWindows(); cam.close()
        robot.home(sec=2.0); robot.disconnect()


if __name__ == "__main__":
    main()
