"""カメラ位置調整用ライブプレビュー。VLM に渡すのと同じ格子を重ねて表示する。

  python scripts/preview_cam.py          # q で終了、s で calib/preview.jpg に保存
  python scripts/preview_cam.py --device /dev/video2
  python scripts/preview_cam.py --wrist   # 手首カメラ。ピント合わせ用に鮮明度(sharpness)を表示、大きいほど合焦
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, show  # noqa: E402
from perception.camera import Camera  # noqa: E402
from perception.vlm import draw_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--wrist", action="store_true", help="config の wrist_device を使う")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.wrist:
        cfg["camera"]["device"] = cfg["camera"]["wrist_device"]
    if args.device:
        cfg["camera"]["device"] = args.device
    cam = Camera(cfg["camera"]); cam.open()
    cols, rows = cfg["vlm"]["grid_cols"], cfg["vlm"]["grid_rows"]
    work = None   # 教示範囲（緑枠 = アームが正確に届く範囲）。俯瞰のときだけ描く
    if not args.wrist and Path(cfg["ik"]["grid_file"]).exists():
        import json
        pts = [(q["x"], q["y"]) for q in json.load(open(cfg["ik"]["grid_file"]))["points"]]
        fm = cfg["detector"].get("fence_margin", 40); fb = cfg["detector"].get("fence_margin_bottom", fm)
        work = (int(min(x for x, _ in pts)) - fm, int(min(y for _, y in pts)) - fm, int(max(x for x, _ in pts)) + fm, int(max(y for _, y in pts)) + fb)
    print("q: 終了 / s: 保存")
    try:
        while True:
            img = cam.mask_cells(cam.grab())   # 録画と同じ黒塗りを表示
            disp = img.copy() if args.no_grid else draw_grid(img, cols, rows)
            h, w = img.shape[:2]
            if work:
                cv2.rectangle(disp, (work[0], work[1]), (work[2], work[3]), (0, 255, 0), 2)
                cv2.putText(disp, "reachable", (work[0] + 6, work[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 鮮明度: 中央 1/2 領域のラプラシアン分散（ピントが合うほど大きい）
            roi = cv2.cvtColor(img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4], cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(roi, cv2.CV_64F).var()
            cv2.rectangle(disp, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 1)
            cv2.putText(disp, f"sharpness {sharp:7.1f}", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.line(disp, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)  # 中心線
            cv2.line(disp, (0, h // 2), (w, h // 2), (0, 0, 255), 1)
            show("preview (q: quit, s: save)", disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                out = Path(__file__).resolve().parents[1] / "calib" / "preview.jpg"
                cv2.imwrite(str(out), img); print("saved", out)
    finally:
        cam.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
