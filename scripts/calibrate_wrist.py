"""手首カメラの自動校正（人手の計測なし）。hover 姿勢を変えたら必ず実行する。

1) 俯瞰で物体（既定: blue eraser）を検出し、その真上の hover 姿勢へ
2) 検出枠内の代表色（HSV）を手首画像で追跡し、物体の重心 = ツールポイント（fingertip 十字）を得る
3) 俯瞰画素で ±step ずらして追跡し、手首画像の移動 = J @ 俯瞰画素の移動 を最小二乗で求める
結果: calib/wrist_jacobian.json を更新し、config.yaml の grasp.tool_point を書き換える。

  python scripts/calibrate_wrist.py              # blue eraser
  python scripts/calibrate_wrist.py "red cap"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_config, make_run_dir  # noqa: E402
from main import workspace_roi  # noqa: E402
from perception.camera import Camera  # noqa: E402
from perception.detector import Detector  # noqa: E402
from perception.workspace import Workspace  # noqa: E402
from robot.kinematics import make_ik  # noqa: E402
from robot.so101 import make_robot  # noqa: E402


def dominant_hsv(img, box):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; r = max(4, min(x1 - x0, y1 - y0) // 4)
    hsv = cv2.cvtColor(img[cy - r:cy + r, cx - r:cx + r], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.median(hsv, axis=0)


def track(img, hsv_ref, tol=(12, 70, 80)):
    """色相を主に使って追跡（彩度・明度は照明で変わるので段階的に緩める）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for h_tol, s_min, v_min in ((tol[0], max(hsv_ref[1] - tol[1], 40), max(hsv_ref[2] - tol[2], 40)), (15, 60, 40), (20, 40, 30)):
        lo = np.array([max(hsv_ref[0] - h_tol, 0), s_min, v_min]); hi = np.array([min(hsv_ref[0] + h_tol, 179), 255, 255])
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 600]
        if cnts:
            c = max(cnts, key=cv2.contourArea); M = cv2.moments(c)
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object", nargs="?", default="blue eraser")
    ap.add_argument("--step", type=float, default=50)
    args = ap.parse_args()
    cfg = load_config(); run = make_run_dir(cfg, "wristcal")
    det = Detector(cfg["detector"]); roi = workspace_roi(cfg)
    ik = make_ik(cfg["ik"]); ws = Workspace(cfg["workspace"]["homography_file"])
    wcfg = dict(cfg["camera"]); wcfg["device"] = cfg["camera"]["wrist_device"]
    robot = make_robot(cfg["robot"]); robot.connect()
    cam = Camera(cfg["camera"]); cam.open(); wcam = Camera(wcfg); wcam.open()
    samples = []
    try:
        robot.home(sec=2.5); robot.gripper(open_=False, sec=0.5)
        over = cam.grab()
        c = det.detect(over, args.object, roi=roi, threshold=0.12, topk=1)
        if not c:
            sys.exit(f"俯瞰で '{args.object}' が見つからない")
        u0, v0 = c[0]["center"]; hsv_ref = dominant_hsv(over, c[0]["box"])
        print(f"俯瞰 {args.object}: ({u0:.0f},{v0:.0f}) 代表色 HSV={hsv_ref.astype(int).tolist()}")
        for du, dv in [(0, 0), (args.step, 0), (0, args.step), (-args.step, 0), (0, -args.step), (0, 0)]:
            q = ik.solve(*ws.pixel_to_xy(u0 + du, v0 + dv), "hover")
            robot.move_to(q, sec=1.8); time.sleep(0.5)
            w = track(wcam.grab(), hsv_ref)
            print(f"  俯瞰 Δ=({du:+.0f},{dv:+.0f}) → 手首 {w and tuple(int(x) for x in w)}")
            if w:
                samples.append((du, dv, w[0], w[1]))
        pan = ik.solve(*ws.pixel_to_xy(u0, v0), "hover")["shoulder_pan"]
        robot.home(sec=2.5)
    finally:
        cam.close(); wcam.close(); robot.disconnect()
    base_s = [s for s in samples if s[0] == 0 and s[1] == 0]
    if len(samples) < 4 or not base_s:
        sys.exit("サンプル不足")
    base = np.mean([[s[2], s[3]] for s in base_s], 0)
    A = np.array([[s[0], s[1]] for s in samples if not (s[0] == 0 and s[1] == 0)], float)
    B = np.array([[s[2] - base[0], s[3] - base[1]] for s in samples if not (s[0] == 0 and s[1] == 0)], float)
    J, *_ = np.linalg.lstsq(A, B, rcond=None); J = J.T
    resid = float(np.abs(A @ J.T - B).max())
    out = Path(cfg["grasp"]["servo_jacobian"])
    json.dump({"J": J.tolist(), "level": "hover", "pan_at_calib": float(pan), "tool_point_measured": base.tolist(),
               "residual_px": resid, "samples": samples}, open(out, "w"), indent=2)
    cfgp = ROOT / "config.yaml"; txt = cfgp.read_text()
    txt = re.sub(r"  tool_point: \[\d+, \d+\]", f"  tool_point: [{int(base[0])}, {int(base[1])}]", txt)
    cfgp.write_text(txt)
    print(f"J = {np.round(J, 2).tolist()} 残差 {resid:.0f}px  ツールポイント ({int(base[0])},{int(base[1])}) pan {pan:.1f}\n→ {out} と config.yaml を更新")


if __name__ == "__main__":
    main()
