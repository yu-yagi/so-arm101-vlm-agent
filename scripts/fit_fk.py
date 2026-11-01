"""教示データ（calib/grid_points.json の down 姿勢）から平面順運動学の関節オフセットを当てはめ、calib/fk_fit.json を作る。

モデル: 肩 lift・肘・手首の 3 関節の平面リンク（リンク長は SO-101 の公称値 0.116 / 0.135 / 0.11 m、肩軸高さ 0.06 m）。
机面で教示した 9 姿勢すべてで先端高さ z = 0 になるよう、3 つの関節オフセット [deg] を最小二乗で推定する。
関節の符号は (-1, -1, -1)（LeRobot の度数が増えると各リンクが下向きに回る）。

  python scripts/fit_fk.py            # → calib/fk_fit.json（rms 残差を表示）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

LINKS = [0.116, 0.135, 0.11, 0.06]   # L1, L2, L3, 肩軸の高さ [m]
SIGNS = [-1, -1, -1]


def tip_z(offsets, lift, elbow, wrist):
    s, o = SIGNS, offsets
    t1 = np.deg2rad(s[0] * (lift + o[0])); t2 = t1 + np.deg2rad(s[1] * (elbow + o[1])); t3 = t2 + np.deg2rad(s[2] * (wrist + o[2]))
    return LINKS[3] + LINKS[0] * np.sin(t1) + LINKS[1] * np.sin(t2) + LINKS[2] * np.sin(t3)


def main():
    pts = json.load(open(ROOT / "calib" / "grid_points.json"))["points"]
    q = np.array([[p["down"]["shoulder_lift"], p["down"]["elbow_flex"], p["down"]["wrist_flex"]] for p in pts])
    res = least_squares(lambda o: tip_z(o, q[:, 0], q[:, 1], q[:, 2]), x0=[-80.0, 70.0, 10.0])
    z = tip_z(res.x, q[:, 0], q[:, 1], q[:, 2])
    out = {"signs": SIGNS, "offsets": [float(v) for v in res.x], "links": LINKS}
    json.dump(out, open(ROOT / "calib" / "fk_fit.json", "w"), indent=2)
    print(f"offsets = {np.round(res.x, 2)}  rms z = {np.sqrt((z ** 2).mean()) * 1000:.2f} mm → calib/fk_fit.json")


if __name__ == "__main__":
    main()
