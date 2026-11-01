"""教示データからフィットした平面順運動学（肩 lift・肘・手首の 3 関節）。

calib/fk_fit.json（scripts と同じ手順で推定: 9 点の机面姿勢で先端 z=0 になるよう関節オフセットを最小二乗）。
用途: 机面姿勢から「同じ机上点の真上・同じ俯角」で高さ z [m] の姿勢を解く（hover / 運搬高さを絶対値で指定）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


class PlanarFK:
    def __init__(self, fit_file: str):
        d = json.load(open(fit_file))
        self.s = np.array(d["signs"], float)
        self.o = np.array(d["offsets"], float)
        self.L1, self.L2, self.L3, self.BZ = d["links"]

    def tip(self, lift: float, elbow: float, wrist: float) -> tuple[float, float]:
        """(z, r) [m]: 先端の高さと、肩軸からの水平距離"""
        s, o = self.s, self.o
        t1 = np.deg2rad(s[0] * (lift + o[0]))
        t2 = t1 + np.deg2rad(s[1] * (elbow + o[1]))
        t3 = t2 + np.deg2rad(s[2] * (wrist + o[2]))
        z = self.BZ + self.L1 * np.sin(t1) + self.L2 * np.sin(t2) + self.L3 * np.sin(t3)
        r = self.L1 * np.cos(t1) + self.L2 * np.cos(t2) + self.L3 * np.cos(t3)
        return float(z), float(r)

    # 関節の可動範囲 [deg]（教示値の周辺に余裕を持たせた保守的な範囲）
    LIMITS = {"shoulder_lift": (-110, 60), "elbow_flex": (-85, 110), "wrist_flex": (10, 100)}   # wrist は実機で約 102° で飽和

    def raise_pose(self, down: Dict[str, float], z_target: float, keep_pitch: float = 1.0) -> Dict[str, float]:
        """down（机面姿勢）と同じ水平位置で先端高さ z_target の姿勢を返す。
        keep_pitch: 手先俯角を一定に保つ重み（hover では強く=手首カメラの見え方を揃える、運搬では弱く=到達性優先）。"""
        from scipy.optimize import least_squares

        s = self.s
        l0, e0, w0 = down["shoulder_lift"], down["elbow_flex"], down["wrist_flex"]
        _, r0 = self.tip(l0, e0, w0)
        lo = np.array([self.LIMITS[k][0] - v for k, v in (("shoulder_lift", l0), ("elbow_flex", e0), ("wrist_flex", w0))])
        hi = np.array([self.LIMITS[k][1] - v for k, v in (("shoulder_lift", l0), ("elbow_flex", e0), ("wrist_flex", w0))])

        def resid(x):
            dl, de, dw = x
            z, r = self.tip(l0 + dl, e0 + de, w0 + dw)
            pitch_dev = (s[0] * dl + s[1] * de + s[2] * dw)   # 俯角の変化 [deg]
            return [(z - z_target) * 20, (r - r0) * 10, keep_pitch * pitch_dev / 30, 0.002 * dl, 0.002 * de, 0.002 * dw]

        best = None
        for x0 in ([-20, 0, 20], [-30, -20, 50], [-10, -40, 50], [-40, -10, 30], [10, -60, 60]):
            x0 = np.clip(x0, lo + 1, hi - 1)
            res = least_squares(resid, x0, bounds=(lo, hi))
            if best is None or res.cost < best.cost:
                best = res
        dl, de, dw = best.x
        q = dict(down); q["shoulder_lift"], q["elbow_flex"], q["wrist_flex"] = float(l0 + dl), float(e0 + de), float(w0 + dw)
        return q


def load_fk(root: Path) -> PlanarFK | None:
    f = root / "calib" / "fk_fit.json"
    return PlanarFK(str(f)) if f.exists() else None
