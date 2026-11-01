"""SO-101 の逆運動学（画素座標 → 関節角）。

* GridIK  : 教示点 (u, v) → 関節角 を補間する方式（scripts/teach_pixels.py で教示）。URDF 精度に依存せず実機で確実。
            calib/fk_fit.json（平面 FK）があれば、机面姿勢から hover / carry の高さを絶対値で解く。
* PolarIK : 教示点の凸包の外側用。台座中心の極座標モデル + 平面 FK の逆解。GridIK が自動で使う。

solve(x, y, level) -> {関節名: deg}（gripper は含まない）。level は "down" | "hover" | "carry"（| 動的に追加した高さ）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from .so101 import JOINTS


class ReachError(Exception):
    pass


# ----------------------------------------------------------------------------
class GridIK:
    """教示点から補間。各点に 'down'（接触高さ）と 'up'（ホバー）の 2 姿勢を持つ。"""

    def __init__(self, grid_file: str, use_taught_up: bool = True, hover_delta: dict | None = None,
                 fk=None, levels: dict | None = None):
        self.use_taught_up = use_taught_up
        self.hover_delta = hover_delta or {"shoulder_lift": -20}
        self.fk = fk                                   # robot.fk.PlanarFK（あれば高さを絶対値で解く）
        self.levels = levels or {"hover": 0.06, "carry": 0.14}
        self.keep_pitch = {"hover": 1.0, "carry": 0.15}
        p = Path(grid_file)
        if not p.exists():
            raise FileNotFoundError(f"{grid_file} がありません。scripts/teach_grid.py で教示してください")
        data = json.load(open(p))
        self.pts = np.array([[d["x"], d["y"]] for d in data["points"]])  # (N,2)
        self.poses = {
            level: np.array([[d[level][j] for j in JOINTS] for d in data["points"]])  # (N,5)
            for level in ("down", "up")
        }
        self._interp = {}
        # 外挿用: 各関節を (u, v) の二次多項式で最小二乗フィット（点数が 6 未満なら一次）
        self._poly = {}
        deg = 2 if len(self.pts) >= 7 else 1
        A = self._design(self.pts, deg)
        for level, P in self.poses.items():
            coef, *_ = np.linalg.lstsq(A, P, rcond=None)
            self._poly[level] = (deg, coef)
        self.polar = None
        if fk is not None:
            try:
                self.polar = PolarIK(self.pts, self.poses["down"], fk)
                print(f"[ik] 教示範囲外は極座標モデルで解く（フィット誤差 max {self.polar.fit_err.max():.0f}px）")
            except Exception as e:  # noqa: BLE001
                print(f"[ik] PolarIK 構築失敗（二次外挿で続行）: {e}")
        try:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

            for level, P in self.poses.items():
                lin = LinearNDInterpolator(self.pts, P) if len(self.pts) >= 3 else None
                near = NearestNDInterpolator(self.pts, P)
                self._interp[level] = (lin, near)
        except ImportError:
            pass

    @staticmethod
    def _design(pts: np.ndarray, deg: int) -> np.ndarray:
        u, v = pts[:, 0] / 1000.0, pts[:, 1] / 1000.0  # スケーリングして条件数を抑える
        cols = [np.ones_like(u), u, v]
        if deg >= 2:
            cols += [u * u, u * v, v * v]
        return np.stack(cols, 1)

    def _extrapolate(self, x: float, y: float, level: str) -> np.ndarray:
        deg, coef = self._poly[level]
        return (self._design(np.array([[x, y]], float), deg) @ coef)[0]

    def solve(self, x: float, y: float, z: float | str = "down") -> Dict[str, float]:
        """level: "down"（机面）| "up"/"hover"（物の少し上）| "high"/"carry"（運搬: 背の高い物を跨ぐ）。
        FK がある場合は、机面姿勢から同じ机上点の真上・指定高さ [m] の姿勢を解く。"""
        if isinstance(z, str):
            alias = {"up": "hover", "high": "carry", "table": "down"}
            level = alias.get(z, z)
        else:
            level = "down" if z < 0.04 else "hover"
        if self.fk is not None and level != "down" and level in self.levels:   # hover / carry / 動的に追加した高さ（rim など）
            down = self.solve(x, y, "down")
            return self.fk.raise_pose(down, self.levels[level], keep_pitch=self.keep_pitch.get(level, 1.0))
        if level == "carry":
            level = "up"   # FK なし: 旧方式（down + delta は呼び出し側で加算）
        if level == "hover":
            level = "up"
        if level == "up" and not self.use_taught_up:
            q = self.solve(x, y, "down")
            for j, d in self.hover_delta.items():
                q[j] += d
            return q
        if self._interp:
            lin, near = self._interp[level]
            q = lin([[x, y]])[0] if lin is not None else None
            if q is None or np.any(np.isnan(q)):
                if self.polar is not None and level == "down":
                    print(f"[ik] ({x:.0f},{y:.0f}) は教示範囲外 → 極座標モデル")
                    return self.polar.solve_down(x, y)     # ReachError はそのまま上げる
                q = self._extrapolate(x, y, level)  # 凸包の外は二次フィットで外挿
                print(f"[ik] ({x:.0f},{y:.0f}) は教示範囲外 → 外挿")
        else:  # scipy なし: 逆距離加重
            d = np.linalg.norm(self.pts - [x, y], axis=1) + 1e-6
            w = 1 / d**2
            q = (self.poses[level] * w[:, None]).sum(0) / w.sum()
        return {j: float(v) for j, v in zip(JOINTS, q)}


# ----------------------------------------------------------------------------
class PolarIK:
    """教示点の外側用: 画素 → (pan, r) の極座標モデル + 平面 FK の逆解で机面姿勢を解く。

    モデル: 画素 = base + R(θ)·[ (r+rb)(cos a, sin a) + d·(−sin a, cos a) ] / s,  a = kp·pan
      base: 台座（pan 軸）の画素位置, s: m/px, θ: 画像回転, rb: r のオフセット, d: 指先の横オフセット, kp: pan 符号/倍率
    r は PlanarFK で教示姿勢から求めた肩軸からの水平距離。俯角 t3 は r の一次式（教示姿勢からフィット）。
    教示点 9 点の leave-one-out 誤差は約 5〜28 px（二次多項式外挿は 13〜127 px）。"""

    def __init__(self, pts: np.ndarray, poses: np.ndarray, fk):
        from scipy.optimize import least_squares

        self.fk = fk
        self.pan = poses[:, 0]
        self.r = np.array([fk.tip(q[1], q[2], q[3])[1] for q in poses])
        self.roll = float(poses[:, 4].mean())
        s_, o_ = fk.s, fk.o
        t1 = s_[0] * (poses[:, 1] + o_[0]); t2 = t1 + s_[1] * (poses[:, 2] + o_[1]); t3 = t2 + s_[2] * (poses[:, 3] + o_[2])
        self.elbow_sign = float(np.sign(np.median(t2 - t1)))          # 肘の曲がる向き（教示姿勢に合わせる）
        self.pitch_coef = np.polyfit(self.r, t3, 1)                   # t3 ≈ a·r + b [deg]
        self.pitch_range = (float(t3.min()) - 10, float(t3.max()) + 10)

        def res(x):
            return (self._fwd(x, self.pan, self.r) - pts).ravel()

        best = None
        for th0 in (0, 90, 180, -90):
            for kp0 in (1, -1):
                f = least_squares(res, [pts[:, 0].mean(), pts[:, 1].mean(), 0.0006, th0, 0.0, 0.0, kp0])
                if best is None or f.cost < best.cost:
                    best = f
        self.x = best.x
        self.fit_err = np.linalg.norm(self._fwd(self.x, self.pan, self.r) - pts, axis=1)

    @staticmethod
    def _fwd(x, pan, r):
        bu, bv, s, th, rb, dlat, kp = x
        a = np.deg2rad(kp * pan); R = r + rb
        X = R * np.cos(a) - dlat * np.sin(a); Y = R * np.sin(a) + dlat * np.cos(a)
        c, sn = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        return np.stack([bu + (c * X - sn * Y) / s, bv + (sn * X + c * Y) / s], 1)

    def pixel_to_polar(self, u: float, v: float) -> tuple[float, float]:
        """画素 → (pan [deg], r [m])。横オフセット d があるので数値的に解く。"""
        bu, bv, s, th, rb, dlat, kp = self.x
        c, sn = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        X, Y = (u - bu) * s, (v - bv) * s
        X, Y = c * X + sn * Y, -sn * X + c * Y          # R(−θ)
        rho = math.hypot(X, Y)
        if rho <= abs(dlat):
            raise ReachError(f"({u:.0f},{v:.0f}) は台座に近すぎます")
        R = math.sqrt(rho * rho - dlat * dlat)
        a = math.atan2(Y, X) - math.atan2(dlat, R)
        return math.degrees(a) / kp, R - rb

    def solve_down(self, u: float, v: float) -> Dict[str, float]:
        pan, r = self.pixel_to_polar(u, v)
        t3 = float(np.clip(np.polyval(self.pitch_coef, r), *self.pitch_range))
        fk = self.fk; L1, L2, L3, BZ = fk.L1, fk.L2, fk.L3, fk.BZ
        t3r = math.radians(t3)
        wr, wz = r - L3 * math.cos(t3r), -BZ - L3 * math.sin(t3r)     # 手首軸の位置（先端 z=0）
        D = math.hypot(wr, wz)
        if D > L1 + L2 - 1e-3 or D < abs(L1 - L2) + 1e-3:
            raise ReachError(f"({u:.0f},{v:.0f}) は到達不可 (r={r*1000:.0f}mm, D={D*1000:.0f}mm)")
        cos_e = (L1 * L1 + L2 * L2 - D * D) / (2 * L1 * L2)
        inner = math.acos(max(-1.0, min(1.0, cos_e)))
        bend = math.pi - inner                                        # |t2 − t1|
        alpha = math.acos(max(-1.0, min(1.0, (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D))))
        base = math.atan2(wz, wr)
        t1 = base - self.elbow_sign * alpha
        t2 = t1 + self.elbow_sign * bend
        s_, o_ = fk.s, fk.o
        lift = math.degrees(t1) / s_[0] - o_[0]
        elbow = math.degrees(t2 - t1) / s_[1] - o_[1]
        wrist = (t3 - math.degrees(t2)) / s_[2] - o_[2]
        for k, val in (("shoulder_lift", lift), ("elbow_flex", elbow), ("wrist_flex", wrist)):
            lo, hi = fk.LIMITS[k]
            if not (lo <= val <= hi):
                raise ReachError(f"({u:.0f},{v:.0f}) は関節範囲外 ({k}={val:.0f}°)")
        return {"shoulder_pan": float(pan), "shoulder_lift": float(lift), "elbow_flex": float(elbow),
                "wrist_flex": float(wrist), "wrist_roll": self.roll}


def make_ik(cfg: dict):
    fk = None
    try:
        from .fk import load_fk

        fk = load_fk(Path(cfg["grid_file"]).resolve().parents[1])
    except Exception as e:  # noqa: BLE001
        print(f"[ik] FK 読み込み失敗（旧方式で続行）: {e}")
    return GridIK(cfg["grid_file"], cfg.get("use_taught_up", True), cfg.get("hover_delta"), fk=fk, levels=cfg.get("levels"))
