"""画素座標 ⇄ 作業座標の変換（ホモグラフィ）。

本システムでは作業座標 = 俯瞰画像の画素（scripts/teach_pixels.py が単位行列の homography.npy を作る）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Workspace:
    def __init__(self, homography_file: str):
        p = Path(homography_file)
        if not p.exists():
            raise FileNotFoundError(f"{homography_file} がありません。scripts/teach_pixels.py を実行してください")
        self.H = np.load(p)  # (3,3) 画素→作業面
        self.Hinv = np.linalg.inv(self.H)

    def pixel_to_xy(self, u: float, v: float) -> tuple[float, float]:
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def xy_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        p = self.Hinv @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    @staticmethod
    def fit(pixels: np.ndarray, xys: np.ndarray) -> np.ndarray:
        import cv2

        H, _ = cv2.findHomography(pixels.astype(np.float32), xys.astype(np.float32), 0)
        return H
