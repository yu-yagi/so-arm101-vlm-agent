"""カメラ取得。opencv（USB カメラ）と mock（画像ファイル）を同じ API で。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class Camera:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cap = None
        # 俯瞰カメラのみ: 指定セルを黒塗りで隠す（録画・プレビュー用、映り込み対策）。手首カメラ（device == wrist_device）には適用しない
        is_wrist = cfg.get("wrist_device") and cfg.get("device") == cfg.get("wrist_device")
        self.blur_cells = [] if is_wrist else [c.strip().upper() for c in (cfg.get("blur_cells") or [])]
        self.blur_grid = tuple(cfg.get("blur_grid") or (10, 7))

    def open(self):
        if self.cfg.get("backend", "opencv") == "mock":
            return
        import cv2

        dev = self.cfg.get("device") or self.cfg.get("index", 0)
        self.cap = cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {dev} を開けません")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # EMEET C960 は MJPG で 720p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.get("width", 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.get("height", 720))
        for _ in range(5):  # 露出安定のため捨てフレーム
            self.cap.read()
            time.sleep(0.05)

    def mask_cells(self, img: np.ndarray) -> np.ndarray:
        """blur_cells で指定した格子セル（"J5" 形式、vlm.grid_cols/rows と同じ分割）を黒く塗りつぶす。"""
        if not self.blur_cells:
            return img
        import cv2

        h, w = img.shape[:2]
        cols, rows = self.blur_grid
        cw, ch = w / cols, h / rows
        out = img.copy()
        for cell in self.blur_cells:
            c, r = ord(cell[0]) - ord("A"), int(cell[1:]) - 1
            if not (0 <= c < cols and 0 <= r < rows):
                continue
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = min(w, int((c + 1) * cw) + 1), min(h, int((r + 1) * ch) + 1)
            out[y0:y1, x0:x1] = 0  # 黒塗り
        return out

    def grab(self) -> np.ndarray:
        """BGR 画像 (H, W, 3) uint8 を返す。黒塗り（blur_cells）はここでは掛けない。
        VLM・検出器には生画像を渡し、録画（Recorder）とプレビューだけが mask_cells() を適用する。"""
        import cv2

        if self.cap is None:
            p = Path(self.cfg["mock_image"])
            if not p.exists():  # 画像がなければグレー画面
                return np.full((self.cfg.get("height", 720), self.cfg.get("width", 1280), 3), 128, np.uint8)
            return cv2.imread(str(p))
        # V4L2 のドライババッファに溜まった古いフレームを捨ててから読む。
        # これをしないと、前回取得から時間が空いたとき（教示中・動作後）に古い画が返る。
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < self.cfg.get("flush_sec", 0.4):
            self.cap.grab()
        for _ in range(3):
            ok, frame = self.cap.read()
            if ok:
                return frame
        raise RuntimeError("カメラから画像を取得できません")

    def close(self):
        if self.cap is not None:
            self.cap.release()
