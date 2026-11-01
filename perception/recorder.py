"""実行中の俯瞰・手首映像を 1 本の動画に録画する（提出動画の素材）。

同一スレッドから tick() を呼ぶ設計（カメラは別スレッドから触らない）。
  - ロボットの move_to ループと、エージェントの各ステップから tick() を呼ぶ
  - 画面構成: 左 俯瞰 (960x540) / 右上 手首 (320x180) / 右下 キャプション（タスク・ステップ・VLM の判断）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Recorder:
    W, H, FPS = 1280, 720, 10

    def __init__(self, path: Path, cam, wrist_cam=None, task: str = ""):
        self.path = Path(path); self.cam = cam; self.wrist = wrist_cam
        self.task = task; self.step_text = ""; self.obs_text = ""
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
        self.last = 0.0; self.n = 0

    def _read(self, cam):
        if cam is None:
            return None
        cap = getattr(cam, "cap", None)
        if cap is None:
            try:
                f = cam.grab()
            except Exception:
                return None
        else:
            ok, f = cap.read()
            if not ok:
                return None
        mask = getattr(cam, "mask_cells", None)   # 録画にだけ黒塗りを適用（手首カメラは blur_cells が空）
        return mask(f) if mask is not None else f

    def set_step(self, step: str, obs: str = ""):
        self.step_text, self.obs_text = step, obs

    def tick(self, force: bool = False):
        now = time.time()
        if not force and now - self.last < 1.0 / self.FPS:
            return
        self.last = now
        over = self._read(self.cam); wrist = self._read(self.wrist)
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        if over is not None:
            canvas[90:630, 0:960] = cv2.resize(over, (960, 540))
        if wrist is not None:
            canvas[90:270, 960:1280] = cv2.resize(wrist, (320, 180))
        cv2.putText(canvas, "overhead", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, "wrist", (968, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        self._text(canvas, f"Task: {self.task}", (8, 36), 0.8, (255, 255, 255))
        y = 300
        for line in self._wrap(self.step_text, 34)[:4]:
            self._text(canvas, line, (968, y), 0.55, (0, 255, 255)); y += 26
        y += 8
        for line in self._wrap(self.obs_text, 34)[:10]:
            self._text(canvas, line, (968, y), 0.5, (200, 255, 200)); y += 22
        self._text(canvas, time.strftime("%H:%M:%S"), (8, 700), 0.6, (180, 180, 180))
        self.writer.write(canvas); self.n += 1

    @staticmethod
    def _wrap(text: str, n: int):
        text = text or ""
        return [text[i:i + n] for i in range(0, len(text), n)] or [""]

    _FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"

    @classmethod
    def _text(cls, img, s, org, scale, color):
        """日本語は PIL（Noto Sans CJK）で描く。フォントが無ければ ASCII のみ OpenCV で描く。
        注意: ステップ番号の行 "step N:" は make_submission_video.py がテンプレート照合で読むので、ASCII のまま OpenCV で描く。"""
        if s.isascii() or not Path(cls._FONT).exists():
            s = "".join(ch if ord(ch) < 128 else "?" for ch in s)
            cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            return
        from PIL import Image, ImageDraw, ImageFont

        size = int(scale * 30)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((org[0], org[1] - size), s, font=ImageFont.truetype(cls._FONT, size), fill=tuple(int(c) for c in color[::-1]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def close(self):
        self.writer.release()
        return self.path, self.n
