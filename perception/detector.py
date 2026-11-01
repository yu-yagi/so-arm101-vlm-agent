"""オープン語彙物体検出（OWLv2）。自然言語の物体名から画素座標を高精度に求める。

GPT-4o 系 VLM は「どこにあるか」を数十 px の精度でしか答えられない（実測: 格子方式で約 90 px 誤差）。
そこで位置特定はローカル GPU の OWLv2 に任せ、VLM は計画・成否判定・候補の選択に使う。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Detector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("model", "google/owlv2-base-patch16-ensemble")
        self.device = cfg.get("device", "cuda")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.proc = Owlv2Processor.from_pretrained(self.model_name)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device).eval()

    def detect(self, img_bgr: np.ndarray, query: str, roi: Optional[tuple[int, int, int, int]] = None,
               threshold: float = 0.05, topk: int = 5) -> list[dict]:
        """query に合う候補を信頼度順に返す。各要素: {score, box:(x0,y0,x1,y1), center:(u,v)}（画像全体の座標）。
        roi=(x0,y0,x1,y1) を与えるとその範囲だけで検出（作業範囲外の誤検出を避け、小物体の検出率も上がる）。"""
        import cv2
        import torch
        from PIL import Image

        self._load()
        ox, oy = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            img_bgr = img_bgr[y0:y1, x0:x1]; ox, oy = x0, y0
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        q = query if query.lower().startswith(("a ", "an ", "the ")) else f"a {query}"
        with torch.no_grad():
            inp = self.proc(text=[[q]], images=pil, return_tensors="pt").to(self.device)
            out = self._model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, threshold=threshold, target_sizes=torch.tensor([[max(pil.size)] * 2]).to(self.device))[0]
        rows = sorted(zip(res["scores"].tolist(), res["boxes"].tolist()), reverse=True)[:topk]
        out_list = []
        for s, b in rows:
            bx = (int(b[0]) + ox, int(b[1]) + oy, int(b[2]) + ox, int(b[3]) + oy)
            out_list.append({"score": float(s), "box": bx, "center": ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)})
        return out_list


def draw_candidates(img: np.ndarray, cands: list[dict]) -> np.ndarray:
    """候補に 1,2,3... の番号を描く（VLM に選ばせる set-of-marks 用）。"""
    import cv2

    out = img.copy()
    for i, c in enumerate(cands, 1):
        x0, y0, x1, y1 = c["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, str(i), (x0, max(y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, str(i), (x0, max(y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def crop_sheet(img: np.ndarray, cands: list[dict], size: int = 240) -> np.ndarray:
    """各候補を周囲込みで切り抜いて横に並べる（番号付き）。VLM が色・文字を判別しやすくする。"""
    import cv2

    tiles = []
    h, w = img.shape[:2]
    for i, c in enumerate(cands, 1):
        x0, y0, x1, y1 = c["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = int(max(x1 - x0, y1 - y0) * 0.9) + 20
        xa, ya = max(cx - half, 0), max(cy - half, 0)
        xb, yb = min(cx + half, w), min(cy + half, h)
        tile = cv2.resize(img[ya:yb, xa:xb], (size, size))
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (0, 0, 255), 3)
        cv2.putText(tile, str(i), (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(tile, str(i), (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)
