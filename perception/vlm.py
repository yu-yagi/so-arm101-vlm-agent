"""VLM（OpenAI GPT API）呼び出し。

* _ask     : プロンプト + 画像（俯瞰・手首）→ JSON 応答。gpt-6-* は Responses API、それ以外は chat completions
* ask_yesno: 1 枚の画像への yes/no 質問（把持確認など）
* draw_grid: 全セルにラベルを付けた格子を描く（set-of-marks）。VLM はセル名 + セル内位置で場所を答える
送った画像と応答は out_dir に保存する。VLM_MOCK=1 なら API を呼ばず固定応答（配線確認用）。
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

def _b64(img_bgr: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def draw_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """ラベル付き格子を描く。各セルの中央に "C4" のようなラベルを置く（set-of-marks 方式）。
    上端・左端だけのラベルだと VLM が列を読み違えるため、全セルにラベルを入れる。"""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    cw, ch = w / cols, h / rows
    for c in range(cols + 1):
        x = min(int(c * cw), w - 1)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
    for r in range(rows + 1):
        y = min(int(r * ch), h - 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
    scale = max(0.45, min(cw, ch) / 160)
    for c in range(cols):
        for r in range(rows):
            label = f"{chr(ord('A') + c)}{r + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            x = int(c * cw + cw / 2 - tw / 2); y = int(r * ch + ch / 2 + th / 2)
            cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)
    return out


class VLMUnavailable(RuntimeError):
    """API 残高切れ・認証エラーなど、続行不能な失敗。呼び出し側は安全に停止する。"""


class VLM:
    def __init__(self, cfg: dict, out_dir: Optional[Path] = None):
        self.cfg = cfg
        self.mock = os.getenv("VLM_MOCK", "0") == "1"
        self.out_dir = out_dir            # VLM に渡した画像と応答 JSON を保存する先（runs/<run>/）
        self.n_calls = 0
        self.tokens_in = 0; self.tokens_out = 0
        if not self.mock:
            from openai import OpenAI

            self.client = OpenAI()

    # ------------------------------------------------------------------ core
    def _ask(self, prompt: str, images: list[np.ndarray], tag: str) -> dict:
        import cv2

        self.n_calls += 1
        if self.out_dir:
            for i, im in enumerate(images):
                cv2.imwrite(str(self.out_dir / f"{self.n_calls:03d}_{tag}_{i}.jpg"), im)
        if self.mock:
            res = self._mock(tag, prompt)
        else:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(im)}", "detail": "high"}}
                for im in images
            ]
            res = None
            model = self.cfg.get("model", "gpt-4o")
            use_responses = self.cfg.get("api", "auto") == "responses" or (self.cfg.get("api", "auto") == "auto" and model.startswith("gpt-6"))
            for attempt in range(3):
                text = None; note = ""
                if use_responses:
                    # Responses API（GPT-6 Astra）。temperature は使えない。reasoning effort を指定
                    import cv2 as _cv2
                    small = tag.startswith("check")
                    ims = [(_cv2.resize(im, (640, 360)) if small else im) for im in images]
                    rcontent = [{"type": "input_text", "text": prompt}] + [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{_b64(im)}", "detail": "low" if small else self.cfg.get("image_detail", "high")}
                        for im in ims]
                    try:
                        resp = self.client.responses.create(
                            model=self.cfg.get("check_model", model) if tag.startswith("check") else model,
                            reasoning={"effort": self.cfg.get("reasoning_effort", "low")},
                            input=[{"role": "user", "content": rcontent}],
                            text={"format": {"type": "json_object"}},
                        )
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)
                        if "429" in msg or "credit" in msg.lower() or "401" in msg or "insufficient_quota" in msg:
                            raise VLMUnavailable(msg[:200]) from e
                        print(f"[VLM:{tag}] API エラー (試行 {attempt + 1}): {msg[:120]}"); continue
                    text = resp.output_text
                    u = getattr(resp, "usage", None)
                    if u is not None:
                        self.tokens_in += getattr(u, "input_tokens", 0) or 0; self.tokens_out += getattr(u, "output_tokens", 0) or 0
                    note = f"status={getattr(resp, 'status', None)}"
                else:
                    resp = self.client.chat.completions.create(
                        model=model, temperature=self.cfg.get("temperature", 0.0),
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": content}],
                    )
                    msg = resp.choices[0].message; text = msg.content
                    u = getattr(resp, "usage", None)
                    if u is not None:
                        self.tokens_in += getattr(u, "prompt_tokens", 0) or 0; self.tokens_out += getattr(u, "completion_tokens", 0) or 0
                    note = f"refusal={getattr(msg, 'refusal', None)} finish={resp.choices[0].finish_reason}"
                if text:
                    try:
                        res = json.loads(text)
                        break
                    except json.JSONDecodeError:
                        print(f"[VLM:{tag}] JSON 解析失敗 (試行 {attempt + 1}): {text[:200]}")
                else:
                    print(f"[VLM:{tag}] 空応答 (試行 {attempt + 1}) {note}")
            if res is None:
                res = {"error": "no valid response", "steps": [], "found": False, "success": False, "choice": 0}
        if self.out_dir:
            (self.out_dir / f"{self.n_calls:03d}_{tag}.json").write_text(
                json.dumps({"prompt": prompt, "response": res}, ensure_ascii=False, indent=2)
            )
        print(f"[VLM:{tag}] {json.dumps(res, ensure_ascii=False)}")
        return res

    def _mock(self, tag: str, prompt: str) -> dict:
        if tag.startswith("check"):
            return {"answer": True, "reason": "mock"}
        if tag.startswith("agent"):
            n = int(tag[5:])
            seq = [{"name": "goto_cell", "cell": "E4"}, {"name": "move", "direction": "left", "size": "small"},
                   {"name": "rotate", "degrees": 45}, {"name": "grasp"}, {"name": "goto_cell", "cell": "C5"}, {"name": "drop"},
                   {"name": "done", "success": True, "reason": "mock"}]
            return {"observation": "mock", "reasoning": "mock", "action": seq[min(n, len(seq) - 1)], "task_done": False}
        return {"found": False, "answer": True, "reason": "mock"}

    # ------------------------------------------------------------------ api
    def ask_yesno(self, img: np.ndarray, question: str, tag: str = "check") -> tuple[bool, str]:
        res = self._ask(question + '\n出力形式(JSON のみ): {"answer": true|false, "reason": "..."}', [img], tag)
        return bool(res.get("answer", False)), str(res.get("reason", ""))
