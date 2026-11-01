"""設定読み込みと run ディレクトリ作成。"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# OpenCV 同梱 Qt のフォント警告を抑止（全スクリプト共通。cv2 の import 前に効く）
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(open(p))
    # 相対パスは src/ 基準に
    cfg["ik"]["grid_file"] = str(ROOT / cfg["ik"]["grid_file"])
    cfg["workspace"]["homography_file"] = str(ROOT / cfg["workspace"]["homography_file"])
    cfg["camera"]["mock_image"] = str(ROOT / cfg["camera"]["mock_image"])
    cfg["camera"].setdefault("blur_grid", [cfg["vlm"].get("grid_cols", 10), cfg["vlm"].get("grid_rows", 7)])
    cfg["run"]["out_dir"] = str(ROOT / cfg["run"]["out_dir"])
    if "grasp" in cfg and cfg["grasp"].get("servo_jacobian"):
        cfg["grasp"]["servo_jacobian"] = str(ROOT / cfg["grasp"]["servo_jacobian"])
    return cfg


def make_run_dir(cfg: dict, tag: str = "run") -> Path:
    d = Path(cfg["run"]["out_dir"]) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


WINDOW_POS = (2000, 80)  # 画像ウィンドウの初期位置。メイン画面 DP-2 (1920x1080+1920+0) の左上付近


def show(name: str, img):
    """cv2.imshow + 初回はメイン画面へ移動。"""
    import cv2

    first = not getattr(show, "_seen", set()).__contains__(name)
    cv2.imshow(name, img)
    if first:
        cv2.moveWindow(name, *WINDOW_POS)
        show._seen = getattr(show, "_seen", set()) | {name}
