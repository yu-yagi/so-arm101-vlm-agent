"""SO-ARM101 × VLM: 自然言語指示から未知タスクをゼロショット実行する（VLM エージェントモード）。

使い方:
  python main.py "青い消しゴムをコップに入れて"
  python main.py "青い消しゴムをカレンダーの上に乗せて" --dry-run   # ハードなし・VLM 呼び出しなし（Mock）
  python main.py --interactive                                        # 対話で指示を繰り返す

流れ: 毎ステップ 俯瞰画像（格子+目標点+到達範囲）と手首画像（指先十字）を VLM に渡し、
      事前定義コマンド（goto_cell / move / shift / rotate / grasp / hover / place / drop / done）を 1 つ選ばせて実行。
      実行中の映像と VLM の判断は runs/<run>/ に画像・JSON・mp4 で残す。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2

from common import load_config, make_run_dir
from perception.camera import Camera
from perception.vlm import VLM
from perception.workspace import Workspace
from robot.kinematics import make_ik
from robot.so101 import make_robot


def workspace_roi(cfg: dict):
    """教示点の外接矩形 + 余白を検出 ROI にする（scripts/calibrate_wrist.py が使う）。"""
    import numpy as np

    p = Path(cfg["ik"]["grid_file"])
    if not p.exists():
        return None
    pts = np.array([[q["x"], q["y"]] for q in json.load(open(p))["points"]])
    m = cfg["detector"].get("roi_margin", 60)
    x0, y0 = (pts.min(0) - m).clip(0).astype(int)
    x1, y1 = (pts.max(0) + m).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def build(cfg: dict, dry_run: bool, run_dir: Path):
    if dry_run:
        cfg["robot"]["backend"] = "mock"
        cfg["camera"]["backend"] = "mock"
        os.environ["VLM_MOCK"] = "1"
    robot = make_robot(cfg["robot"])
    cam = Camera(cfg["camera"])
    vlm = VLM(cfg["vlm"], out_dir=run_dir)
    try:
        ws = Workspace(cfg["workspace"]["homography_file"])
    except FileNotFoundError as e:
        if not dry_run:
            raise
        print(f"[dry-run] {e} → 単位行列で代用")
        import numpy as np

        # 画像中心を台座前方 0.20 m に写す疑似ホモグラフィ（0.4 mm/px、画像下→ロボット手前、画像右→ロボット右）
        s_, w_, h_ = 0.0004, cfg["camera"].get("width", 1280), cfg["camera"].get("height", 720)
        ws = Workspace.__new__(Workspace)
        ws.H = np.array([[0, -s_, 0.20 + s_ * h_ / 2], [-s_, 0, s_ * w_ / 2], [0, 0, 1.0]])
        ws.Hinv = np.linalg.inv(ws.H)
    ik = make_ik(cfg["ik"])
    return robot, cam, vlm, ws, ik


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", nargs="?", help="自然言語の指示")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="実機・カメラ・API なしで配線確認")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    if not args.instruction and not args.interactive:
        ap.error("指示を与えるか --interactive を指定してください")

    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg, "dry" if args.dry_run else "run")
    print(f"[run_dir] {run_dir}")
    robot, cam, vlm, ws, ik = build(cfg, args.dry_run, run_dir)
    wrist_cam = None
    if not args.dry_run and cfg["camera"].get("wrist_device"):
        wcfg = dict(cfg["camera"]); wcfg["device"] = cfg["camera"]["wrist_device"]
        wrist_cam = Camera(wcfg)
    robot.connect()
    cam.open()
    if wrist_cam is not None:
        wrist_cam.open()

    def run_one(instr: str) -> bool:
        recorder = None
        if cfg.get("run", {}).get("record", True) and not args.dry_run:
            from perception.recorder import Recorder

            recorder = Recorder(run_dir / f"video_{time.strftime('%H%M%S')}.mp4", cam, wrist_cam, task=instr)
            robot.recorder = recorder
        try:
            from agent.primitives import Primitives
            from agent.vlm_agent import VLMAgent

            prims = Primitives(robot, ik, ws, cam, wrist_cam, cfg, vlm=vlm)
            agent = VLMAgent(vlm, prims, max_steps=cfg.get("agent", {}).get("max_steps", 30), out_dir=run_dir, recorder=recorder)
            ok = agent.run(instr)
            prims.home()
            return ok
        finally:
            if recorder is not None:
                robot.recorder = None
                path, n = recorder.close()
                print(f"[video] {path} ({n} frames, {n / recorder.FPS:.0f}s)")

    try:
        robot.home()
        if args.interactive:
            while True:
                try:
                    instr = input("\n指示> ").strip()
                except EOFError:
                    break
                if instr in ("", "q", "quit", "exit"):
                    break
                run_one(instr)
        else:
            ok = run_one(args.instruction)
            sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\n中断")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            robot.home()
        finally:
            cam.close()
            if wrist_cam is not None:
                wrist_cam.close()
            robot.disconnect()
            print(f"[VLM calls] {vlm.n_calls}  tokens in/out {vlm.tokens_in}/{vlm.tokens_out}  [logs] {run_dir}")


if __name__ == "__main__":
    main()
