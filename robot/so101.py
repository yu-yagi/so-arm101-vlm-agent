"""SO-ARM101 フォロワーアームの薄いラッパー。

LeRobot の SO101Follower を包み、
  * 関節名→度数の dict で move_to()（安全のため線形補間しながら送る）
  * グリッパ開閉
  * 教示用のトルク OFF/ON
を提供する。backend: mock でハードなしでも同じ API で動く。
"""
from __future__ import annotations

import time
from typing import Dict

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_KEYS = [f"{j}.pos" for j in JOINTS] + ["gripper.pos"]


class BaseRobot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hz = cfg.get("move_hz", 30)
        self.default_sec = cfg.get("default_move_sec", 1.5)
        self.recorder = None   # perception.recorder.Recorder（あれば動作中に tick する）

    # --- サブクラスで実装 ---
    def connect(self): ...
    def disconnect(self): ...
    def _read(self) -> Dict[str, float]: ...
    def _write(self, action: Dict[str, float]): ...
    def set_torque(self, enabled: bool): ...

    # --- 共通 ---
    def joints(self) -> Dict[str, float]:
        """{'shoulder_pan': deg, ..., 'gripper': deg}"""
        obs = self._read()
        return {k[:-4]: float(v) for k, v in obs.items() if k.endswith(".pos")}

    def move_to(self, target: Dict[str, float], sec: float | None = None):
        """現在姿勢から target（関節名→deg、省略した関節は保持）へ線形補間で移動。"""
        sec = self.default_sec if sec is None else sec
        cur = self.joints()
        goal = {**cur, **target}
        n = max(1, int(sec * self.hz))
        for i in range(1, n + 1):
            a = i / n
            step = {f"{k}.pos": cur[k] + (goal[k] - cur[k]) * a for k in goal}
            t0 = time.time()
            self._write(step)
            if self.recorder is not None:
                self.recorder.tick()
            dt = time.time() - t0
            if dt < 1 / self.hz:
                time.sleep(1 / self.hz - dt)

    def load(self) -> Dict[str, float]:
        """各関節の負荷（Feetech Present_Load、符号付き生値）。Mock は 0。"""
        return {j: 0.0 for j in JOINTS}

    def set_torque_limit(self, pct: float, joints=None):
        """出力トルク上限 [%]（STS3215 Torque_Limit, addr 48, 0〜1000）。Mock は何もしない。"""

    def descend_until_contact(self, start: Dict[str, float], end: Dict[str, float], extra_deg: float = 1.5,
                              load_thr: float = -45.0, err_thr: float = 1.2, steps: int = 40, dt: float = 0.1,
                              torque_pct: float = 25.0, backoff_deg: float = 3.0) -> bool:
        """start から end（さらに shoulder_lift を extra_deg だけ）へ小刻みに降ろし、接触で止まって退避する。
        安全策（陶器など壊れ物対応）:
          * 降下中は肩・肘・手首の Torque_Limit を torque_pct% に下げ、当たっても小さな力しか出さない
          * 接触（負荷 < load_thr または追従誤差 > err_thr）を検知したら押し続けず、即座に backoff_deg 戻す
        2026-11-01 実測: 空中の負荷 0〜+30、机に触れると −40 → −160。"""
        goal = {**self.joints(), **end}
        goal["shoulder_lift"] += extra_deg
        cur = {**self.joints(), **start}
        self.set_torque_limit(torque_pct, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        hit = False
        try:
            for i in range(1, steps + 1):
                a = i / steps
                tgt = {k: cur[k] + (goal[k] - cur[k]) * a for k in goal}
                self._write({f"{k}.pos": v for k, v in tgt.items()})
                time.sleep(dt)
                ld = self.load().get("shoulder_lift", 0.0)
                now = self.joints()
                err = tgt["shoulder_lift"] - now["shoulder_lift"]
                if ld < load_thr or err > err_thr:
                    back = dict(now); back["shoulder_lift"] -= backoff_deg   # 即退避（押し続けない）
                    self._write({f"{k}.pos": back[k] for k in goal})
                    time.sleep(0.15)
                    print(f"[contact] step {i}/{steps} lift {now['shoulder_lift']:.1f} load {ld:.0f} err {err:.1f} → {backoff_deg}° 退避")
                    hit = True
                    break
        finally:
            self.set_torque_limit(100.0, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        return hit

    def follow_until_contact(self, waypoints, torque_pct: float = 25.0, load_thr: float = -45.0, err_thr: float = 1.2,
                             sec_per_wp: float = 0.35, backoff_deg: float = 3.0) -> bool:
        """経由点列（関節角 dict のリスト）を低トルクで順に辿り、接触（負荷 or 追従誤差）で止まって退避する。
        垂直降下用: 経由点は FK で高さを刻んで作る（同じ机上点の真上を降りる）。"""
        self.set_torque_limit(torque_pct, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        hit = False
        try:
            for wp0 in waypoints:
                cur = self.joints()
                wp = {**cur, **wp0}   # 全関節（gripper 含む）を揃える: LeRobot の max_relative_target がキー一致を要求
                n = max(2, int(sec_per_wp * self.hz))
                for i in range(1, n + 1):
                    a = i / n
                    tgt = {k: cur[k] + (wp[k] - cur[k]) * a for k in wp}
                    self._write({f"{k}.pos": v for k, v in tgt.items()})
                    if self.recorder is not None:
                        self.recorder.tick()
                    time.sleep(1 / self.hz)
                time.sleep(0.05)
                ld = self.load().get("shoulder_lift", 0.0)
                now = self.joints()
                err = wp["shoulder_lift"] - now["shoulder_lift"]
                if ld < load_thr or err > err_thr:
                    back = dict(now); back["shoulder_lift"] -= backoff_deg
                    self._write({f"{k}.pos": back[k] for k in wp}); time.sleep(0.15)
                    print(f"[contact] lift {now['shoulder_lift']:.1f} load {ld:.0f} err {err:.1f} → {backoff_deg}° 退避")
                    hit = True
                    break
        finally:
            self.set_torque_limit(100.0, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        return hit

    def gripper(self, open_: bool, sec: float = 0.6):
        val = self.cfg["gripper_open"] if open_ else self.cfg["gripper_close"]
        self.move_to({"gripper": val}, sec=sec)

    def home(self, sec: float = 2.5):
        self.move_to(dict(self.cfg["home_pose"]), sec=sec)


class LeRobotSO101(BaseRobot):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        try:  # lerobot >= 0.4: so100/so101 は so_follower に統合
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # 旧版
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        rcfg = SO101FollowerConfig(
            port=cfg["port"],
            id=cfg.get("id", "so101_follower"),
            use_degrees=cfg.get("use_degrees", True),
            max_relative_target=(
                {k: float(v) for k, v in cfg["max_relative_target"].items()} if isinstance(cfg.get("max_relative_target"), dict)
                else float(cfg["max_relative_target"]) if cfg.get("max_relative_target") is not None else None),
            cameras={},  # カメラは perception/camera.py で別管理
        )
        self.robot = SO101Follower(rcfg)

    def connect(self):
        # 未キャリブレーションなら LeRobot の対話キャリブレーションが走る。
        # 接続直後に 1 回だけ応答が無いことがある（手首ロール ID5 で経験）ので、通信エラーは少し待って再試行する
        for attempt in range(3):
            try:
                self.robot.connect()
                return
            except ConnectionError as e:
                print(f"[robot] 接続失敗 ({attempt + 1}/3): {e}")
                try:
                    self.robot.bus.disconnect(False)
                except Exception:
                    pass
                time.sleep(1.0)
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()

    def _read(self):
        return self.robot.get_observation()

    def _write(self, action):
        self.robot.send_action(action)

    def set_torque(self, enabled: bool):
        bus = self.robot.bus
        (bus.enable_torque if enabled else bus.disable_torque)()

    def load(self):
        try:
            return {k: float(v) for k, v in self.robot.bus.sync_read("Present_Load", normalize=False).items()}
        except Exception:
            return {j: 0.0 for j in JOINTS}

    def set_torque_limit(self, pct: float, joints=None):
        val = int(max(0, min(1000, pct * 10)))
        for j in (joints or JOINTS):
            try:
                self.robot.bus.write("Torque_Limit", j, val, normalize=False)
            except Exception as e:  # noqa: BLE001
                print(f"[torque_limit] {j} 書き込み失敗: {e}")


class MockRobot(BaseRobot):
    """ハードなしで動作確認するためのダミー。送った値をそのまま状態として保持。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hz = 1000  # 待たない
        self.state = {f"{k}.pos": float(v) for k, v in cfg["home_pose"].items()}
        self.state["gripper.pos"] = float(cfg["gripper_open"])
        self.log = []

    def connect(self):
        print("[MockRobot] connected")

    def disconnect(self):
        print("[MockRobot] disconnected")

    def _read(self):
        return dict(self.state)

    def _write(self, action):
        self.state.update(action)
        self.log.append(dict(action))

    def set_torque(self, enabled: bool):
        print(f"[MockRobot] torque {'ON' if enabled else 'OFF'}")


def make_robot(cfg: dict) -> BaseRobot:
    if cfg.get("backend", "lerobot") == "mock":
        return MockRobot(cfg)
    return LeRobotSO101(cfg)
