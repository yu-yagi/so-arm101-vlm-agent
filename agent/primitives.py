"""事前定義された動作プリミティブ（VLM エージェントのアクション空間）。

状態: 目標画素 (u, v)（俯瞰画像の机面上の点）、高さレベル、ロール、グリッパ開閉。
各プリミティブは状態を更新して、教示ベースの IK でその姿勢に移動する。
  goto_cell(cell)        : 格子セルの中心へ水平移動（粗い移動）
  move(direction, size)  : 俯瞰画像の上下左右へ small/large だけ水平移動（微調整）
  level(high|hover|table): 昇降。table へは接触検知で降ろす（相手の高さ不要）
  rotate(deg)            : 手首ロールを相対回転
  grasp / release        : グリッパ閉／開
  home                   : 退避
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

LEVELS = ("high", "hover", "table")


class Primitives:
    def __init__(self, robot, ik, ws, cam, wrist_cam, cfg: dict, vlm=None):
        self.robot, self.ik, self.ws, self.cam, self.wrist_cam = robot, ik, ws, cam, wrist_cam
        self.vlm = vlm
        self.fk = getattr(ik, "fk", None)
        self.cfg = cfg
        a = cfg.get("agent", {})
        self.step_px = {"small": a.get("step_small_px", 25), "large": a.get("step_large_px", 60)}
        self.grid = (cfg["vlm"].get("grid_cols", 10), cfg["vlm"].get("grid_rows", 7))
        self.img_size = (cfg["camera"].get("width", 1280), cfg["camera"].get("height", 720))
        self.container_delta = cfg["ik"].get("container_delta", {"shoulder_lift": -35})
        self.max_roll_delta = cfg.get("grasp", {}).get("max_roll_delta", 100)
        # 状態
        pts = getattr(ik, "pts", None)
        c = pts.mean(0) if pts is not None else np.array([640, 400])
        self.u, self.v = float(c[0]), float(c[1])
        self.level = "high"
        self.roll_delta = 0.0
        self.gripper_open = True
        self.holding = False
        self.close_val = cfg["robot"].get("gripper_close", 5.0)
        self.roi = None
        self.fence = None
        self.work = None
        self.held_hsv = None       # 把持した物の色（HSV 中央値）。drop の俯瞰追跡に使う
        # 手首画像の移動 → 俯瞰画素の移動（calib/wrist_jacobian.json: wrist = J @ overhead、hover 高さで計測）
        self.J = None; self.pan_calib = 0.0
        jf = cfg.get("grasp", {}).get("servo_jacobian")
        try:
            import json as _json
            d = _json.load(open(jf)); self.J = np.array(d["J"], float); self.pan_calib = float(d.get("pan_at_calib", 0.0))
        except Exception:
            print("[primitives] wrist_jacobian が読めないため move は俯瞰画像基準になります")
        if pts is not None:
            m = cfg.get("detector", {}).get("roi_margin", 150)
            self.roi = (max(int(pts[:, 0].min() - m), 0), max(int(pts[:, 1].min() - m), 0), int(pts[:, 0].max() + m), int(pts[:, 1].max() + m))
            f = cfg.get("detector", {}).get("fence_margin", 40)
            fb = cfg.get("detector", {}).get("fence_margin_bottom", f)   # 画像下側（台座から遠い側）は腕が伸び切るので狭く
            self.fence = (max(int(pts[:, 0].min() - f), 0), max(int(pts[:, 1].min() - f), 0), int(pts[:, 0].max() + f), int(pts[:, 1].max() + fb))
            # 置き先・目標点を許す範囲（緑枠）。教示範囲の外側は PolarIK で解き、到達不可なら _check_reach が拒否する
            self.work = self.fence

    # ---------------------------------------------------------------- pose
    def _pose(self, level: Optional[str] = None):
        level = level or self.level
        x, y = self.ws.pixel_to_xy(self.u, self.v)
        if getattr(self.ik, "fk", None) is not None:
            q = self.ik.solve(x, y, {"hover": "hover", "high": "carry", "table": "down", "rim": "rim"}[level])
        else:
            q = self.ik.solve(x, y, "up" if level == "hover" else "down")
            if level == "high":
                for j, d in self.container_delta.items():
                    q[j] += d
        taught_roll = q["wrist_roll"]
        q["wrist_roll"] = taught_roll + max(-self.max_roll_delta, min(self.max_roll_delta, self.roll_delta))
        return q

    def _goto(self, sec: float = 1.2):
        """現在の (u,v,level) の姿勢へ。水平移動が大きいときは高さ一定の経由点で辿る（関節補間で先端が沈まないように）。"""
        target = self._pose()
        prev = getattr(self, "_last_uv", None)
        n = 0
        if self.fk is not None and prev is not None and self.level in ("hover", "high"):
            du, dv = self.u - prev[0], self.v - prev[1]
            n = int(((du * du + dv * dv) ** 0.5) // 45)
            for i in range(1, n + 1):
                a = i / (n + 1)
                x, y = self.ws.pixel_to_xy(prev[0] + du * a, prev[1] + dv * a)
                q = self.ik.solve(x, y, {"hover": "hover", "high": "carry"}[self.level])
                q["wrist_roll"] = target["wrist_roll"]
                self.robot.move_to(q, sec=max(0.35, sec / (n + 1)))
        self.robot.move_to(target, sec=sec if n == 0 else max(0.5, sec / (n + 1)))
        self._last_uv = (self.u, self.v)

    def _clamped(self, u: float, v: float):
        box = self.fence or self.roi
        if box:
            u = min(max(u, box[0]), box[2]); v = min(max(v, box[1]), box[3])
        return u, v

    def _clamp(self):
        self.u, self.v = self._clamped(self.u, self.v)

    # ---------------------------------------------------------------- actions
    def _travel_to(self, u: float, v: float):
        """粗い水平移動: carry 高さ（18cm）へ上げてから水平に動き、何も持っていなければ hover へ垂直に降りる。
        hover 高さ（6cm）で横移動しないので、近くの背の高い物（コップ）に当たらない。"""
        u, v = self._clamped(u, v)
        err = self._check_reach(u, v)
        if err:
            return err
        if self.level != "high":
            self.level = "high"; self._goto(sec=1.2)
        self.u, self.v = u, v; self._goto(sec=1.8)
        if not self.holding:
            self.level = "hover"; self._goto(sec=1.2)

    def _travel_level(self) -> str:
        """移動に使う高さ: 何も持っていなければ hover（手首画像で合わせやすい）、持っていれば high（跨ぐ）"""
        return "high" if self.holding else "hover"

    def _safe_gripper(self):
        """何も持っていない移動中はジョーを閉じておく（開いたジョーで物を引っかけない）"""
        if not self.holding and self.gripper_open:
            self.robot.gripper(open_=False, sec=0.5); self.gripper_open = False

    def goto_cell(self, cell: str, **_) -> str:
        """セルへ水平移動。offset={"u":0..1,"v":0..1} でセル内の位置も指定できる（左上 0,0 / 右下 1,1）。"""
        m = re.match(r"\s*([A-Za-z])\s*(\d+)", str(cell))
        if not m:
            return f"不正なセル '{cell}'"
        c = ord(m.group(1).upper()) - ord("A"); r = int(m.group(2)) - 1
        cols, rows = self.grid; w, h = self.img_size
        if not (0 <= c < cols and 0 <= r < rows):
            return f"セル '{cell}' は範囲外"
        self._safe_gripper()
        if self.level == "table" or (self.level == "high" and not self.holding):
            self.level = self._travel_level()
            self._goto(sec=1.0)   # まず上げてから水平移動
        fu, fv = 0.5, 0.5
        off = _.get("offset") if isinstance(_.get("offset"), dict) else None
        if off:
            fu = min(max(float(off.get("u", 0.5)), 0.0), 1.0); fv = min(max(float(off.get("v", 0.5)), 0.0), 1.0)
        nu, nv = (c + fu) * w / cols, (r + fv) * h / rows
        if self.work and not (self.work[0] <= nu <= self.work[2] and self.work[1] <= nv <= self.work[3]):
            return (f"{cell.upper()} (u={nu:.0f}, v={nv:.0f}) は緑枠（可動範囲）の外なので移動しません。"
                    f"緑枠の内側（{self.work_cells_text()}）から選び直してください")
        err = self._check_reach(nu, nv)
        if err:
            return err
        self.u, self.v = nu, nv
        self._clamp(); self._goto(sec=1.8)
        return f"セル {cell.upper()}" + (f"（セル内 u={fu:.1f}, v={fv:.1f}）" if off else " の中心") + f"へ移動（level={self.level}）"

    def _check_reach(self, u: float, v: float) -> Optional[str]:
        """(u,v) が机面・運搬高さの両方で解けるか。解けなければ理由文字列（移動はしない）。"""
        from robot.kinematics import ReachError

        x, y = self.ws.pixel_to_xy(u, v)
        try:
            self.ik.solve(x, y, "down")
            if getattr(self.ik, "fk", None) is not None:
                self.ik.solve(x, y, "carry")
        except ReachError as e:
            return f"目標 (u={u:.0f}, v={v:.0f}) にはアームが届きません（{e}）。台座に近い側の別の場所を選んでください"
        return None

    def work_cells_text(self) -> str:
        """緑枠（教示範囲）が覆うセルの範囲を '列 C〜F, 行 3〜6' の形で返す。"""
        if not self.work:
            return "不明"
        cols, rows = self.grid; w, h = self.img_size
        c0, c1 = int(self.work[0] * cols / w), int(min(self.work[2], w - 1) * cols / w)
        r0, r1 = int(self.work[1] * rows / h) + 1, int(min(self.work[3], h - 1) * rows / h) + 1
        return f"列 {chr(ord('A') + c0)}〜{chr(ord('A') + c1)}, 行 {r0}〜{r1}（枠の縁のセルは一部しか入らないので、中心寄りを選ぶ）"

    def _wrist_to_overhead(self, wdir: np.ndarray) -> np.ndarray:
        """手首画像上の方向ベクトル → 俯瞰画素上の方向ベクトル（単位）。
        手首画像はロールとパンの変化で回転するので、校正時からの差分だけ戻してから J^-1 を掛ける。"""
        if self.J is None:
            return wdir / (np.linalg.norm(wdir) + 1e-9)
        pan = self._pose()["shoulder_pan"]
        ang = np.deg2rad(self.roll_delta + (pan - self.pan_calib))
        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        # J は「カメラ（俯瞰画素）が o 動いたとき、景色が手首画像で w = J o 動く」の関係。
        # 十字（カメラ）を wdir の向きへ動かしたい ⇔ 景色は −wdir 方向へ動く ⇒ o = −J⁻¹ R⁻¹ wdir
        o = -np.linalg.solve(self.J, R @ wdir)
        return o / (np.linalg.norm(o) + 1e-9)

    def move(self, direction: str, size: str = "small", **_) -> str:
        """手首画像（2 枚目）の上下左右に微調整。内部で俯瞰画素の移動に変換する。"""
        d = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
             "forward": (0, -1), "backward": (0, 1)}.get(str(direction).lower())
        if d is None:
            return f"不正な方向 '{direction}'"
        px = self.step_px.get(str(size).lower(), self.step_px["small"])
        self._safe_gripper()
        if self.level == "table" or (self.level == "high" and not self.holding):
            self.level = self._travel_level()
        o = self._wrist_to_overhead(np.array(d, float)) * px
        nu, nv = self._clamped(self.u + float(o[0]), self.v + float(o[1]))
        err = self._check_reach(nu, nv)
        if err:
            return err
        self.u, self.v = nu, nv
        self._goto(sec=0.8)
        return f"手首画像の {direction} へ約 {px}px 移動（level={self.level}）"

    def shift(self, direction: str, size: str = "small", **_) -> str:
        """俯瞰画像（1 枚目）の上下左右に水平移動。背の高い容器の上に合わせるときはこちら（俯瞰の target 十字を見る）。"""
        d = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(str(direction).lower())
        if d is None:
            return f"不正な方向 '{direction}'"
        px = self.step_px.get(str(size).lower(), self.step_px["small"]) * 1.5
        self._safe_gripper()
        self._travel_to(self.u + d[0] * px, self.v + d[1] * px)
        return f"俯瞰画像の {direction} へ約 {px:.0f}px 移動（level={self.level}）"

    def hover(self, **_) -> str:
        """今の目標点の真上で hover（6cm）まで垂直に降りる。持っている物を低い物の上に置く前の位置合わせ用。
        運搬高さからの降下は低トルクで 2cm 刻みに辿り、途中で何か（コップの縁など）に触れたら止めて運搬高さへ戻す。"""
        if self.level == "high" and self.fk is not None:
            hit = self._lower_contact("hover")
            if hit:
                self.level = "high"; self._goto(sec=1.2)
                return ("降下中に何かに触れたため中止し、運搬高さ（18cm）へ戻った。真下に背の高い物（コップなど）がある。"
                        "容器に入れるなら hover せず、俯瞰画像で target を容器の底に合わせて drop する")
            self.level = "hover"
        else:
            self.level = "hover"; self._goto(sec=1.2)
        return "hover（6cm）へ降りた。手首画像の十字で置き先の中心を確認し、move で合わせてから place"

    def _lower_contact(self, level: str) -> bool:
        """運搬高さ（18cm）から level（hover / rim）の高さへ、低トルクで 2cm 刻みに垂直降下する。接触したら退避して True。"""
        a = self.cfg.get("agent", {})
        down = self._pose("table")
        z_top, z_to = self.ik.levels.get("carry", 0.18), self.ik.levels[{"hover": "hover", "rim": "rim"}[level]]
        wps = []
        z = z_top - 0.02
        while z > z_to + 0.005:
            kp = 0.15 + 0.85 * (z_top - z) / max(z_top - z_to, 1e-3)       # 俯角の拘束を carry(0.15) → 1.0 へ滑らかに
            q = self.fk.raise_pose(down, z, keep_pitch=kp); q["wrist_roll"] = self._pose(level)["wrist_roll"]; wps.append(q)
            z -= 0.02
        wps.append(self._pose(level))
        hit = self.robot.follow_until_contact(wps, torque_pct=a.get("contact_torque_pct", 25), backoff_deg=a.get("contact_backoff_deg", 3.0))
        if not hit:
            self.level = level
        return hit

    def level_to(self, level: str, **_) -> str:
        level = str(level).lower()
        if level not in LEVELS:
            return f"不正なレベル '{level}'"
        if level == "table":
            return "table への降下は grasp / release に含まれています"
        self.level = level
        self._goto(sec=1.2)
        return f"level={level} へ移動"

    def rotate(self, degrees: float = 0, **_) -> str:
        try:
            deg = float(degrees)
        except (TypeError, ValueError):
            return f"不正な角度 '{degrees}'"
        self.roll_delta = max(-self.max_roll_delta, min(self.max_roll_delta, self.roll_delta + deg))
        self.robot.move_to({"wrist_roll": self._pose()["wrist_roll"]}, sec=0.8)
        return f"ロール {deg:+.0f}°（累積 {self.roll_delta:+.0f}°）"

    def _descend(self) -> bool:
        """hover から机面へ。FK があれば同じ机上点の真上を 1cm 刻みで垂直に降り、接触で止まる。"""
        a = self.cfg.get("agent", {})
        start = self._pose("hover")
        if self.level != "hover":
            self.robot.move_to(start, sec=1.0)
        down = self._pose("table")
        if self.fk is not None:
            z_top = self.ik.levels.get("hover", 0.06)
            wps = []
            z = z_top - 0.01
            while z > 0.004:
                q = self.fk.raise_pose(down, z, keep_pitch=1.0); q["wrist_roll"] = down["wrist_roll"]; wps.append(q); z -= 0.01
            last = dict(down); last["shoulder_lift"] += a.get("contact_extra_deg", 1.5); wps.append(last)
            hit = self.robot.follow_until_contact(wps, torque_pct=a.get("contact_torque_pct", 25), backoff_deg=a.get("contact_backoff_deg", 3.0))
        else:
            hit = self.robot.descend_until_contact(start, down, extra_deg=a.get("contact_extra_deg", 1.5),
                                                   torque_pct=a.get("contact_torque_pct", 25), backoff_deg=a.get("contact_backoff_deg", 3.0))
        self.level = "table"
        return hit

    def _held_by_wrist_cam(self) -> tuple[bool, str]:
        """手首画像を VLM に見せて、指の間に物体を掴んでいるか確認する（開度だけでは誤判定するため）。"""
        if self.wrist_cam is None or self.vlm is None:
            g = self.robot.joints()["gripper"]
            return g > self.close_val + 4, f"開度 {g:.0f}"
        import time; time.sleep(0.3)
        img = self.wrist_cam.grab()
        ok, why = self.vlm.ask_yesno(img, "これはロボットの手首カメラの画像です。画面下の白い部品はグリッパの指です。"
                                          "いま指の間に物体（例: 消しゴム）を挟んで持ち上げていますか？ 机の上に物が見えているだけなら false。", tag="check_hold")
        return ok, why

    def _contact_height(self) -> float:
        """接触で止まった時点の先端高さ [m]（FK）。机面なら ≈0、物体の上に指が乗ったなら物体の高さ。"""
        if self.fk is None:
            return 0.0
        j = self.robot.joints()
        z, _ = self.fk.tip(j["shoulder_lift"], j["elbow_flex"], j["wrist_flex"])
        return float(z)

    def _object_gone(self, uv0, name_hint: str = "") -> bool:
        """俯瞰で把持位置に物体が残っていないか（残っていれば False）。検出器が無ければ True。"""
        try:
            from perception.detector import Detector
        except Exception:
            return True
        det = getattr(self, "_det", None)
        if det is None:
            try:
                self._det = det = Detector(self.cfg.get("detector", {}))
            except Exception:
                return True
        img = self.cam.grab()
        for q in ([name_hint] if name_hint else []) + ["object", "small object"]:
            cands = det.detect(img, q, roi=self.roi, threshold=0.15, topk=5)
            if any(((c["center"][0] - uv0[0]) ** 2 + (c["center"][1] - uv0[1]) ** 2) ** 0.5 < 40 for c in cands):
                return False
            if cands:
                break
        return True

    def _sample_color(self, img, center, r: int = 18):
        import cv2
        x, y = int(center[0]), int(center[1])
        patch = img[max(y - r, 0):y + r, max(x - r, 0):x + r]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        med = np.median(hsv, axis=0)
        return med if med[1] > 50 else None   # 彩度が低い（白・灰）物は追跡しない

    def _track_color(self, img, hsv_ref):
        import cv2
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for h_tol, s_min, v_min in ((12, max(hsv_ref[1] - 70, 40), max(hsv_ref[2] - 80, 40)), (15, 60, 40), (20, 40, 30)):
            lo = np.array([max(hsv_ref[0] - h_tol, 0), s_min, v_min]); hi = np.array([min(hsv_ref[0] + h_tol, 179), 255, 255])
            m = cv2.inRange(hsv, lo, hi); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) > 600]
            if cnts:
                c = max(cnts, key=cv2.contourArea); M = cv2.moments(c)
                return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        return None

    def _servo_object_to(self, hsv_ref, target, iters: int = 2) -> Optional[np.ndarray]:
        """手首画像で色追跡した物体が target[px] に来るよう俯瞰画素を修正（物体が押されてずれても追従）。"""
        w = None; prev = None
        max_step = self.cfg.get("agent", {}).get("servo_max_step_px", 30)
        import time
        for _ in range(iters):
            w = self._track_color(self.wrist_cam.grab(), hsv_ref)
            if w is None or self.J is None:
                return w
            if prev is not None and np.hypot(*(w - prev)) > 250:
                print("[servo] 追跡位置が大きく跳んだため中止（別の物体を追った可能性）"); return None
            prev = w
            dw = np.array(target, float) - w
            if np.hypot(*dw) < 12:
                return w
            o = np.linalg.solve(self.J, dw)
            n = float(np.hypot(*o))
            if n > max_step:
                o = o / n * max_step
            nu, nv = self._clamped(self.u + float(o[0]), self.v + float(o[1]))
            if self._check_reach(nu, nv):
                print("[servo] 到達不可の方向のため中止"); return None
            self.u, self.v = nu, nv; self._goto(sec=0.8)
            time.sleep(0.3)
        return self._track_color(self.wrist_cam.grab(), hsv_ref)

    def grasp(self, **_) -> str:
        """事前定義の把持動作（固定指が片側にあるグリッパ用）。
        物体を手首画像で色追跡して十字に合わせてから垂直降下し、接触高さ（FK）で状況を判定して 1 次元探索する:
          ・物体の上で止まった（>1.2cm）= 固定指が物体に乗った → 物体が画像で右（可動ジョー側）へ来るようにずらして再試行
          ・机面まで降りて空振り = 物体が隙間の外 → 物体が画像で左（固定指側）へ来るようにずらして再試行
          ・把持できたら 手首画像(VLM) と 俯瞰（元の位置から消えた）の両方で確認
        """
        import time

        if self.holding:
            return "既に何かを持っています（先に place / drop）"
        a = self.cfg.get("agent", {})
        u0, v0 = self.u, self.v
        on_top_thr = a.get("on_top_z", 0.012)
        step_img = a.get("grasp_search_step_img_px", 40)   # 1 回の探索で物体の画像位置をずらす量 [手首px]（約 1cm）
        max_tries = a.get("grasp_max_tries", 6)
        tp = np.array(self.cfg.get("grasp", {}).get("tool_point", (640, 360)), float)
        hsv_ref = None
        if self.wrist_cam is not None:
            self.level = "hover"; self._goto(sec=0.8)
            hsv_ref = self._sample_color(self.wrist_cam.grab(), tp)
        target = tp.copy()
        tried = []
        for i in range(max_tries):
            self.level = "hover"; self._goto(sec=0.8)
            if hsv_ref is not None:
                w = self._servo_object_to(hsv_ref, target, iters=5)
                if w is None:
                    tried.append("追跡不能"); break
                if np.hypot(*(np.array(target) - w)) > 25:
                    tried.append(f"位置合わせ不十分({int(w[0])},{int(w[1])})")
                    continue
            if not self.gripper_open:
                self.robot.gripper(open_=True, sec=0.5); self.gripper_open = True
            hit = self._descend()
            zc = self._contact_height()
            if hit and zc > on_top_thr:
                tried.append(f"x={int(target[0])}:指が物体の上({zc*100:.1f}cm)")
                print(f"[grasp] 試行 {i + 1}: 物体 x≈{int(target[0])} で固定指が乗った（{zc*100:.1f}cm）→ 物体を右へ {step_img}px")
                target = target + np.array([step_img, 0.0])      # 物体を可動ジョー側（画像右）へ
                self.level = "hover"; self._goto(sec=0.8)
                continue
            self.robot.gripper(open_=False, sec=0.7); self.gripper_open = False
            self.level = "hover"; self._goto(sec=1.0); time.sleep(0.3)
            held, why = self._held_by_wrist_cam()
            gone = self._object_gone((u0, v0))   # 参考情報（隣に似た物があると誤る）
            print(f"[grasp] 試行 {i + 1}: 物体 x≈{int(target[0])} 接触 {zc*100:.1f}cm / 手首判定 {'把持' if held else '空'} / 俯瞰 {'消えた' if gone else '残っている(参考)'}（{why[:50]}）")
            if held:
                self.level = "high"; self._goto(sec=1.5); time.sleep(0.3)
                held2, _ = self._held_by_wrist_cam()
                if held2:
                    self.holding = True
                    self.held_hsv = hsv_ref
                    return f"把持成功（手首画像と俯瞰で確認、試行 {i + 1} 回目）。運搬高さへ"
                print("[grasp] 持ち上げ中に落とした")
                self.level = "hover"; self._goto(sec=1.0)
            tried.append(f"x={int(target[0])}:空振り")
            self.robot.gripper(open_=True, sec=0.4); self.gripper_open = True
            target = target - np.array([step_img * 0.6, 0.0])   # 隙間の外 → 物体を固定指側（画像左）へ少し
        self.u, self.v = u0, v0; self.level = "hover"; self._goto(sec=0.8)
        self.robot.gripper(open_=False, sec=0.4); self.gripper_open = False
        return "把持失敗（" + ", ".join(tried) + "）。手首画像で向きを見直し rotate してから再度 grasp"

    def _still_holding(self) -> bool:
        held, _ = self._held_by_wrist_cam()
        if not held:
            self.holding = False
        return self.holding

    def place(self, **_) -> str:
        """机や低い物の上に置く: 低トルクで接触まで降ろす → 開く → 持ち上げる。"""
        if not self.holding:
            return "何も持っていないので place できません（先に grasp）"
        if not self._still_holding():
            return "持っていた物を途中で落としたようです（開度が閉じきり）。grasp からやり直してください"
        hit = self._descend()
        self.robot.gripper(open_=True, sec=0.6); self.gripper_open = True
        self.holding = False
        self.level = "hover"; self._goto(sec=1.0)
        self.robot.gripper(open_=False, sec=0.4); self.gripper_open = False
        self.home()
        return ("置いた（" + ("物に触れた高さで開放" if hit else "机面で開放") + "）→ 退避姿勢へ戻った。"
                "俯瞰画像で置いた位置を確認し、指示どおりなら done、ずれていれば掴み直して置き直す")

    def _track_near(self, img, hsv_ref, center, radius: int = 220):
        """俯瞰画像の center 周辺だけで色追跡（別の同色物を拾わないため）。"""
        h, w = img.shape[:2]
        x0, y0 = max(int(center[0] - radius), 0), max(int(center[1] - radius), 0)
        x1, y1 = min(int(center[0] + radius), w), min(int(center[1] + radius), h)
        p = self._track_color(img[y0:y1, x0:x1], hsv_ref)
        return None if p is None else p + np.array([x0, y0], float)

    def _ask_container(self):
        """VLM に、target 近くの容器の「底の中心」「口の中心」「高さ」を聞く。失敗時は None。"""
        if self.vlm is None or getattr(self.vlm, "mock", False):
            return None
        disp, _, _ = self.observe()
        cols, rows = self.grid; w, h = self.img_size
        prompt = ("画像は俯瞰カメラで、黄色の格子（各セル中央にラベル）と赤い target 十字があります。"
                  "target 十字の近くにある容器（コップなど）について答えてください。\n"
                  "base: 容器の底（机に接している部分）の中心、rim: 容器の口（上の縁が作る楕円）の中心を、"
                  "それぞれ最も近いセルのラベルとセル内相対位置 u（左端=0,右端=1）, v（上端=0,下端=1）で。height_cm: 容器の高さの推定 [cm]。\n"
                  '出力形式(JSON のみ): {"found": true, "base": {"cell": "F5", "u": 0.5, "v": 0.4}, '
                  '"rim": {"cell": "F4", "u": 0.6, "v": 0.5}, "height_cm": 9}')
        try:
            res = self.vlm._ask(prompt, [disp], "container")
        except Exception as e:  # noqa: BLE001
            print(f"[drop] 容器の問い合わせ失敗: {e}"); return None
        if not res.get("found", True):
            return None

        def to_px(d):
            m = re.match(r"\s*([A-Za-z])\s*(\d+)", str(d.get("cell", "")))
            if not m:
                return None
            c = ord(m.group(1).upper()) - ord("A"); r = int(m.group(2)) - 1
            fu = min(max(float(d.get("u", 0.5)), 0.0), 1.0); fv = min(max(float(d.get("v", 0.5)), 0.0), 1.0)
            return np.array([(c + fu) * w / cols, (r + fv) * h / rows])

        try:
            base, rim = to_px(res.get("base", {})), to_px(res.get("rim", {}))
            hc = float(res.get("height_cm", 9.0)) / 100.0
        except Exception:  # noqa: BLE001
            return None
        if base is None or rim is None:
            return None
        return {"base": base, "rim": rim, "height": min(max(hc, 0.03), 0.20)}

    def _servo_overhead(self, goal, max_iter: int, tol: float = 15.0, torque_pct: Optional[float] = None) -> Optional[float]:
        """俯瞰画像で把持物の色を追跡し、その見かけ位置が goal[px] に来るよう水平移動する（閉ループ）。
        戻り値: 最終誤差 [px]（追跡不能なら None）。"""
        import time
        max_step = self.cfg.get("agent", {}).get("servo_max_step_px", 30)
        err = None
        if torque_pct is not None:
            self.robot.set_torque_limit(torque_pct, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        try:
            for i in range(max_iter):
                time.sleep(0.25)
                p = self._track_near(self.cam.grab(), self.held_hsv, goal, 260)
                if p is None:
                    print("[drop] 俯瞰で把持物を追跡できない"); return None
                e = np.array(goal, float) - p
                err = float(np.hypot(*e))
                print(f"[drop] servo {i}: 物体 ({p[0]:.0f},{p[1]:.0f}) → 目標 ({goal[0]:.0f},{goal[1]:.0f}) 誤差 {err:.0f}px")
                if err < tol:
                    break
                step = e * 0.8
                n = float(np.hypot(*step))
                if n > max_step:
                    step = step / n * max_step
                nu, nv = self._clamped(self.u + float(step[0]), self.v + float(step[1]))
                if self._check_reach(nu, nv):
                    print("[drop] 到達不可の方向のため中止"); break
                self.u, self.v = nu, nv; self._goto(sec=0.8)
                if torque_pct is not None:
                    ld = self.robot.load()
                    if min(ld.get(j, 0.0) for j in ("shoulder_lift", "elbow_flex", "wrist_flex")) < -60:
                        print("[drop] 横移動中に負荷を検出 → 中止"); return err
        finally:
            if torque_pct is not None:
                self.robot.set_torque_limit(100.0, ["shoulder_lift", "elbow_flex", "wrist_flex"])
        return err

    def drop(self, **_) -> str:
        """容器に入れる。手順（陶器などに押し付けないための設計）:
          1. 運搬高さ（18cm）のまま、俯瞰画像で把持物を色追跡し、容器の口の真上に合わせる（高さ差の見かけのずれは容器自身の
             底→口のずれから推定）。FK の高さ方向の誤差に依存しない閉ループ。
          2. 口の少し上（高さ +4cm）まで低トルクで垂直降下（縁に触れたら止めて 18cm へ戻す）。
          3. その高さで再度、把持物を口の中心に合わせ（低トルク）、開いて落とす → 退避。
        容器が見つからない／追跡できないときは従来どおり 18cm で開く。"""
        import time
        if not self.holding:
            return "何も持っていないので drop できません（先に grasp）"
        if not self._still_holding():
            return "持っていた物を途中で落としたようです（開度が閉じきり）。grasp からやり直してください"
        if self.level != "high":
            self.level = "high"; self._goto(sec=1.0)
        note = ""
        info = None
        if self.held_hsv is not None and self.fk is not None and self.vlm is not None and not getattr(self.vlm, "mock", False):
            u0, v0 = self.u, self.v
            self.home()                                   # アームが容器に被らない画像で容器の形を聞く
            info = self._ask_container()
            self.u, self.v = u0, v0
            self.level = "high"; self._goto(sec=1.8)
        if info is not None:
            base, rim, hc = info["base"], info["rim"], info["height"]
            k = (rim - base) / hc                                       # 高さ 1m あたりの見かけのずれ [px/m]（この容器から推定）
            z_carry = self.ik.levels.get("carry", 0.18)
            h_obj = z_carry - 0.02                                      # 把持物の高さ（指先の少し下）
            goal1 = rim + k * (h_obj - hc)                              # 18cm で把持物が口の真上にあるときの見かけ位置
            print(f"[drop] 容器 底=({base[0]:.0f},{base[1]:.0f}) 口=({rim[0]:.0f},{rim[1]:.0f}) 高さ {hc*100:.0f}cm → 18cm での目標 ({goal1[0]:.0f},{goal1[1]:.0f})")
            e1 = self._servo_overhead(goal1, max_iter=6, tol=15.0)
            if e1 is None:
                note = "（俯瞰で把持物を追跡できず、18cm で開放）"
            else:
                # 口の少し上まで降りる（rim レベルを動的に定義）
                self.ik.levels["rim"] = hc + 0.04; self.ik.keep_pitch["rim"] = 1.0
                hit = self._lower_contact("rim")
                if hit:
                    self.level = "high"; self._goto(sec=1.0)
                    return ("降下中に容器の縁に触れたため中止し、18cm へ戻った（まだ持っている）。"
                            "俯瞰画像で target を容器の口の中心に合わせ直してから、再度 drop")
                z_obj = self.ik.levels["rim"] - 0.02
                goal2 = rim + k * (z_obj - hc)                          # ほぼ口と同じ高さなので見かけのずれはごく小さい
                e2 = self._servo_overhead(goal2, max_iter=5, tol=12.0, torque_pct=self.cfg.get("agent", {}).get("contact_torque_pct", 25))
                note = f"（口の上 {int((self.ik.levels['rim']-hc)*100)}cm で開放、位置誤差 {e2:.0f}px）" if e2 is not None else "（口の上で開放）"
        else:
            note = "（容器情報なし: 18cm で開放）"
        self.robot.gripper(open_=True, sec=0.6); self.gripper_open = True
        self.holding = False
        time.sleep(0.4)
        self.robot.gripper(open_=False, sec=0.4); self.gripper_open = False
        if self.level != "high":
            self.level = "high"; self._goto(sec=1.0)
        self.home()
        return f"開いて落とした{note} → 退避姿勢へ戻った。俯瞰画像で容器に入ったか確認し、入っていれば done"

    def release(self, **_) -> str:
        return self.place()

    def home(self, **_) -> str:
        if self.level != "high":          # まず真上に上げてから退避（低い高さで横に振らない）
            self.level = "high"; self._goto(sec=1.0)
        self.roll_delta = 0.0
        self._safe_gripper()
        self.robot.home(sec=2.5)
        self._last_uv = None              # 次の移動で古い位置からの経由点を作らない
        return "退避姿勢へ"

    # ---------------------------------------------------------------- observation
    def state_text(self) -> str:
        cols, rows = self.grid; w, h = self.img_size
        c = chr(ord("A") + int(self.u * cols / w)); r = int(self.v * rows / h) + 1
        return (f"目標点: セル {c}{r} 付近 (u={self.u:.0f}, v={self.v:.0f}) / 高さ: {self.level} / "
                f"ロール: {self.roll_delta:+.0f}° / グリッパ: {'開' if self.gripper_open else '閉'} / "
                f"把持中: {'はい' if self.holding else 'いいえ'} / 緑枠（置ける範囲）: {self.work_cells_text()}")

    def observe(self):
        """(俯瞰画像[格子+目標点付き], 手首画像)"""
        import cv2

        from perception.vlm import draw_grid

        over = self.cam.grab()
        disp = draw_grid(over, *self.grid)
        if self.work:
            cv2.rectangle(disp, (self.work[0], self.work[1]), (self.work[2], self.work[3]), (0, 255, 0), 2)
            cv2.putText(disp, "reachable", (self.work[0] + 6, self.work[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.drawMarker(disp, (int(self.u), int(self.v)), (0, 0, 255), cv2.MARKER_CROSS, 36, 3)
        cv2.putText(disp, "target", (int(self.u) + 20, int(self.v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        wrist = self.wrist_cam.grab() if self.wrist_cam is not None else None
        if wrist is not None:
            tp = tuple(self.cfg.get("grasp", {}).get("tool_point", (640, 360)))
            cv2.drawMarker(wrist, tp, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(wrist, "fingertip", (tp[0] + 20, tp[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return disp, wrist, over
