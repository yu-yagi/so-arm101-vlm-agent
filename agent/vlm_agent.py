"""VLM エージェント: computer use と同じ要領で、画像を見て 1 手ずつ事前定義コマンドを選ぶ。

各ステップ: 俯瞰画像（格子＋目標点）と手首画像（指先位置）を VLM に渡し、
JSON で次のアクションを受け取り、実行結果の文字列を履歴に積む。done まで繰り返す。
"""
from __future__ import annotations

import json
from typing import Optional

from agent.primitives import Primitives
from perception.vlm import VLMUnavailable

SYSTEM = """あなたは 5 軸ロボットアーム（SO-ARM101）を操作するエージェントです。
毎ステップ 2 枚の画像を見て、次に実行するコマンドを 1 つだけ JSON で返します。

[画像]
1 枚目: 俯瞰カメラ。黄色の格子（列 A〜{cols_last} 左→右、行 1〜{rows} 上→下、各セル中央にラベル）と、
        赤い十字 "target" = 現在の目標点（グリッパの真下の机上の点）。
        緑の枠 "reachable" = アームが届く範囲の目安。目標は必ずこの枠の内側に選ぶ（枠の外は拒否される）。
        枠内でも台座（アームの根元）から遠い左側は届かないことがあり、その場合は「届きません」と返るので別の場所を選ぶ。
2 枚目: 手首カメラ（グリッパに固定。グリッパと一緒に回転・移動する）。赤い十字 "fingertip" = 降りたとき指が閉じる位置。
        画面下の白い部品は自分の指で、指は画像の左右方向に閉じる。

[コマンド（事前定義された動き。上下動・開閉・接触検知はコマンド内で自動）]
- {{"name":"goto_cell","cell":"C4","offset":{{"u":0.5,"v":0.5}}}} : セルへ水平移動（粗い移動）。offset はセル内の位置（左上 0,0〜右下 1,1、省略時は中心）
- {{"name":"move","direction":"left|right|up|down","size":"small|large"}} : 水平微調整。方向は 2 枚目（手首画像）での向き（small≈1cm, large≈3cm）。机の上の物に合わせるとき用
- {{"name":"shift","direction":"left|right|up|down","size":"small|large"}}: 水平微調整。方向は 1 枚目（俯瞰画像）での向き。背の高い容器の上に合わせるとき用
- {{"name":"rotate","degrees":-45|-20|20|45|90}}  : 手首ロールを相対回転（手首画像が同じ向きに回る）。細長い物は長辺が画像の上下方向になるように
- {{"name":"grasp"}}  : その場で降りて掴んで持ち上げる。空振りなら自動で周囲を少しずらして再試行し、結果を報告する
- {{"name":"hover"}}  : 持ったまま目標の真上で 6cm まで垂直に降りる（低い物の上に置く前に、手首画像で合わせるため）
- {{"name":"place"}}  : 机や低い物の上に置く。低トルクで降りて触れたら止まり、放して持ち上げる
- {{"name":"drop"}}   : コップなど容器に入れる。俯瞰画像で把持物を自動追跡して容器の口の真上に合わせ、口の少し上まで低トルクで
                       降りてから開いて落とす（壊れ物に押し付けないため、容器には必ずこちら。hover は使わない）
- {{"name":"done","success":true|false,"reason":"..."}} : タスク完了（または不可能）

[典型手順]
掴む: goto_cell で対象の上へ → 手首画像で十字を対象の中心に合わせる（move）→ 細長い物は rotate で長辺を上下方向に → grasp
      grasp が失敗と報告したら、手首画像を見直して move / rotate してから再度 grasp
置く（机や低い物の上）: 持ったまま goto_cell で置き先の上へ（18cm で運ぶ）→ hover（6cm へ降りる）→ 手首画像で十字を置き先の中心に（move）→ place
      ※ 18cm のままでは手首画像の十字は当てにならない。必ず hover してから合わせる
入れる（コップなど容器）: 手首画像の十字は机の上の物にしか使えない（高い物は斜めに見えてずれる）。
      俯瞰画像だけで判断する: 容器の「底（机に接している部分）」の中心に赤い target 十字が重なるよう
      goto_cell（offset 指定）で合わせる → drop（細かい位置合わせは drop が自動で行う）
[合わせ方の判断基準（重要）]
- 十字が対象の輪郭の内側にあれば「合っている」。完璧を目指さず grasp / release に進む（指は数 cm の幅で閉じる）。
- 容器に入れるときは、俯瞰画像で target 十字が容器の底の輪郭の内側にあれば release。
- 直前の move と逆方向に戻したくなったら十分近い証拠。move せず grasp / place / drop へ。
- 何も持っていないのに place / drop はできない。grasp の結果（把持成功/失敗）を必ず確認してから次へ。
- 「X の隣に置く」= X の縁に接するくらい近く（X から 1 セル以内）の空いた場所。離れた場所は「隣」ではない。
  「左隣/右隣/上/下」は俯瞰画像での方向。
- place / drop の後はアームが自動で退避する。退避後の俯瞰画像で結果を確認してから done を返す
  （置いた物が指示の位置に無い・倒れている・容器に入っていない場合は done ではなく掴み直す）。
- アームが対象に被っていて見えないときは、見えたことにせず、確認できる情報だけで判断する。
- 対象が緑枠の外（特に画像の下端や左端）に落ちた・ある場合は、無理に取りに行かず done（success=false）で理由を報告する。
- 人の手や体が作業領域に見えたら、動かずに done（success=false）で報告する。
1 ステップ 1 コマンド。

[出力形式（JSON のみ）]
{{"observation":"2 枚の画像から分かること（対象と十字の位置関係）","reasoning":"次の一手の理由","action":{{...}},"task_done":false}}
"""


class VLMAgent:
    def __init__(self, vlm, prims: Primitives, max_steps: int = 30, out_dir=None, recorder=None):
        self.vlm, self.prims, self.max_steps, self.out_dir = vlm, prims, max_steps, out_dir
        self.recorder = recorder

    def run(self, instruction: str) -> bool:
        import cv2

        cols, rows = self.prims.grid
        system = SYSTEM.format(cols_last=chr(ord("A") + cols - 1), rows=rows)
        history: list[str] = []
        moves: list[str] = []   # 連続 move の方向（振動検出用）
        for step in range(self.max_steps):
            disp, wrist, raw = self.prims.observe()
            hist = "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(history[-8:])) or "  (なし)"
            hint = ""
            if len(moves) >= 2 and {moves[-1], moves[-2]} in ({"left", "right"}, {"up", "down"}):
                hint = "\n[システムからの注意] 直前の move が往復しています。対象は十分近いので、これ以上 move せず grasp / place / drop に進んでください。"
            elif len(moves) >= 4 and len(set(moves[-4:])) == 1:
                hint = f"\n[システムからの注意] 同じ方向 ({moves[-1]}) の move が 4 回続いています。見え方が変わらないなら判断を見直し、goto_cell で位置を取り直すか、別の方向を試してください。"
            prompt = (system + f"\n[タスク] {instruction}\n[現在の状態] {self.prims.state_text()}\n[直近の履歴]\n{hist}{hint}\n"
                      f"次のコマンドを JSON で返してください。")
            images = [disp] + ([wrist] if wrist is not None else [])
            try:
                res = self.vlm._ask(prompt, images, f"agent{step:02d}")
            except VLMUnavailable as e:
                print(f"[agent] VLM が利用不能のため安全停止: {e}")
                return False
            act = res.get("action") or {}
            name = str(act.get("name", "")).lower()
            key = json.dumps(act, sort_keys=True, ensure_ascii=False)
            if self.recorder is not None:
                self.recorder.set_step(f"step {step}: {json.dumps(act, ensure_ascii=False)}", str(res.get("reasoning", ""))[:200])
                self.recorder.tick(force=True)
            repeats = sum(1 for h in history[-4:] if h.startswith(key))
            if repeats >= 4:
                print(f"[agent] 同一コマンドが 5 回連続 → 安全のため中断: {key}")
                return False
            print(f"[agent {step}] obs: {res.get('observation', '')[:80]}")
            print(f"[agent {step}] -> {json.dumps(act, ensure_ascii=False)}  ({res.get('reasoning', '')[:80]})")
            if name == "done" or res.get("task_done"):
                ok = bool(act.get("success", res.get("task_done", False)))
                print(f"[agent] 終了: success={ok} {act.get('reason', '')}")
                return ok
            fn = {"goto_cell": self.prims.goto_cell, "move": self.prims.move, "shift": self.prims.shift,
                  "rotate": self.prims.rotate, "grasp": self.prims.grasp, "place": self.prims.place, "hover": self.prims.hover,
                  "drop": self.prims.drop, "release": self.prims.release, "home": self.prims.home}.get(name)
            if fn is None:
                result = f"未知のコマンド '{name}'"
            else:
                try:
                    result = fn(**{k: v for k, v in act.items() if k != "name"})
                except VLMUnavailable as e:
                    print(f"[agent] VLM が利用不能のため安全停止: {e}")
                    return False
                except Exception as e:  # noqa: BLE001
                    result = f"実行エラー: {e}"
            print(f"[agent {step}] result: {result}")
            history.append(f"{key} → {result}")
            if self.recorder is not None:
                self.recorder.set_step(f"step {step}: {json.dumps(act, ensure_ascii=False)}", f"result: {result}"[:200])
                self.recorder.tick(force=True)
            if name in ("move", "shift"):
                moves.append(str(act.get("direction", "")).lower())
            else:
                moves.clear()
        print("[agent] ステップ上限に到達")
        return False
