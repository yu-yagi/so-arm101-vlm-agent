# SO-ARM101 × VLM: 自然言語指示によるロボットアームのゼロショット操作

Physical AI 応用1 最終課題のコード一式。自然言語の指示（例:「青い消しゴムをコップに入れて」）を、
事前教示なしに SO-ARM101 実機で実行する。VLM（OpenAI GPT-6 Astra）が俯瞰カメラと手首カメラの画像を見て、
**事前定義コマンド（プリミティブ）を 1 手ずつ選ぶ** computer use 方式のエージェントである。

> 実行ログ（画像・JSON・録画）、提出動画、レポートはこのリポジトリには含めない（.gitignore）。

## 仕組み

```
指示文 + 俯瞰画像（格子 A1〜J7 + 目標点の十字 + 到達範囲の枠）+ 手首画像（指が閉じる位置の十字）
   │
   ▼
GPT-6 Astra（Responses API）→ {"observation": ..., "reasoning": ..., "action": {"name": "grasp"}}
   │
   ▼
事前定義コマンド: goto_cell / move / shift / rotate / grasp / hover / place / drop / done
   （降下・開閉・接触検知・把持確認・容器の口への閉ループ位置合わせ・退避はコマンド内で自動）
   │
   ▼
画素 → 関節角（教示 9 点の補間 + 極座標 IK モデル + 平面 FK）→ LeRobot → Feetech STS3215
   │
   └── 実行結果の文字列と新しい 2 画像を履歴に積み、done まで繰り返す
```

コアアイデア
1. VLM に全てをやらせず、事前定義の動きを選ばせる。VLM は意味理解と画像の見比べは得意だが距離や高さの数値は苦手なので、
   仕事を「2 枚の画像を見て次の一手を選ぶ」ことに限定し、危険になりうる部分はコマンド側で自動化する。
2. 3 次元の作業を 2 次元の格子で表現する。俯瞰画像にラベル付き格子を描き（set-of-marks）、VLM はセルとセル内位置で場所を答える。
   画素と関節角は机上 9 点の教示で対応づけ、高さは測らず低トルクで降りて接触で止まる。最後の位置合わせは手首画像の十字で行う。

安全設計（陶器のコップがある環境）
- 降下中は肩・肘・手首のトルク上限を 25% に下げ、負荷または追従誤差で接触を検知したら即退避
- 容器には降りずに口の上で開く drop。俯瞰画像で把持物を色追跡して口の真上に合わせ、口の少し上まで低トルクで降りてから開放
- 横移動は 18 cm の運搬高さで行い、目標点は教示範囲 + 余白の柵内に限定。到達不可は拒否して VLM に理由を返す
- place / drop 後は自動で退避し、アームが映っていない画像で VLM が結果を検証。人の手が見えたら停止（done, success=false）

## 構成

```
main.py                   エントリ（--dry-run で実機・カメラ・API なしの配線確認、--interactive で連続実行）
config.yaml               ポート・カメラ・IK・VLM・エージェントの設定
agent/vlm_agent.py        VLM エージェント（システムプロンプト、1 手ずつのループ、安全停止）
agent/primitives.py       事前定義コマンドの実装（移動・把持・配置・投入・退避、到達判定、閉ループ位置合わせ）
robot/so101.py            LeRobot ラッパー + Mock（トルク制限、接触検知付き降下、録画フック）
robot/kinematics.py       GridIK（教示点補間）+ PolarIK（教示範囲外: 極座標モデル + 平面 FK の逆解）
robot/fk.py               教示データから当てはめた平面順運動学（高さ指定の姿勢を解く）
perception/camera.py      カメラ（黒塗りセルは録画・プレビューのみ。VLM には生画像）
perception/vlm.py         GPT API 呼び出し（Responses API / chat）、格子描画、mock
perception/detector.py    OWLv2（把持後に「元の位置から物が消えたか」の確認、手首カメラ校正の物体検出）
perception/recorder.py    実行中の録画（俯瞰 + 手首 + VLM の判断）
perception/workspace.py   画素 ⇄ 作業座標（本システムでは単位行列）
scripts/                  校正・教示・プレビュー・提出動画生成（下記）
calib/                    校正結果: grid_points.json（教示 9 点）, fk_fit.json, wrist_jacobian.json, homography.npy
```

## セットアップ

```bash
# Python 3.12 の venv（LeRobot が 3.14 未対応）
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # LeRobot 0.6.1, OpenAI SDK, OpenCV, transformers (OWLv2), imageio-ffmpeg
cp .env.example .env                       # OPENAI_API_KEY を記入
lerobot-find-port                          # アームのポート → config.yaml の robot.port
bash scripts/calibrate.sh                  # 初回のみ LeRobot の対話キャリブレーション
python main.py --dry-run "青い消しゴムをコップに入れて"   # ハードなしで一巡（Mock）
```

## 実機での再現手順

1. 俯瞰カメラと手首カメラを固定し、`python scripts/preview_cam.py`（`--wrist` で手首）で画角とピントを確認。
   config.yaml の `camera.blur_cells` に、録画で黒塗りにするセル（作業領域外の映り込み）を指定する。
2. `python scripts/teach_pixels.py` で教示。手でアームを机上の点に置く → 俯瞰画像で先端をクリック → 持ち上げる、を 9 点。
   `calib/grid_points.json` と単位行列の `calib/homography.npy` ができる。
3. `python scripts/fit_fk.py` で平面 FK のオフセットを当てはめる（`calib/fk_fit.json`）。
4. `python scripts/calibrate_wrist.py` で手首カメラの指先位置とヤコビアンを校正（`calib/wrist_jacobian.json`、config の `grasp.tool_point`）。
5. `python scripts/goto_click.py --hover-only` でクリック位置に 6 cm まで降りる精度確認（教示範囲外は極座標モデル）。
6. 実行: `python main.py "青い消しゴムをカレンダーの上に乗せて"` / `python main.py "青い消しゴムをコップに入れて"`。
   `runs/<日時>_run/` に VLM への入力画像・応答 JSON・録画 mp4 が残る。
7. 提出動画: `scripts/make_submission_video.py` の `SEGMENTS` に使う run 名を書き、`python scripts/make_submission_video.py`。
   録画から映像部分を切り出し、各ステップの VLM の出力を字幕として描き直して結合する（講座の OP/ED は `../video/assets/` にあれば前後に付く）。

注意: 実機を動かすときは作業領域に手を入れない。接触検知は低トルクだが、停止までに 1 ステップ（数秒）かかる。

## 設計と実装について

設計と実装は Claude（Anthropic の AI、Claude Code）と議論しながら進めた。
特に「3 次元の作業を 2 次元の格子で表現する」考え方は、その議論の中で得られたものである。
