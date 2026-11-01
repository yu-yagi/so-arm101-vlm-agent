#!/usr/bin/env bash
# SO-ARM101 フォロワーの LeRobot キャリブレーション（対話式・初回のみ）
# 保存先: ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json
set -e
cd "$(dirname "$0")/.."
PORT=$(grep -E '^\s*port:' config.yaml | awk '{print $2}')
exec .venv/bin/lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$PORT" \
  --robot.id=so101_follower
