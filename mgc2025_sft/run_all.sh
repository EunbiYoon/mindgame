#!/usr/bin/env bash
set -e

export MGC_SCRIPT_DIR="${MGC_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=mgc2025_sft/lib.sh
source "${MGC_SCRIPT_DIR}/lib.sh"

maf_nfs_safe_exec "$@"
mgc_setup_root
mgc_new_run_id all
maf_train_lora_extra_args

echo "=== MGC2025: convert all games (max_trajectories=${MGC_MAX_TRAJ}) ==="
mgc_convert_all

for game in mafia blotto ipd codenames; do
  train="${MGC_DATA_DIR}/${game}_train.jsonl"
  test="${MGC_DATA_DIR}/${game}_test.jsonl"
  if [[ ! -s "${train}" ]]; then
    echo "Skip ${game}: empty train file ${train}" >&2
    continue
  fi

  echo
  echo "=== MGC2025: train + eval ${game} (RUN_ID=${RUN_ID}) ==="
  rid="${RUN_ID}_${game}"

  python eunbi/lora/train_lora.py \
    --train_file "${train}" \
    --run_id "${rid}" \
    "${TRAIN_LORA_EXTRA[@]}" \
    --epochs 1

  python mgc2025_sft/evaluate.py \
    --game "${game}" \
    --run_id "${rid}" \
    --model_dir "eunbi/lora/runs/${rid}" \
    --test_file "${test}" \
    --n "${MGC_EVAL_N}"

  if [[ "${game}" == "blotto" ]]; then
    python eunbi/eval/evaluate_blotto.py \
      --run_id "${rid}" \
      --model_dir "eunbi/lora/runs/${rid}" \
      --n_games 30
  fi
done

python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

info = {
    "run_id": "${RUN_ID}",
    "data_source": "mgc2025",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mgc_data_dir": "${MGC_DATA_DIR}",
    "max_trajectories": int("${MGC_MAX_TRAJ}"),
    "games": ["mafia", "blotto", "ipd", "codenames"],
    "per_game_run_ids": {
        g: f"${RUN_ID}_{g}" for g in ("mafia", "blotto", "ipd", "codenames")
    },
}
Path("${MGC_DATA_DIR}/run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "MGC2025 full pipeline complete."
echo "  data:  ${MGC_DATA_DIR}"
echo "  lora:  eunbi/lora/runs/${RUN_ID}_<game>"
echo "  eval:  eunbi/eval/runs/${RUN_ID}_<game>"
