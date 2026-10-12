#!/usr/bin/env bash
# 开发集上的单组件实验：基线 + 一个组件。用法：bash scripts/run_addone.sh [tasks_file]
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:-data/dev_ids.json}"

for only in guard ledger monitor audit finish contract; do
  python3 runner/run.py --run-name "dev100_only_${only}" --tasks "$TASKS" --workers 16 --plugin "$only" --plugin-file pi-shop/plugin_v0.ts --timeout 900 | tail -1
  python3 eval/evaluate.py "runs/dev100_only_${only}" > /dev/null
done
echo "add-one done"
