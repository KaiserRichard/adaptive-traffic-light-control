#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .git ]; then
  echo "[ERROR] Run this script from the ATLC repo root."
  exit 1
fi

if [ ! -d yolo_research ]; then
  echo "[INFO] yolo_research does not exist yet; it will be created by copied files."
fi

if [ -f .gitignore_phase11_append.txt ]; then
  if ! grep -q "Phase 11: YOLO research datasets" .gitignore 2>/dev/null; then
    cat .gitignore_phase11_append.txt >> .gitignore
    echo "[OK] Appended Phase 11 block to .gitignore"
  else
    echo "[OK] .gitignore already has Phase 11 block"
  fi
  rm .gitignore_phase11_append.txt
fi

if [ -d "yolo_research/outputs 2" ]; then
  echo "[INFO] Removing duplicate folder: yolo_research/outputs 2"
  rm -rf "yolo_research/outputs 2"
fi

python -m py_compile yolo_research/src/yolo_utils/yolo_io.py
python -m py_compile yolo_research/src/yolo_utils/dataset_check.py
python -m py_compile yolo_research/src/yolo_utils/metrics.py
python -m py_compile yolo_research/src/yolo_utils/plotting.py
python -m py_compile yolo_research/scripts/check_dataset.py
python -m py_compile yolo_research/scripts/train_yolo.py
python -m py_compile yolo_research/scripts/evaluate_yolo.py
python -m py_compile yolo_research/scripts/predict_yolo.py
python -m py_compile yolo_research/scripts/export_model.py
python -m py_compile yolo_research/scripts/generate_plots.py

echo "[OK] Phase 11 source files compile successfully."
echo "Next: run git status --short, then commit/push using PHASE11_APPLY_COMMANDS.md."
