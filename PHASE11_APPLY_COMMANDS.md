# Phase 11 Patch Apply Commands

Run from your ATLC repo root, not from inside `yolo_research`:

```bash
cd "/Users/quockaiser/Library/CloudStorage/GoogleDrive-kaiserquoc@gmail.com/My Drive/atlc"
```

Unzip the patch into repo root:

```bash
unzip -o ~/Downloads/phase11_patch.zip -d .
```

Append Phase 11 gitignore block safely:

```bash
if ! grep -q "Phase 11: YOLO research datasets" .gitignore; then
  cat .gitignore_phase11_append.txt >> .gitignore
fi
rm .gitignore_phase11_append.txt
```

Optional cleanup duplicate folder:

```bash
rm -rf "yolo_research/outputs 2"
```

Compile source files:

```bash
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
```

Check what Git sees:

```bash
git status --short
git check-ignore -v yolo_research/datasets/project_after_merging.yolo26/train/images/09150383_jpg.rf.lbmUge1bgGlqxJiA0W2R.jpg || true
```

Commit source only:

```bash
git add .gitignore
git add yolo_research/README.md
git add yolo_research/__init__.py
git add yolo_research/configs/data.yaml
git add yolo_research/configs/train_config.yaml
git add yolo_research/datasets/README.md
git add yolo_research/notebooks/yolo_training_research.ipynb
git add yolo_research/scripts
git add yolo_research/src
git add yolo_research/outputs/*/.gitkeep

git commit -m "feat: add reproducible YOLO training workflow"
```

Push:

```bash
git push -u origin phase11-custom-yolo-training
```

If `origin` is not configured:

```bash
git remote -v
git remote add origin YOUR_GITHUB_REPO_URL
```
