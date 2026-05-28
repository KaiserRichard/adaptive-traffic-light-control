# ATLC Phase 11 — Custom YOLO Training

This folder contains the reproducible YOLO training workflow for the ATLC project.

```text
pc_app/        = runtime ATLC application
yolo_research/ = dataset validation, YOLO training, evaluation, plots, reports
```

## Classes

```text
0 car
1 motorbike
2 truck
3 bus
```

## Typical workflow

```bash
python -m yolo_research.scripts.check_dataset --data yolo_research/configs/data.yaml
python -m yolo_research.scripts.train_yolo --data yolo_research/configs/data.yaml --model yolo26n.pt --epochs 50 --imgsz 640 --batch 16 --device 0 --project yolo_research/outputs/runs --name atlc_yolo26n_custom --exist-ok
python -m yolo_research.scripts.evaluate_yolo --weights yolo_research/outputs/runs/atlc_yolo26n_custom/weights/best.pt --data yolo_research/configs/data.yaml --project yolo_research/outputs/evaluation --name atlc_yolo26n_custom_val --exist-ok
python -m yolo_research.scripts.generate_plots --data yolo_research/configs/data.yaml --run-dir yolo_research/outputs/runs/atlc_yolo26n_custom --outdir yolo_research/outputs/figures
python -m yolo_research.scripts.export_model --overwrite
```

Do not commit datasets, training runs, prediction folders, or `.pt` weights.
