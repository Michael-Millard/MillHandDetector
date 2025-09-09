"""
Minimal Ultralytics YOLO (YOLO11) training entrypoint for hand detection.

Usage (after installing requirements):
	python main.py --train --epochs 100 --img 640 --batch 16
	python main.py --val
	python main.py --predict path/to/test/image/or/video --model output/yolo11_hand/weights/best.pt --img 640
	python main.py --export --img 640 --opset 12

Expects dataset layout under ./data as per config/hand.yaml.
My dataset was obtained from: https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format?resource=download
"""

from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Ultralytics YOLO11 Hand Detector Trainer")
	m = p.add_mutually_exclusive_group(required=True)
	m.add_argument("--train", action="store_true", help="Run training")
	m.add_argument("--val", action="store_true", help="Run validation on best.pt")
	m.add_argument("--predict", type=str, metavar="SOURCE", help="Run prediction on an image/video/dir")
	m.add_argument("--export", action="store_true", help="Export weights to ONNX for OpenCV DNN")

	p.add_argument("--model", type=str, default="yolo_models/yolo11s.pt", help="Base model or checkpoint path (default: yolo_models/yolo11s.pt)")
	p.add_argument("--data", type=str, default="config/hand.yaml", help="Dataset YAML path")
	p.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
	p.add_argument("--img", type=int, default=640, help="Image size")
	p.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
	p.add_argument("--device", type=str, default="0", help="CUDA device e.g. '0' or '0,1' or 'cpu'")
	p.add_argument("--project", type=str, default="output", help="Project output directory (default: output)")
	p.add_argument("--name", type=str, default="yolo11_hand", help="Output directory (default: yolo11_hand)")
	p.add_argument("--workers", type=int, default=8, help="Number of data loading workers (default: 8)")
	p.add_argument("--exist-ok", action="store_true", help="Overwrite existing run dir")

	# Export-specific args
	p.add_argument("--opset", type=int, default=12, help="ONNX opset for export (12/13 recommended)")
	p.add_argument("--dynamic", action="store_true", help="Use dynamic input shapes (static is safest for OpenCV)")
	p.add_argument("--half", action="store_true", help="Export FP16 (OpenCV prefers FP32; leave off unless sure)")
	p.add_argument("--simplify", action="store_true", help="Run onnxsim simplify after export")
	return p.parse_args()


def _ensure_paths(args: argparse.Namespace) -> None:
	for k in ("model", "data"):
		p = Path(getattr(args, k))
		if not p.exists():
			raise FileNotFoundError(f"{k} not found: {p}")
	Path(args.project).mkdir(parents=True, exist_ok=True)


def main() -> None:
	args = parse_args()
	_ensure_paths(args)

	try:
		from ultralytics import YOLO
	except Exception as e:
		raise SystemExit(
			"Ultralytics YOLO not installed. Install with: pip install ultralytics"
		) from e

	model_path_abs = Path(args.model).absolute()
	print(f"Using model: {model_path_abs}")
	model = YOLO(model_path_abs)

	if args.train:
		model.train(
			data=args.data,
			epochs=args.epochs,
			imgsz=args.img,
			batch=args.batch,
			device=args.device or None,
			project=args.project,
			name=args.name,
			workers=args.workers,
			exist_ok=args.exist_ok,
		)
		# Validate best checkpoint
		best = Path(args.project) / args.name / "weights" / "best.pt"
		if best.exists():
			YOLO(str(best)).val(data=args.data, imgsz=args.img, device=args.device or None)
		return

	if args.val:
		# If a trained run exists, prefer its best.pt
		candidate = Path(args.project) / args.name / "weights" / "best.pt"
		ckpt = str(candidate if candidate.exists() else args.model)
		YOLO(ckpt).val(data=args.data, imgsz=args.img, device=args.device or None)
		return

	if args.predict:
		# If a trained run exists, prefer its best.pt
		candidate = Path(args.project) / args.name / "weights" / "best.pt"
		ckpt = str(candidate if candidate.exists() else args.model)
		results = YOLO(ckpt).predict(
			source=args.predict,
			imgsz=args.img,
			device=args.device or None,
			project=args.project,
			name=f"{args.name}-predict",
			exist_ok=True,
			save=True,
		)
		# Print path to predictions dir
		if results and hasattr(results[0], "save_dir"):
			print(results[0].save_dir)
		return

	if args.export:
		# Prefer best.pt if available
		candidate = Path(args.project) / args.name / "weights" / "best.pt"
		ckpt = str(candidate if candidate.exists() else args.model)
		exported = YOLO(ckpt).export(
			format="onnx",
			imgsz=args.img,
			opset=args.opset,
			dynamic=args.dynamic,
			half=args.half,
			simplify=args.simplify,
			device=args.device or None,
		)
		# ultralytics returns output path(s) as list or str; print it
		print(exported)
		return


if __name__ == "__main__":
	main()

