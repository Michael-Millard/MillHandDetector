"""Quick verification script for exported ONNX hand detector.

Checks:
1. Load ONNX with onnxruntime and run a dummy inference.
2. Load ONNX with OpenCV DNN and run the same dummy inference.
3. Print output tensor shapes for comparison.

Usage:
  python -m src.verify_onnx --model build/yolo11-hand/weights/best.onnx --img 640
Requires: onnxruntime (pip install onnxruntime) and OpenCV built with DNN (already in requirements).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    p.add_argument('--img', type=int, default=640, help='Image size (square)')
    return p.parse_args()


def make_dummy(img: int) -> np.ndarray:
    # Create a random image (simulate RGB)
    return (np.random.rand(img, img, 3) * 255).astype(np.uint8)


def run_onnxruntime(path: str, blob: np.ndarray):
    if ort is None:
        print('onnxruntime not installed; skipping ORT inference')
        return None
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    ort_out = sess.run(out_names, {input_name: blob})
    print('[onnxruntime] outputs:')
    for n, arr in zip(out_names, ort_out):
        print(f'  {n}: shape={arr.shape} dtype={arr.dtype}')
    return ort_out


def run_opencv(path: str, blob: np.ndarray):
    net = cv2.dnn.readNetFromONNX(path)
    net.setInput(blob)
    out = net.forward()
    print('[opencv] single forward output shape:', out.shape)
    return out


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f'Model not found: {model_path}')

    img = make_dummy(args.img)
    # Letterbox not required for square dummy; just scale to [0,1]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(args.img, args.img), swapRB=True, crop=False)
    print('Blob shape:', blob.shape)

    run_onnxruntime(str(model_path), blob)
    run_opencv(str(model_path), blob)

if __name__ == '__main__':
    main()
