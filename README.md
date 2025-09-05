# MillHandDetector

Train an Ultralytics YOLO11-based hand detector in Python.

## Dataset options
TODO: Reference dataset from: https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format?resource=download

TODO: Fix project structure below 

## Repository layout

```
.
├── main.py                   # train/val/predict entrypoint
├── yolo11n.pt                # base model
├── config/
│   └── hand.yaml             # dataset config (points to ./data)
└── data/
	 ├── images/
	 │   ├── train/
	 │   └── val/
	 └── labels/
		  ├── train/
		  └── val/
```

Each image must have a same-named `*.txt` label file in `labels/...` with YOLO format lines:

`class_id cx cy w h` in normalized [0,1], where class_id is `0` for `hand`.

## Environment setup

Python 3.9–3.11 recommended. I used Python3.11

```
python3.x -m venv .millhand-venv
source .millhand-venv/bin/activate
pip install --upgrade pip
pip install ultralytics
```

Optional but helpful: `pip install opencv-python tqdm numpy`.

## Train (YOLO11)

```
python main.py --train --epochs 50 --img 640 --batch 16 --model yolo11n.pt
```

Artifacts go to `build/yolo11-hand/`. Best weights: `build/yolo11-hand/weights/best.pt`.

## Validate

```
python main.py --val --model build/yolo11-hand/weights/best.pt
```

## Predict

```
python main.py --predict path/to/image_or_dir_or_video.mp4 --model build/yolo11-hand/weights/best.pt --img 640
```

## Export to ONNX for OpenCV DNN (YOLO11)

Recommended defaults for OpenCV DNN: opset 12 or 13, static input size, FP32.

```
python main.py --export --img 640 --opset 12 --model build/yolo11-hand/weights/best.pt
```

This will export from `build/yolo11-hand/weights/best.pt` if present, otherwise from `yolo11n.pt`. Output is typically `best.onnx` in the same folder as the weights.

Notes:
- Keep `--dynamic` off for static shapes; OpenCV handles static best.
- Keep `--half` off (FP32) unless your OpenCV build supports FP16.
- You can try `--simplify` if you have `onnxsim` installed.

### Using the ONNX in OpenCV (C++)

Ultralytics YOLOv8 ONNX exports commonly output detections as `[num, 84]` for COCO; for a single class, expect `[num, 6]` or `[num, 7]` depending on export version: `[x, y, w, h, conf, class_conf (optional), class_id (optional)]`. If the model exports in the newer format, you may get a 1xN x(5+C) tensor. Inspect the model output once to confirm.

Sketch:

```cpp
#include <opencv2/opencv.hpp>

cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");

// Preprocess
cv::Mat img = cv::imread("image.jpg");
int input = 640; // must match export imgsz if static
cv::Mat blob = cv::dnn::blobFromImage(img, 1/255.0, cv::Size(input, input), cv::Scalar(), true, false);
net.setInput(blob);

// Forward
cv::Mat out = net.forward(); // shape varies; e.g., [1, N, 6]

// Postprocess (example for [x, y, w, h, conf, class])
std::vector<int> classIds;
std::vector<float> confidences;
std::vector<cv::Rect> boxes;
float confThreshold = 0.25f, nmsThreshold = 0.45f;

for (int i = 0; i < out.size[1]; ++i) {
	float* data = out.ptr<float>(0, i);
	float x = data[0];
	float y = data[1];
	float w = data[2];
	float h = data[3];
	float conf = data[4];
	if (conf < confThreshold) continue;
	int cls = 0; // single class

	// Convert xywh (center) to xyxy in original image scale
	int cx = static_cast<int>(x * img.cols);
	int cy = static_cast<int>(y * img.rows);
	int bw = static_cast<int>(w * img.cols);
	int bh = static_cast<int>(h * img.rows);
	int left = cx - bw / 2;
	int top = cy - bh / 2;

	classIds.push_back(cls);
	confidences.push_back(conf);
	boxes.emplace_back(left, top, bw, bh);
}

// NMS
std::vector<int> indices;
cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
for (int idx : indices) {
	cv::rectangle(img, boxes[idx], cv::Scalar(0, 255, 0), 2);
}

cv::imwrite("result.jpg", img);
```

If your export yields a different output layout (e.g., a 84-dim row per box with per-class confidences), adjust indices accordingly. You can quickly print the output shape using a short Python script with `onnxruntime` to confirm.

## Converting annotations to YOLO

If your source dataset is in Pascal VOC (XML) or COCO (JSON), use tools like:

- Roboflow export to YOLO
- CVAT export to YOLO
- Python converters (e.g., fiftyone or custom scripts)

Checklist for correct YOLO labels:

- One `*.txt` per image in `labels/...`.
- Only one class: `0` (hand).
- Boxes normalized by image width/height.
- Train/val splits balanced across scenes, lighting, skin tones, gloves, occlusions.

## Tips

- Start with `yolo11n.pt` for speed; switch to `yolo11s.pt`/`m.pt` if accuracy lags.
- Increase epochs to 100–300 once the pipeline works.
- Use mosaic/augmentations (Ultralytics enables sensible defaults).
- If hands are small, try `--img 896` or `--img 1024` and smaller strides (larger models) for better small-object recall.
