# MillHandDetector

Train an Ultralytics YOLO11-based hand detector in Python.

## Dataset

This project uses the "Hand Detection Dataset (VOC/YOLO Format)" by Nouman Ahsan, available on Kaggle: [Hand Detection Dataset](https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format?resource=download). The dataset contains labeled images for training and validation in YOLO format.

## Repository Layout

```
.
├── main.py                   # train/val/predict entrypoint
├── yolo_models/             # directory for YOLO models (e.g., yolo11n.pt, yolo11s.pt)
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

`class_id cx cy w h` in normalized [0,1], where `class_id` is `0` for `hand`.

## Environment Setup

Python 3.9–3.11 recommended. This project was developed using Python 3.11.

```
python3.x -m venv .millhand-venv
source .millhand-venv/bin/activate
pip install --upgrade pip
pip install ultralytics
```

Optional but helpful: `pip install opencv-python tqdm numpy`.

## CLI Usage

### Train (YOLO11)

```
python main.py --train --epochs 50 --img 640 --batch 16 --model yolo_models/yolo11s.pt
```

Artifacts go to `output/yolo11_hand/`. Best weights: `output/yolo11_hand/weights/best.pt`.

### Validate

```
python main.py --val --model output/yolo11_hand/weights/best.pt
```

### Predict

```
python main.py --predict path/to/image_or_dir_or_video.mp4 --model output/yolo11_hand/weights/best.pt --img 640
```

### Export to ONNX for OpenCV DNN (YOLO11)

Recommended defaults for OpenCV DNN: opset 12 or 13, static input size, FP32.

```
python main.py --export --img 640 --opset 12 --model output/yolo11_hand/weights/best.pt
```

This will export from `output/yolo11_hand/weights/best.pt` if present, otherwise from `yolo_models/yolo11s.pt`. Output is typically `best.onnx` in the same folder as the weights.

Notes:
- Keep `--dynamic` off for static shapes; OpenCV handles static best.
- Keep `--half` off (FP32) unless your OpenCV build supports FP16.
- You can try `--simplify` if you have `onnxsim` installed.

### Using the ONNX in OpenCV (C++)

Ultralytics YOLOv11 ONNX exports commonly output detections as `[1, 5, N]` for single-class models: `[x, y, w, h, conf]`. Inspect the model output once to confirm.

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
cv::Mat out = net.forward(); // shape: [1, 5, N]

// Postprocess (example for [x, y, w, h, conf])
std::vector<cv::Rect> boxes;
std::vector<float> confidences;
for (int i = 0; i < out.size[2]; ++i) {
    float* data = out.ptr<float>(0, 0, i);
    float conf = data[4];
    if (conf > 0.3) { // confidence threshold
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        boxes.emplace_back(cv::Rect(x - w/2, y - h/2, w, h));
        confidences.push_back(conf);
    }
}
```

This README provides a complete overview of the dataset, CLI, and deployment workflow.
