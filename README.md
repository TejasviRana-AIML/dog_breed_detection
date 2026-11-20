# Dog Breed Detection and Tracking

Deep learning project using YOLOv8m, ByteTrack and fine-tuned ResNet50 for dog detection, tracking and breed classification across 120 breeds.

## Technologies
- Python, PyTorch, Torchvision
- ResNet50, YOLOv8m, ByteTrack
- OpenCV, Pandas

## Features
- Detects dogs in videos.
- Tracks dogs across frames.
- Predicts dog breeds.
- Generates annotated output videos.

## Dataset
Kaggle Dog Breed Identification dataset (120 breeds).
https://www.kaggle.com/c/dog-breed-identification

## Setup
Install dependencies: `pip install torch torchvision pandas pillow opencv-python ultralytics lap`
Place model weights and dataset labels in the paths expected by run_video.py.

## Run
Update video paths in run_video.py and execute: `python run_video.py`

## Testing
Six videos tested: five produced retained dog tracks; one produced no retained tracks. These are pipeline observations, not benchmark metrics.

## Note
Dataset, model weights, virtual environment and generated videos are excluded from this repository.
