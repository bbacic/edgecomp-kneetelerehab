# NVIDIA Jetson Orin NX: Cross-Platform Benchmarking Workflow

## Overview

This folder contains the platform-specific scripts and setup used to run the MediaPipe-based knee-rehabilitation benchmarking workflow on the **NVIDIA Jetson Orin NX**.

The Jetson workflow supports both:

* **Pre-recorded video processing**
* **Real-time CSI camerafeed processing**

and is used to evaluate the feasibility of Jetson as an edge platform for near-real-time, privacy-preserving rehabilitation analytics.

## Contents

This folder includes scripts and configurations for:

* Pre-recorded video analysis on Jetson
* Real-time CSI camera capture and processing
* GUI and headless execution
* Benchmarking and performance logging

## Environment Setup

Install the required dependencies:

```bash
pip install --upgrade pip
pip install opencv-python matplotlib pyyaml
pip install mediapipe
```

Create and activate a virtual environment:

```bash
python3 -m venv knee_rehab
source knee_rehab/bin/activate
```

## Pre-recorded Video Processing

Run the pre-recorded video workflow using:

```bash
python main.py \
  /path/to/video/IMG_1142.mp4 \
  /path/to/output/IMG_1142.csv \
  --export_knee right \
  --direction right
```

### Notes

* This workflow processes stored MP4 video files.
* Output includes CSV data for downstream benchmarking and analysis.
* Pre-recorded video benchmarking is primarily used to assess per-frame timing behaviour across execution modes and platforms.

## Real-time Camerafeed Processing (Jetson CSI Camera)

### Environment Requirements

The Jetson camerafeed workflow relies on:

* GStreamer with NVMM buffers for hardware-accelerated camera capture
* CSI camera access through NVIDIA-supported pipelines
* MediaPipe Pose estimation
* custom knee-angle computation
* benchmarking and performance logging

### Important Notes

* MediaPipe should be imported after `cap.isOpened()` if required to avoid TLS-related conflicts.
* Real-time performance on Jetson is optimised for CSI camera input using NVMM buffers.
* Compared with live CSI capture, pre-recorded video decoding may impose additional CPU overhead on Jetson.

## Running in GUI Mode

```bash
python3 main_csi_camerafeed.py \
  --export_knee both \
  --direction forward
```

## Running in Headless Mode

```bash
python3 main_csi_camerafeed.py \
  --export_knee both \
  --direction forward \
  --headless
```

## Execution Modes

* **GUI mode** enables live visualisation and plotting during execution.
* **Headless mode** disables GUI interaction and is recommended for benchmarking, long-duration runs, and reduced visualisation overhead.

## Output

Depending on the script and execution mode, outputs may include:

* CSV benchmarking logs
* Knee-angle computation results
* Live or replayed pose visualisation
* Timing metrics for downstream analysis

## Role in the Benchmark Study

Within the broader cross-platform feasibility study, the Jetson workflow is used to assess:

* Live edge deployment performance
* CSI camera-based rehabilitation monitoring
* GUI versus headless execution overhead
* Platform suitability for compact, privacy-preserving, near-real-time rehabilitation analytics

## Related Folders

* **Data/** – summary benchmarking datasets and analysis scripts
* **Linux Mint/** – platform-specific benchmarking workflow for Linux Mint
* **Windows/** – platform-specific benchmarking workflow for Windows
