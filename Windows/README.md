# Windows Folder: Cross-Platform Benchmarking Workflow

## Overview

This folder contains the platform-specific scripts and execution workflow used to run the MediaPipe-based knee-rehabilitation benchmarking pipeline on **Windows**.

The Windows workflow supports both:

* **pre-recorded video processing**
* **real-time camera feed processing**

and contributes to the cross-platform feasibility study by enabling comparison with **Linux Mint** and **NVIDIA Jetson Orin NX** under GUI and CLI execution conditions.

## Environment Requirements

The Windows workflow assumes:

* Anaconda or Miniconda runtime environment
* MediaPipe
* OpenCV
* Matplotlib
* NumPy
* Pandas

Install required packages after activating the relevant conda environment.

## Pre-recorded Video Processing

Run the pre-recorded video workflow using:

```bash id="7o42ha"
python main.py \
  /path/to/video/IMG_1142.mp4 \
  /path/to/output/IMG_1142.csv \
  --export_knee right \
  --direction right
```

### Notes

* This workflow processes stored video files for offline benchmarking.
* Output includes CSV data for downstream timing analysis and cross-platform comparison.
* Pre-recorded video experiments are primarily used to evaluate per-frame timing behaviour under different execution modes.

## Real-time Camera Feed Processing

Run the live camera workflow using:

```bash id="7cs8jx"
python main.py \
  /path/to/video/IMG_1142.mp4 \
  /path/to/output/IMG_1142.csv \
  --export_knee right \
  --direction right \
  --camera
```

### Notes

* This workflow enables real-time benchmarking using a connected camera source.
* Real-time experiments are used to assess live processing behaviour and near-real-time feasibility.
* Depending on the setup, GUI and CLI execution can be compared for timing and overhead analysis.

## Pipeline Components

The Windows scripts include:

* MediaPipe Pose estimation
* custom knee-angle computation
* CSV export
* benchmarking and timing logging

## Output

Depending on the script and execution mode, outputs may include:

* CSV benchmarking logs
* knee-angle computation outputs
* processed video or live-feed results
* analysis-ready timing data for cross-platform comparison

## Role in the Benchmark Study

Within the broader benchmark study, the Windows workflow is used to assess:

* performance of a general-purpose desktop platform
* pre-recorded and live processing behaviour
* GUI versus CLI execution overhead
* cross-platform timing differences relative to Linux Mint and NVIDIA Jetson

## Related Folders

* **Data/** – summary benchmarking datasets and analysis scripts
* **Jetson/** – platform-specific benchmarking workflow for NVIDIA Jetson Orin NX
* **Linux Mint/** – platform-specific benchmarking workflow for Linux Mint
