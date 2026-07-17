# Linux Mint: Cross-Platform Benchmarking Workflow

## Overview

This folder contains the platform-specific scripts and execution workflow used to run the MediaPipe-based knee-rehabilitation benchmarking pipeline on **Linux Mint**.

The Linux Mint workflow supports both:

* **Pre-recorded video processing**
* **Real-time camera feed processing**

and contributes to the cross-platform feasibility study by enabling comparison with **Windows** and **NVIDIA Jetson Orin NX** under GUI and CLI execution conditions.

## Environment Requirements

The Linux Mint workflow assumes:

* Anaconda or Miniconda runtime environment
* MediaPipe
* OpenCV
* Matplotlib
* NumPy
* Pandas

Install required packages after activating the relevant conda environment.

## Pre-recorded Video Processing

Run the pre-recorded video workflow using:

```bash id="dzjrk3"
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

```bash id="hchecx"
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

The Linux Mint scripts include:

* MediaPipe Pose estimation
* Custom knee-angle computation
* CSV export
* Benchmarking and timing logging

## Output

Depending on the script and execution mode, outputs may include:

* CSV benchmarking logs
* Knee-angle computation outputs
* Processed video or live-feed results
* Analysis-ready timing data for cross-platform comparison

## Role in the Benchmark Study

Within the broader benchmark study, the Linux Mint workflow is used to assess:

* Performance of a repurposed general-purpose platform
* Pre-recorded and live processing behaviour
* GUI versus CLI execution overhead
* Cross-platform timing differences relative to Windows and NVIDIA Jetson

## Related Folders

* **Data/** – summary benchmarking datasets and analysis scripts
* **Jetson/** – platform-specific benchmarking workflow for NVIDIA Jetson Orin NX
* **Windows/** – platform-specific benchmarking workflow for Windows
