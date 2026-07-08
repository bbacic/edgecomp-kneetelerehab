# Cross-Platform Benchmarking for Edge-Based Knee Rehabilitation Analytics

<!-- # edgecomp-kneetelerehab -->

## Overview

This repository contains the code and experimental workflow used for a cross-platform benchmarking study of a MediaPipe-based knee rehabilitation analysis pipeline. The study evaluates the feasibility of deploying Human Activity Recognition (HAR) and pose-estimation-based movement analytics on the NVIDIA Jetson Orin NX, with comparative benchmarking against Windows and Linux platforms.

The benchmark focuses on rehabilitation-oriented movement analysis under two operating conditions:

* **Phase I:** pre-recorded video processing
* **Phase II:** live camerafeed processing

Each phase is evaluated under both:

* **GUI mode**
* **CLI mode**

The benchmark is designed to quantify timing behaviour, identify system-level bottlenecks, and assess deployment trade-offs for privacy-preserving, near-real-time telerehabilitation and rehabilitation monitoring applications.

## Study Aim

The purpose of this benchmark is to evaluate whether a pose-estimation-based knee rehabilitation pipeline can be deployed effectively on compact edge hardware, while also comparing its behaviour against general-purpose computing platforms.

More specifically, the study examines:

* cross-platform timing behaviour
* GUI versus CLI execution overhead
* live versus pre-recorded input conditions
* platform suitability for near-real-time edge deployment
* system bottlenecks related to read time, inference time, and non-inference overhead

## Benchmark Scope

The benchmark does **not** evaluate clinical validity, diagnostic accuracy, or patient outcomes. Its focus is strictly on **computational feasibility**, **system performance**, and **deployment suitability**.

## Pipeline Summary

The benchmarking workflow is based on a MediaPipe Pose pipeline for knee rehabilitation movement analysis. The core stages include:

1. video or camerafeed acquisition
2. frame preprocessing
3. pose landmark inference using MediaPipe Pose
4. knee-angle and movement-related kinematic computation
5. optional GUI rendering and visual overlays
6. logging of timing and execution metadata

The extracted benchmarking parameters include:

* total processing time
* number of processed frames
* nominal input FPS
* average frame time (AFT)
* average read time (ART)
* average MediaPipe inference time (AMT)
* average other time (AOT)
* effective FPS
* slowdown factor
* execution metadata

## Experimental Platforms

The benchmark was conducted across the following platforms:

* **Windows 11**
* **Linux Mint**
* **NVIDIA Jetson Orin NX**

Windows and Linux experiments were executed on older laptop hardware to evaluate deployment trade-offs associated with repurposed general-purpose systems. Jetson experiments were conducted as the target embedded edge platform for live, privacy-preserving rehabilitation analytics.

## Features

* Cross-platform benchmarking of a rehabilitation movement analysis pipeline
* Support for both pre-recorded and live camerafeed processing
* GUI and CLI execution benchmarking
* MediaPipe-based pose landmark extraction
* Knee-angle and movement-related kinematic analysis
* CSV export of benchmarking metrics
* Timing analysis for AFT, ART, AMT, and AOT
* Effective FPS and slowdown factor calculation for live processing experiments

## Repository Purpose

This repository is intended to support:

* reproducible benchmarking of edge-based rehabilitation analytics
* comparative analysis of embedded and desktop-class deployment environments
* future research on privacy-preserving, locally governed telerehabilitation
* extension toward multimodal and real-time rehabilitation monitoring systems

## Requirements

* Python 3.8+
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Matplotlib

Depending on the platform, additional dependencies may be required for:

* webcam or CSI camera access
* GStreamer integration
* platform-specific OpenCV builds

## Installation

Install required Python packages using:

```bash id="tuww79"
pip install -r requirements.txt
```

If dependencies are being installed manually, ensure that MediaPipe, OpenCV, NumPy, Pandas, and Matplotlib are available in the environment.

## Usage

Run the benchmarking workflow from terminal or command prompt using the relevant script for:

* pre-recorded video processing
* live camerafeed processing
* GUI mode execution
* CLI mode execution

A generic execution structure is:

```bash id="8x0nux"
python main.py [input_source] [output_csv] [execution_mode]
```

Where:

* `input_source` may refer to a video file or live camera stream
* `output_csv` is the path for exported benchmarking data
* `execution_mode` specifies GUI or CLI execution

Adapt script arguments as required for your local environment and hardware configuration.

## Output

The benchmark can generate:

* CSV files containing timing and execution metadata
* augmented or visualised processing output in GUI mode
* cross-platform timing summaries
* analysis-ready benchmarking results for figures and tables

## Notes on Interpretation

* Pre-recorded videos in the benchmark were recorded at nominal frame rates of **59.94 fps** and **29.97 fps**.
* Accordingly, cross-platform comparisons for pre-recorded experiments should be based on **normalised per-frame timing metrics** rather than raw total processing time.
* Effective FPS and slowdown factor are primarily relevant to **live camerafeed experiments**, where near-real-time feasibility is the main concern.

## Intended Use

This repository is intended for:

* benchmarking studies
* systems feasibility analysis
* prototype edge deployment research
* privacy-preserving rehabilitation analytics research

It is **not** intended to function as a clinical decision-support tool or a validated medical device.

## Citation

If you use this code, workflow, or benchmarking methodology in academic work, please cite the associated publication.

### BibTeX

```bibtex id="gh2g3j"
@inproceedings{jetson_telerehab_benchmark,
  title     = {Edge Deployment for Privacy-preserving Telerehabilitation: A Cross-platform Feasibility Study for Knee Rehabilitation},
  author    = {Weerakoon, Tharika and Ba{\v{c}}i{\'{c}}, Boris and GholamHosseini, Hamid and Martin,  {\v{S}tufi}},
  booktitle = {Future Technologies Conference},
  year      = {2026},
  note      = {Camera-ready version}
}
```

Update the entry as needed once full publication metadata is available.

## Future Directions

The benchmark provides a systems-level foundation for future work on:

* live data-streaming rehabilitation systems
* multimodal movement analysis
* locally governed and patient-controlled analytics workflows
* assistive monitoring and sports rehabilitation extensions
* integration of cognitive, biomechanical, and neurofeedback-informed processing

## Disclaimer

This repository is provided for research and benchmarking purposes only. The code and results are intended to support computational feasibility analysis and should not be interpreted as evidence of clinical effectiveness or diagnostic validity.
