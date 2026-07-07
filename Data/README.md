# Results Analysis for Cross-Platform Benchmarking

## Overview

This folder contains the summary-level benchmarking data and analysis code used to evaluate the performance of a MediaPipe-based knee-rehabilitation pipeline across **Windows**, **Linux Mint**, and **NVIDIA Jetson** platforms under **GUI** and **CLI** execution modes.

The analysis supports the cross-platform feasibility study by reproducing summary metrics, derived performance measures, and publication figures from both:

* **pre-recorded video experiments**
* **live camerafeed experiments**

## Contents

* `video_master_analysis.csv`
  Summary benchmarking results for pre-recorded video experiments.

* `camerafeed_master_analysis.csv`
  Summary benchmarking results for live camerafeed experiments.

* `results_analysis.py`
  Python analysis script for loading summary datasets, classifying execution environments, computing derived performance metrics, and generating publication-ready figures and summary outputs.

## Input Data Format

The analysis script expects one CSV file at a time, with one row per experimental run and at minimum the following columns:

* `total_time_s`
* `num_frames`
* `input_fps`
* `avg_frame_time_ms`
* `avg_read_time_ms`
* `avg_mediapipe_time_ms`
* `avg_other_time_ms`
* `os`
* `os_release`
* `python_version`
* `exe_mode` (e.g., `headed` / `headless`, corresponding to GUI / CLI execution)

## Derived Metrics

The analysis script can compute additional benchmarking measures, including:

* **effective FPS**
* **slowdown factor**
* cross-platform timing summaries
* execution-mode comparisons
* mean timing breakdowns for publication figures

## Running the Analysis

Run the script with one input CSV at a time.

### Example: pre-recorded video analysis

```bash
python results_analysis.py --csv video_master_analysis.csv --out ../figures --no-show
```

### Example: live camerafeed analysis

```bash
python results_analysis.py --csv camerafeed_master_analysis.csv --out ../figures --no-show
```

## Output

The analysis script can generate:

* summary figures for publication
* derived benchmarking metrics
* timing breakdown summaries
* analysis-ready outputs for cross-platform comparison

## Notes

* Pre-recorded videos were recorded at nominal frame rates of **59.94 fps** and **29.97 fps**. Accordingly, cross-platform comparison for pre-recorded experiments should rely on **normalised per-frame timing metrics** rather than raw total processing time.
* Effective FPS and slowdown factor are primarily relevant to **live camerafeed experiments**, where near-real-time feasibility is the main concern.
* The terms `headed` and `headless` in the raw CSV files correspond to **GUI mode** and **CLI mode**, respectively.

## Related Folders

* **Jetson/** – platform-specific scripts and configuration for NVIDIA Jetson experiments
* **Linux Mint/** – platform-specific scripts and configuration for Linux Mint experiments
* **Windows/** – platform-specific scripts and configuration for Windows experiments
