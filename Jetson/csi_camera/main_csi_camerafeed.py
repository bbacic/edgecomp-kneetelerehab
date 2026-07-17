# -*- coding: utf-8 -*-
"""
Jetson CSI camera – knee rehab pose analysis (GUI/GUIless, GStreamer).

Pipeline:
  Jetson CSI → GStreamer → OpenCV → MediaPipe Pose → CSV + time-series plot

Features:
- Opens CSI via GStreamer
- Imports MediaPipe only AFTER cap.isOpened() (Jetson TLS workaround)
- Computes left/right knee angles per frame
- Logs landmarks + angles + knee-correctness + timing
- Generates a final left/right knee angle time-series plot after exit
"""

import cv2
import math
import csv
import os
import time
import platform
import subprocess
import matplotlib

# ---------------------------------------------------------------------
# CONFIG: toggle this for GUIless (SSH) vs GUI (with display) mode
# ---------------------------------------------------------------------
HEADLESS = False  # set to True when running over SSH / no monitor

# Choose backend before importing pyplot
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# GStreamer pipeline – same as the working CSI minimal script
# ------------------------------------------------------------

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def calculate_angle(a, b, c):
    """Knee angle at b from three pose landmarks (2D)."""
    ab = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
    bc = math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)
    ac = math.sqrt((c.x - a.x) ** 2 + (c.y - a.y) ** 2)
    if ab * bc == 0:
        return 0.0
    cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def delta(a, b, c):
    """
    Orientation helper used to detect whether the knee is 'inside'
    or 'outside' relative to hip–knee–ankle line (for frontal view).
    """
    return (
        a.x * b.y + b.x * c.y + c.x * a.y
        - a.x * c.y - b.x * a.y - c.x * b.y
    )

def query_gpu():
    """Best-effort GPU stats; safe if nvidia-smi is missing."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        util_str, mem_str = [x.strip() for x in line.split(",")]
        return float(util_str), float(mem_str)
    except Exception:
        return None, None

# ------------------------------------------------------------
# Timeseries plot
# ------------------------------------------------------------

def plot_timeseries(timeframes,
                    left_angles,
                    right_angles,
                    left_correct_flags,
                    right_correct_flags,
                    headless=False,
                    output_png="jetson_csi_timeseries.png"):
    """
    Draw final time-series plot for left/right knee angles.

    Behaviour:
      - Always tries to save a PNG.
      - If headless=True: uses non-GUI backend (Agg) and does not call plt.show().
      - If headless=False: calls plt.show() (if backend supports it).
    """
    if not timeframes:
        print("[WARN] No frames with valid pose; skipping time-series plotting.")
        return

    print(f"[INFO] Plotting {len(timeframes)} samples to: {output_png}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4)

    ax1.set_title("Left Knee Angle")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (degrees)")

    ax2.set_title("Right Knee Angle")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (degrees)")

    # Left knee time series
    if left_angles:
        ax1.plot(timeframes, left_angles, label="Left knee angle")
        # Highlight segments marked incorrect (flag == 0)
        for i in range(1, len(left_angles)):
            if i < len(left_correct_flags) and left_correct_flags[i] == 0:
                ax1.plot(
                    timeframes[i-1:i+1],
                    left_angles[i-1:i+1],
                    "r",
                    linewidth=2,
                )
        ax1.legend()

    # Right knee time series
    if right_angles:
        ax2.plot(timeframes, right_angles, label="Right knee angle")
        for i in range(1, len(right_angles)):
            if i < len(right_correct_flags) and right_correct_flags[i] == 0:
                ax2.plot(
                    timeframes[i-1:i+1],
                    right_angles[i-1:i+1],
                    "r",
                    linewidth=2,
                )
        ax2.legend()

    plt.tight_layout()

    # Save PNG
    try:
        plt.savefig(output_png, dpi=150)
        print(f"[INFO] Saved time-series plot to: {os.path.abspath(output_png)}")
    except Exception as e:
        print(f"[WARN] Could not save plot PNG: {e}")

    # Show plot only in headed mode
    if not headless:
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] plt.show() failed (GUI backend issue?): {e}")

    plt.close(fig)

# ------------------------------------------------------------
# Main CSI analysis run
# ------------------------------------------------------------

def main():
    # ---- output locations ----
    angles_csv     = "jetson_csi_angles.csv"
    benchmark_csv  = "jetson_csi_benchmark.csv"
    export_knee    = "both"   # "left", "right", or "both"

    # ---- open CSI camera using the known-good pipeline ----
    pipeline = gstreamer_pipeline(flip_method=0)
    print("[INFO] Using pipeline:")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print("[INFO] cap.isOpened():", cap.isOpened())
    if not cap.isOpened():
        print("[ERROR] Could not open CSI camera via GStreamer.")
        return

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input <= 0:
        fps_input = 30.0  # default for timing

    # Log one row approx every 0.5 s
    frame_interval = int(max(1, fps_input // 2))

    # --------------------------------------------------------
    # IMPORTANT: import mediapipe ONLY AFTER camera is open
    # --------------------------------------------------------
    import mediapipe as mp

    mp_pose_module = mp.solutions.pose
    pose = mp_pose_module.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmark = mp_pose_module.PoseLandmark
    drawing_utils = mp.solutions.drawing_utils

    # ---- CSV for angles ----
    writer = None
    if angles_csv:
        out_dir = os.path.dirname(angles_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        f_csv = open(angles_csv, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow([
            "timeframe",
            "left_hip_x", "left_hip_y", "left_hip_z",
            "left_knee_x", "left_knee_y", "left_knee_z",
            "left_ankle_x", "left_ankle_y", "left_ankle_z",
            "right_hip_x", "right_hip_y", "right_hip_z",
            "right_knee_x", "right_knee_y", "right_knee_z",
            "right_ankle_x", "right_ankle_y", "right_ankle_z",
            "left_knee_angle", "right_knee_angle",
            "rep_id", "phase", "is_valid_rep",
        ])

    # ---- time-series storage for final plotting ----
    timeframes = []
    left_knee_angles = []
    right_knee_angles = []
    left_correct_flags = []   # 1 = correct, 0 = incorrect
    right_correct_flags = []

    # ---- timing + GPU stats ----
    frame_count = 0
    read_time_sum = 0.0
    mp_time_sum   = 0.0
    other_time_sum = 0.0
    total_start = time.perf_counter()
    gpu_util_start, gpu_mem_start = query_gpu()

    print("[INFO] Press 'q' to stop capture (headed). In headless mode, use Ctrl+C.")
    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed – stopping.")
            break
        read_done = time.perf_counter()
        read_time_sum += (read_done - loop_start)

        # ---- Mediapipe inference ----
        mp_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mp_done = time.perf_counter()
        mp_time_sum += (mp_done - mp_start)

        left_angle = 0.0
        right_angle = 0.0
        metrics = {"rep_id": 0, "phase": "live", "is_valid_rep": False}

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_hip   = lm[pose_landmark.LEFT_HIP]
            left_knee  = lm[pose_landmark.LEFT_KNEE]
            left_ankle = lm[pose_landmark.LEFT_ANKLE]
            right_hip   = lm[pose_landmark.RIGHT_HIP]
            right_knee  = lm[pose_landmark.RIGHT_KNEE]
            right_ankle = lm[pose_landmark.RIGHT_ANKLE]

            # ---- knee angles ----
            if export_knee in ("left", "both"):
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            if export_knee in ("right", "both"):
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # ---- time in seconds ----
            t = frame_count / fps_input

            # ---- per-frame knee correctness (frontal assumption) ----
            # Left knee: delta < 0 ⇒ knee on the inside (incorrect)
            left_inside = delta(left_hip, left_knee, left_ankle) < 0
            left_knee_correct = 0 if left_inside else 1

            # Right knee: delta > 0 ⇒ knee on the inside (incorrect)
            right_inside = delta(right_hip, right_knee, right_ankle) > 0
            right_knee_correct = 0 if right_inside else 1

            metrics["is_valid_rep"] = bool(left_knee_correct and right_knee_correct)

            # --- store for final time-series plot ---
            timeframes.append(t)
            left_knee_angles.append(left_angle)
            right_knee_angles.append(right_angle)
            left_correct_flags.append(left_knee_correct)
            right_correct_flags.append(right_knee_correct)

            # ---- CSV logging ----
            if writer and frame_count % frame_interval == 0:
                writer.writerow([
                    t,
                    left_hip.x,  left_hip.y,  left_hip.z,
                    left_knee.x, left_knee.y, left_knee.z,
                    left_ankle.x, left_ankle.y, left_ankle.z,
                    right_hip.x,  right_hip.y, right_hip.z,
                    right_knee.x, right_knee.y, right_knee.z,
                    right_ankle.x, right_ankle.y, right_ankle.z,
                    left_angle, right_angle,
                    metrics.get("rep_id", ""),
                    metrics.get("phase", ""),
                    metrics.get("is_valid_rep", ""),
                ])

            # Draw pose overlay
            drawing_utils.draw_landmarks(
                frame_bgr, results.pose_landmarks,
                mp_pose_module.POSE_CONNECTIONS
            )

        # ---- simple FPS label + GUI ----
        cv2.putText(
            frame_bgr,
            f"CSI stream ~{fps_input:.1f} FPS (input)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if not HEADLESS:
            cv2.imshow("Jetson CSI Pose (headed)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # In headless mode: no imshow / waitKey; exit via Ctrl+C.

        frame_count += 1
        loop_end = time.perf_counter()
        other_time_sum += (loop_end - mp_done)

    total_end = time.perf_counter()
    gpu_util_end, gpu_mem_end = query_gpu()
    total_time = total_end - total_start

    if writer:
        f_csv.close()

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # ---- write benchmark CSV (one row) ----
    os_name    = platform.system()
    os_release = platform.release()
    python_ver = platform.python_version()
    conda_env  = os.environ.get("CONDA_DEFAULT_ENV", "")
    is_conda   = "yes" if conda_env else "no"

    avg_frame_time_ms   = (total_time / frame_count * 1000) if frame_count > 0 else 0.0
    avg_read_time_ms    = (read_time_sum / frame_count * 1000) if frame_count > 0 else 0.0
    avg_mp_time_ms      = (mp_time_sum / frame_count * 1000) if frame_count > 0 else 0.0
    avg_other_time_ms   = (other_time_sum / frame_count * 1000) if frame_count > 0 else 0.0
    effective_input_fps = fps_input

    write_header = not os.path.exists(benchmark_csv)
    with open(benchmark_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "video_file",
                "total_time_s",
                "num_frames",
                "input_fps",
                "avg_frame_time_ms",
                "avg_read_time_ms",
                "avg_mediapipe_time_ms",
                "avg_other_time_ms",
                "os",
                "os_release",
                "python_version",
                "is_conda_env",
                "conda_env",
                "gpu_util_start_percent",
                "gpu_mem_start_MiB",
                "gpu_util_end_percent",
                "gpu_mem_end_MiB",
            ])
        w.writerow([
            "CAMERA_CSI",
            f"{total_time:.6f}",
            frame_count,
            f"{effective_input_fps:.3f}",
            f"{avg_frame_time_ms:.3f}",
            f"{avg_read_time_ms:.3f}",
            f"{avg_mp_time_ms:.3f}",
            f"{avg_other_time_ms:.3f}",
            os_name,
            os_release,
            python_ver,
            is_conda,
            conda_env,
            "" if gpu_util_start is None else f"{gpu_util_start:.1f}",
            "" if gpu_mem_start  is None else f"{gpu_mem_start:.1f}",
            "" if gpu_util_end   is None else f"{gpu_util_end:.1f}",
            "" if gpu_mem_end    is None else f"{gpu_mem_end:.1f}",
        ])

    # ---- FINAL: generate time-series plot after exit ----
    plot_timeseries(
        timeframes=timeframes,
        left_angles=left_knee_angles,
        right_angles=right_knee_angles,
        left_correct_flags=left_correct_flags,
        right_correct_flags=right_correct_flags,
        headless=HEADLESS,
        output_png="jetson_csi_timeseries.png",
    )

    print("[INFO] Finished CAMERA session.")
    print(f"       Frames:      {frame_count}")
    print(f"       Total time:  {total_time:.3f} s")
    print(f"       Benchmarks written to: {benchmark_csv}")
    if angles_csv:
        print(f"       Angles written to:     {angles_csv}")
    print(f"       Time-series plot:      jetson_csi_timeseries.png")

if __name__ == "__main__":
    main()
