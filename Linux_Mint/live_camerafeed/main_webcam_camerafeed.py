import cv2
import mediapipe as mp
import math
import argparse
import csv
import os
import sys
import time
import platform
import subprocess
import matplotlib

# We will switch backend in main() based on --headless
import matplotlib.pyplot as plt

# ---------------------------
# Geometry helpers
# ---------------------------

def delta(a, b, c):
    """
    Oriented area test used to check if the knee is
    'inside' or 'outside' relative to hip–ankle line.
    """
    return (
        a.x * b.y + b.x * c.y + c.x * a.y
        - a.x * c.y - b.x * a.y - c.x * b.y
    )

def distance2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

def distance(a, b):
    return math.sqrt(distance2(a, b))

def calculate_angle(a, b, c, view_type='side', max_ab=None, max_bc=None):
    """
    Calculate angle between three points based on view type.

    Args:
        a: hip point
        b: knee point
        c: ankle point
        view_type: 'side' or 'front'
        max_ab: maximum thigh length (for front view normalisation)
        max_bc: maximum calf length (for front view normalisation)
    Returns:
        angle in degrees
    """
    if view_type == 'front':
        # Project points onto vertical plane
        proj_ab = abs(a.y - b.y)
        proj_bc = abs(b.y - c.y)

        # Use stored max lengths to normalise (avoid >1 cos arguments)
        if max_ab is None or max_ab == 0:
            max_ab = proj_ab if proj_ab != 0 else 1e-6
        if max_bc is None or max_bc == 0:
            max_bc = proj_bc if proj_bc != 0 else 1e-6

        ab_ratio = proj_ab / max_ab
        bc_ratio = proj_bc / max_bc

        ab_ratio = max(-1, min(1, ab_ratio))
        bc_ratio = max(-1, min(1, bc_ratio))

        angle_ab = math.degrees(math.acos(ab_ratio))
        angle_bc = math.degrees(math.acos(bc_ratio))

        return 180 - angle_ab - angle_bc
    else:
        # Standard 2D law-of-cosines angle
        ab = math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)
        bc = math.sqrt((c.x - b.x)**2 + (c.y - b.y)**2)
        ac = math.sqrt((c.x - a.x)**2 + (c.y - a.y)**2)

        if ab * bc == 0:
            return 0.0

        cos_angle = (bc**2 + ab**2 - ac**2) / (2 * bc * ab)
        cos_angle = max(-1, min(1, cos_angle))

        return math.degrees(math.acos(cos_angle))

# ---------------------------
# GPU stats via nvidia-smi
# ---------------------------

def query_gpu():
    """
    Try to query GPU utilization and memory using nvidia-smi.
    Returns (util_percent, mem_used_MiB) or (None, None) if not available.
    """
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

# ---------------------------
# Benchmark CSV writer
# ---------------------------

def write_benchmark(benchmark_path,
                    video_file_label,
                    total_time_s,
                    num_frames,
                    fps_input,
                    read_time_s,
                    mp_time_s,
                    other_time_s,
                    gpu_util_start,
                    gpu_mem_start,
                    gpu_util_end,
                    gpu_mem_end):
    header = [
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
    ]

    avg_frame_time_ms = (total_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_read_ms       = (read_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_mp_ms         = (mp_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_other_ms      = (other_time_s / num_frames * 1000) if num_frames > 0 else 0.0

    os_name    = platform.system()
    os_release = platform.release()
    python_ver = platform.python_version()
    conda_env  = os.environ.get("CONDA_DEFAULT_ENV", "")
    is_conda   = "yes" if conda_env else "no"

    write_header = not os.path.exists(benchmark_path)

    # Ensure directory exists if a path (not just filename) is given
    bench_dir = os.path.dirname(benchmark_path)
    if bench_dir:
        os.makedirs(bench_dir, exist_ok=True)

    with open(benchmark_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            video_file_label,
            f"{total_time_s:.6f}",
            num_frames,
            f"{fps_input:.3f}",
            f"{avg_frame_time_ms:.3f}",
            f"{avg_read_ms:.3f}",
            f"{avg_mp_ms:.3f}",
            f"{avg_other_ms:.3f}",
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

# ---------------------------
# Time-series plotting
# ---------------------------

def plot_timeseries(timeframes,
                    left_angles,
                    right_angles,
                    headless,
                    base_path,
                    left_flags=None,
                    right_flags=None):
    """
    Plot final time-series of left/right knee angles.
    Saves base_path + '_timeseries.png'.
    If headless=True -> no plt.show(), just save PNG.

    left_flags / right_flags: 1 = correct, 0 = incorrect.
    """
    if not timeframes:
        print("[WARN] No samples for timeseries plot; skipping.")
        return

    png_path = base_path + "_timeseries.png"
    print(f"[INFO] Plotting {len(timeframes)} samples to: {png_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4)

    ax1.set_title("Left Knee Angle")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (degrees)")
    ax2.set_title("Right Knee Angle")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (degrees)")

    # Left knee
    if left_angles:
        ax1.plot(timeframes, left_angles, label="Left knee")
        if left_flags is not None and len(left_flags) == len(left_angles):
            # overlay incorrect segments in red
            for i in range(1, len(left_angles)):
                if left_flags[i] == 0:
                    ax1.plot(
                        timeframes[i-1:i+1],
                        left_angles[i-1:i+1],
                        "r",
                        linewidth=2,
                    )
        ax1.legend()

    # Right knee
    if right_angles:
        ax2.plot(timeframes, right_angles, label="Right knee")
        if right_flags is not None and len(right_flags) == len(right_angles):
            for i in range(1, len(right_angles)):
                if right_flags[i] == 0:
                    ax2.plot(
                        timeframes[i-1:i+1],
                        right_angles[i-1:i+1],
                        "r",
                        linewidth=2,
                    )
        ax2.legend()

    plt.tight_layout()

    try:
        plt.savefig(png_path, dpi=150)
        print(f"[INFO] Saved timeseries plot to: {os.path.abspath(png_path)}")
    except Exception as e:
        print(f"[WARN] Could not save timeseries PNG: {e}")

    if not headless:
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] plt.show() failed (backend?): {e}")

    plt.close(fig)

# ---------------------------
# Shared per-frame correctness logic
# ---------------------------

def evaluate_knee_side(direction, hip, knee, ankle, foot_index, heel,
                       thigh_length, calf_length):
    """
    Compute knee angle and correctness flags for one leg.

    Returns:
      angle_deg, knee_correct (1/0), foot_direction, updated_thigh_length, updated_calf_length
    """
    # Update segment lengths (for 'front' view normalisation)
    updated_thigh = max(thigh_length, abs(hip.y - knee.y))
    updated_calf  = max(calf_length, abs(knee.y - ankle.y))

    view_type = 'front' if direction == 'forward' else 'side'

    angle_deg = calculate_angle(
        hip,
        knee,
        ankle,
        view_type=view_type,
        max_ab=updated_thigh,
        max_bc=updated_calf
    )

    # Infer or use given direction for foot
    if direction:
        foot_dir = direction
    else:
        if foot_index.x < ankle.x:
            foot_dir = "left"
        elif foot_index.x > ankle.x:
            foot_dir = "right"
        else:
            foot_dir = "forward"

    # Default: correct
    knee_correct = 1

    return angle_deg, knee_correct, foot_dir, updated_thigh, updated_calf

def left_knee_correctness(direction, hip, knee, ankle, foot_index,
                          angle_deg, knee_correct):
    """
    Apply left-knee specific correctness rules.
    Returns updated angle_deg, knee_correct.
    """
    foot_direction = direction if direction else None
    if not foot_direction:
        if foot_index.x < ankle.x:
            foot_direction = "left"
        elif foot_index.x > ankle.x:
            foot_direction = "right"
        else:
            foot_direction = "forward"

    if foot_direction == "forward":
        # Check if left knee is on the inside relative to hip–ankle
        if delta(hip, knee, ankle) < 0:
            knee_correct = 0
    else:
        if foot_direction == "right":
            angle_deg = 360 - angle_deg

        if ((foot_direction == "left" and knee.x < foot_index.x) or
            (foot_direction == "right" and knee.x > foot_index.x)):
            knee_correct = 0

    return angle_deg, knee_correct, foot_direction

def right_knee_correctness(direction, hip, knee, ankle, foot_index,
                           angle_deg, knee_correct):
    """
    Apply right-knee specific correctness rules.
    Returns updated angle_deg, knee_correct.
    """
    foot_direction = direction if direction else None
    if not foot_direction:
        if foot_index.x < ankle.x:
            foot_direction = "left"
        elif foot_index.x > ankle.x:
            foot_direction = "right"
        else:
            foot_direction = "forward"

    if foot_direction == "forward":
        # Check if right knee is on the inside relative to hip–ankle
        if delta(hip, knee, ankle) > 0:
            knee_correct = 0
    else:
        if ((foot_direction == "left" and knee.x < foot_index.x) or
            (foot_direction == "right" and knee.x > foot_index.x)):
            knee_correct = 0

    return angle_deg, knee_correct, foot_direction

# ---------------------------
# Core processing (video file)
# ---------------------------

def process_video_file(video_file, output_csv, export_knee, direction, headless):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_file}")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input <= 0:
        fps_input = 30.0

    frame_interval = int(max(1, fps_input // 2))

    mp_pose = mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    csv_file = None
    writer = None
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(output_csv, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "timeframe",
            "left_hip_x", "left_hip_y", "left_hip_z",
            "left_knee_x", "left_knee_y", "left_knee_z",
            "left_ankle_x", "left_ankle_y", "left_ankle_z",
            "left_foot_index_x", "left_foot_index_y", "left_foot_index_z",
            "left_heel_x", "left_heel_y", "left_heel_z",
            "left_foot_direction", "left_knee_angle", "left_knee_correct",
            "right_hip_x", "right_hip_y", "right_hip_z",
            "right_knee_x", "right_knee_y", "right_knee_z",
            "right_ankle_x", "right_ankle_y", "right_ankle_z",
            "right_foot_index_x", "right_foot_index_y", "right_foot_index_z",
            "right_heel_x", "right_heel_y", "right_heel_z",
            "right_foot_direction", "right_knee_angle", "right_knee_correct",
        ])

    # For timeseries plotting
    timeframes = []
    left_angles_series  = []
    right_angles_series = []
    left_flags  = []
    right_flags = []

    # Segment length trackers for front-view normalisation
    left_thigh_length  = 0.0
    left_calf_length   = 0.0
    right_thigh_length = 0.0
    right_calf_length  = 0.0

    frame_count = 0
    total_start = time.perf_counter()
    gpu_util_start, gpu_mem_start = query_gpu()
    read_time_sum = 0.0
    mp_time_sum   = 0.0
    other_time_sum = 0.0

    print(f"Processing video: {video_file}")

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        read_done = time.perf_counter()
        read_time_sum += (read_done - loop_start)

        mp_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)
        mp_done = time.perf_counter()
        mp_time_sum += (mp_done - mp_start)

        left_angle  = 0.0
        right_angle = 0.0
        left_knee_correct  = 1
        right_knee_correct = 1
        left_foot_dir  = ""
        right_foot_dir = ""

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Landmarks for both legs
            left_hip   = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee  = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            left_foot_index = lm[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
            left_heel       = lm[mp.solutions.pose.PoseLandmark.LEFT_HEEL]

            right_hip   = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
            right_knee  = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            right_foot_index = lm[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_heel       = lm[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]

            t = frame_count / fps_input

            # LEFT KNEE
            if export_knee in ("left", "both"):
                left_angle_raw, _, _, left_thigh_length, left_calf_length = evaluate_knee_side(
                    direction=direction,
                    hip=left_hip,
                    knee=left_knee,
                    ankle=left_ankle,
                    foot_index=left_foot_index,
                    heel=left_heel,
                    thigh_length=left_thigh_length,
                    calf_length=left_calf_length
                )
                left_angle, left_knee_correct, left_foot_dir = left_knee_correctness(
                    direction=direction,
                    hip=left_hip,
                    knee=left_knee,
                    ankle=left_ankle,
                    foot_index=left_foot_index,
                    angle_deg=left_angle_raw,
                    knee_correct=left_knee_correct
                )
            # RIGHT KNEE
            if export_knee in ("right", "both"):
                right_angle_raw, _, _, right_thigh_length, right_calf_length = evaluate_knee_side(
                    direction=direction,
                    hip=right_hip,
                    knee=right_knee,
                    ankle=right_ankle,
                    foot_index=right_foot_index,
                    heel=right_heel,
                    thigh_length=right_thigh_length,
                    calf_length=right_calf_length
                )
                right_angle, right_knee_correct, right_foot_dir = right_knee_correctness(
                    direction=direction,
                    hip=right_hip,
                    knee=right_knee,
                    ankle=right_ankle,
                    foot_index=right_foot_index,
                    angle_deg=right_angle_raw,
                    knee_correct=right_knee_correct
                )

            # Timeseries buffers
            timeframes.append(t)
            left_angles_series.append(left_angle)
            right_angles_series.append(right_angle)
            left_flags.append(left_knee_correct)
            right_flags.append(right_knee_correct)

            # CSV logging (subsampled)
            if writer and frame_count % frame_interval == 0:
                writer.writerow([
                    t,
                    left_hip.x, left_hip.y, left_hip.z,
                    left_knee.x, left_knee.y, left_knee.z,
                    left_ankle.x, left_ankle.y, left_ankle.z,
                    left_foot_index.x, left_foot_index.y, left_foot_index.z,
                    left_heel.x, left_heel.y, left_heel.z,
                    left_foot_dir, left_angle, left_knee_correct,
                    right_hip.x, right_hip.y, right_hip.z,
                    right_knee.x, right_knee.y, right_knee.z,
                    right_ankle.x, right_ankle.y, right_ankle.z,
                    right_foot_index.x, right_foot_index.y, right_foot_index.z,
                    right_heel.x, right_heel.y, right_heel.z,
                    right_foot_dir, right_angle, right_knee_correct,
                ])

        # Optional live preview if *not* headless
        if not headless:
            cv2.imshow("Video Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        loop_end = time.perf_counter()
        other_time_sum += (loop_end - mp_done)

    total_end = time.perf_counter()
    gpu_util_end, gpu_mem_end = query_gpu()
    total_time = total_end - total_start

    if csv_file:
        csv_file.close()

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Finished video: {video_file}")
    print(f"  Frames: {frame_count}")
    print(f"  Total time: {total_time:.3f}s")

    base = os.path.splitext(output_csv if output_csv else video_file)[0]
    benchmark_path = base + "_benchmark.csv"

    write_benchmark(
        benchmark_path=benchmark_path,
        video_file_label=video_file,
        total_time_s=total_time,
        num_frames=frame_count,
        fps_input=fps_input,
        read_time_s=read_time_sum,
        mp_time_s=mp_time_sum,
        other_time_s=other_time_sum,
        gpu_util_start=gpu_util_start,
        gpu_mem_start=gpu_mem_start,
        gpu_util_end=gpu_util_end,
        gpu_mem_end=gpu_mem_end,
    )

    print(f"Benchmark written to: {benchmark_path}")

    # Final timeseries plot with correctness flags
    plot_timeseries(
        timeframes=timeframes,
        left_angles=left_angles_series,
        right_angles=right_angles_series,
        headless=headless,
        base_path=base,
        left_flags=left_flags,
        right_flags=right_flags,
    )

# ---------------------------
# Core processing (camera)
# ---------------------------

def process_camera(output_csv, export_knee, direction, headless):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device 0.")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input <= 0:
        fps_input = 30.0

    frame_interval = int(max(1, fps_input // 2))

    mp_pose = mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    csv_file = None
    writer = None
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(output_csv, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "timeframe",
            "left_hip_x", "left_hip_y", "left_hip_z",
            "left_knee_x", "left_knee_y", "left_knee_z",
            "left_ankle_x", "left_ankle_y", "left_ankle_z",
            "left_foot_index_x", "left_foot_index_y", "left_foot_index_z",
            "left_heel_x", "left_heel_y", "left_heel_z",
            "left_foot_direction", "left_knee_angle", "left_knee_correct",
            "right_hip_x", "right_hip_y", "right_hip_z",
            "right_knee_x", "right_knee_y", "right_knee_z",
            "right_ankle_x", "right_ankle_y", "right_ankle_z",
            "right_foot_index_x", "right_foot_index_y", "right_foot_index_z",
            "right_heel_x", "right_heel_y", "right_heel_z",
            "right_foot_direction", "right_knee_angle", "right_knee_correct",
        ])

    # For timeseries plotting
    timeframes = []
    left_angles_series  = []
    right_angles_series = []
    left_flags  = []
    right_flags = []

    # Segment length trackers
    left_thigh_length  = 0.0
    left_calf_length   = 0.0
    right_thigh_length = 0.0
    right_calf_length  = 0.0

    frame_count = 0
    total_start = time.perf_counter()
    gpu_util_start, gpu_mem_start = query_gpu()
    read_time_sum = 0.0
    mp_time_sum   = 0.0
    other_time_sum = 0.0

    if headless:
        print("Processing CAMERA (headless). Press Ctrl+C in the terminal to interrupt.")
    else:
        print("Processing CAMERA (headed). Press 'q' in the window to exit.")

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Camera frame grab failed.")
                break
            read_done = time.perf_counter()
            read_time_sum += (read_done - loop_start)

            mp_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(frame_rgb)
            mp_done = time.perf_counter()
            mp_time_sum += (mp_done - mp_start)

            left_angle  = 0.0
            right_angle = 0.0
            left_knee_correct  = 1
            right_knee_correct = 1
            left_foot_dir  = ""
            right_foot_dir = ""

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                left_hip   = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_knee  = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                left_ankle = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                left_foot_index = lm[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                left_heel       = lm[mp.solutions.pose.PoseLandmark.LEFT_HEEL]

                right_hip   = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                right_knee  = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                right_foot_index = lm[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                right_heel       = lm[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]

                t = frame_count / fps_input

                # LEFT KNEE
                if export_knee in ("left", "both"):
                    left_angle_raw, _, _, left_thigh_length, left_calf_length = evaluate_knee_side(
                        direction=direction,
                        hip=left_hip,
                        knee=left_knee,
                        ankle=left_ankle,
                        foot_index=left_foot_index,
                        heel=left_heel,
                        thigh_length=left_thigh_length,
                        calf_length=left_calf_length
                    )
                    left_angle, left_knee_correct, left_foot_dir = left_knee_correctness(
                        direction=direction,
                        hip=left_hip,
                        knee=left_knee,
                        ankle=left_ankle,
                        foot_index=left_foot_index,
                        angle_deg=left_angle_raw,
                        knee_correct=left_knee_correct
                    )

                # RIGHT KNEE
                if export_knee in ("right", "both"):
                    right_angle_raw, _, _, right_thigh_length, right_calf_length = evaluate_knee_side(
                        direction=direction,
                        hip=right_hip,
                        knee=right_knee,
                        ankle=right_ankle,
                        foot_index=right_foot_index,
                        heel=right_heel,
                        thigh_length=right_thigh_length,
                        calf_length=right_calf_length
                    )
                    right_angle, right_knee_correct, right_foot_dir = right_knee_correctness(
                        direction=direction,
                        hip=right_hip,
                        knee=right_knee,
                        ankle=right_ankle,
                        foot_index=right_foot_index,
                        angle_deg=right_angle_raw,
                        knee_correct=right_knee_correct
                    )

                # Timeseries buffers
                timeframes.append(t)
                left_angles_series.append(left_angle)
                right_angles_series.append(right_angle)
                left_flags.append(left_knee_correct)
                right_flags.append(right_knee_correct)

                if writer and frame_count % frame_interval == 0:
                    writer.writerow([
                        t,
                        left_hip.x, left_hip.y, left_hip.z,
                        left_knee.x, left_knee.y, left_knee.z,
                        left_ankle.x, left_ankle.y, left_ankle.z,
                        left_foot_index.x, left_foot_index.y, left_foot_index.z,
                        left_heel.x, left_heel.y, left_heel.z,
                        left_foot_dir, left_angle, left_knee_correct,
                        right_hip.x, right_hip.y, right_hip.z,
                        right_knee.x, right_knee.y, right_knee.z,
                        right_ankle.x, right_ankle.y, right_ankle.z,
                        right_foot_index.x, right_foot_index.y, right_foot_index.z,
                        right_heel.x, right_heel.y, right_heel.z,
                        right_foot_dir, right_angle, right_knee_correct,
                    ])

            if not headless:
                # Overlay FPS and show
                cv2.putText(
                    frame,
                    f"Camera ~{fps_input:.1f} FPS (input)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Camera Pose Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            loop_end = time.perf_counter()
            other_time_sum += (loop_end - mp_done)

    except KeyboardInterrupt:
        print("Camera session interrupted by user.")

    total_end = time.perf_counter()
    gpu_util_end, gpu_mem_end = query_gpu()
    total_time = total_end - total_start

    if csv_file:
        csv_file.close()

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("Finished CAMERA.")
    print(f"  Frames: {frame_count}")
    print(f"  Total time: {total_time:.3f}s")

    effective_fps = (frame_count / total_time) if total_time > 0 else fps_input

    if output_csv:
        base = os.path.splitext(output_csv)[0]
    else:
        base = "camera_session"
    benchmark_path = base + "_benchmark.csv"

    write_benchmark(
        benchmark_path=benchmark_path,
        video_file_label="CAMERA",
        total_time_s=total_time,
        num_frames=frame_count,
        fps_input=effective_fps,
        read_time_s=read_time_sum,
        mp_time_s=mp_time_sum,
        other_time_s=other_time_sum,
        gpu_util_start=gpu_util_start,
        gpu_mem_start=gpu_mem_start,
        gpu_util_end=gpu_util_end,
        gpu_mem_end=gpu_mem_end,
    )

    print(f"Benchmark written to: {benchmark_path}")

    # Final timeseries plot with correctness flags
    plot_timeseries(
        timeframes=timeframes,
        left_angles=left_angles_series,
        right_angles=right_angles_series,
        headless=headless,
        base_path=base,
        left_flags=left_flags,
        right_flags=right_flags,
    )

# ---------------------------
# Main / CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Knee rehab pose analysis (headed/headless)."
    )
    parser.add_argument("video_file", nargs="?", default=None,
                        help="Path to the video file (ignored if --camera is used).")
    parser.add_argument("output_csv", nargs="?", default=None,
                        help="Path to the output CSV for landmarks/angles (optional).")
    parser.add_argument("--export_knee", type=str,
                        choices=["left", "right", "both"],
                        default="both",
                        help="Which knee(s) to export.")
    parser.add_argument("--direction", type=str,
                        choices=["left", "right", "forward"],
                        default="forward",
                        help="Movement direction (used by knee correctness logic).")
    parser.add_argument("--camera", action="store_true",
                        help="Use live camera instead of video file.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (no imshow / no plt.show).")

    args = parser.parse_args()

    # Set matplotlib backend for headless if needed
    if args.headless:
        try:
            matplotlib.use("Agg")
        except Exception as e:
            print(f"[WARN] Could not switch Matplotlib backend to Agg: {e}")

    try:
        if args.camera:
            process_camera(output_csv=args.output_csv,
                           export_knee=args.export_knee,
                           direction=args.direction,
                           headless=args.headless)
        else:
            if args.video_file is None:
                raise ValueError("video_file is required when --camera is not used.")
            process_video_file(video_file=args.video_file,
                               output_csv=args.output_csv,
                               export_knee=args.export_knee,
                               direction=args.direction,
                               headless=args.headless)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
