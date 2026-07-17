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

# Custom pose connections
POSE_CONNECTIONS_BODY = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]

POSE_CONNECTIONS_KNEE = [
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
]


def delta(a, b, c):
    # Determine the position of a point regarding the line determined by another two points
    return a.x * b.y + b.x * c.y + c.x * a.y - a.x * c.y - b.x * a.y - c.x * b.y


def distance2(a, b):
    # Calculate the squared Euclidean distance between two points
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def distance(a, b):
    # Calculate the Euclidean distance between two points
    return math.sqrt(distance2(a, b))


def angle(a, b, c):
    # Calculate the measure of the angle B (between AB and BC) using the cosine theorem
    if distance2(a, b) * distance2(b, c) == 0:
        return 0
    return math.degrees(
        math.acos(
            (distance2(a, b) + distance2(b, c) - distance2(a, c))
            / (2 * distance(a, b) * distance(b, c))
        )
    )


def calculate_angle(a, b, c, view_type='side', max_ab=None, max_bc=None):
    """
    Calculate angle between three points based on view type
    Args:
        a: hip point
        b: knee point
        c: ankle point
        view_type: 'side' or 'front'
        max_ab: maximum thigh length (for front view)
        max_bc: maximum calf length (for front view)
    Returns:
        angle in degrees
    """
    if view_type == 'front':
        # Project points onto vertical plane
        proj_ab = abs(a.y - b.y)
        proj_bc = abs(b.y - c.y)

        ab_ratio = proj_ab / max_ab if max_ab not in (None, 0) else 0
        bc_ratio = proj_bc / max_bc if max_bc not in (None, 0) else 0

        ab_ratio = max(-1, min(1, ab_ratio))
        bc_ratio = max(-1, min(1, bc_ratio))

        angle_ab = math.degrees(math.acos(ab_ratio))
        angle_bc = math.degrees(math.acos(bc_ratio))

        return 180 - angle_ab - angle_bc
    else:
        ab = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        bc = math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)
        ac = math.sqrt((c.x - a.x) ** 2 + (c.y - a.y) ** 2)

        if ab * bc == 0:
            return 0

        cos_angle = (bc ** 2 + ab ** 2 - ac ** 2) / (2 * bc * ab)
        cos_angle = max(-1, min(1, cos_angle))

        return math.degrees(math.acos(cos_angle))


def get_gpu_stats():
    """
    Try to get GPU utilization and memory via nvidia-smi.
    Returns (util_percent, mem_MiB) or (None, None) if not available.
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
        return float(util_str), float(mem_str)  # percentage, MiB
    except Exception:
        return None, None


def write_benchmark(benchmark_path,
                    video_file,
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
    """
    Write a one-line benchmark summary for this run.
    """
    avg_frame_time_ms = (total_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_read_ms = (read_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_mp_ms = (mp_time_s / num_frames * 1000) if num_frames > 0 else 0.0
    avg_other_ms = (other_time_s / num_frames * 1000) if num_frames > 0 else 0.0

    os_name = platform.system()
    os_release = platform.release()
    python_ver = platform.python_version()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    is_conda = "yes" if conda_env else "no"

    header = [
        "video_file",
        "total_time_s",
        "num_frames",
        "input_fps",
        "avg_frame_time_ms",
        "avg_read_time_ms",          # OpenCV read/decoding
        "avg_mediapipe_time_ms",     # MediaPipe inference
        "avg_other_time_ms",         # Python loop + drawing + CSV etc.
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

    write_header = not os.path.exists(benchmark_path)

    with open(benchmark_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            video_file,
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
            "" if gpu_mem_start is None else f"{gpu_mem_start:.1f}",
            "" if gpu_util_end is None else f"{gpu_util_end:.1f}",
            "" if gpu_mem_end is None else f"{gpu_mem_end:.1f}",
        ])


def process_video(video_file, export_knee, output_csv=None, direction=None):
    """Process video file for knee angle analysis with timing and benchmarking (headless)."""
    # Initialize video capture
    cap_file = cv2.VideoCapture(video_file)
    if not cap_file.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    # Get video properties
    fps = cap_file.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        print(f"Warning: Using default FPS value: {fps}")

    frame_interval = int(max(1, fps / 2))

    # Set up MediaPipe Pose with a lighter model
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=0,  # Use the lightest model (0, 1, or 2)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # CSV for landmarks and angles
    csv_file = None
    writer = None
    if output_csv:
        csv_file = open(output_csv, mode="w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            'timeframe',
            'left_hip_x', 'left_hip_y', 'left_hip_z',
            'left_knee_x', 'left_knee_y', 'left_knee_z',
            'left_ankle_x', 'left_ankle_y', 'left_ankle_z',
            'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z',
            'left_heel_x', 'left_heel_y', 'left_heel_z',
            'left_foot_direction', 'left_knee_angle', 'left_knee_correct',
            'right_hip_x', 'right_hip_y', 'right_hip_z',
            'right_knee_x', 'right_knee_y', 'right_knee_z',
            'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
            'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z',
            'right_heel_x', 'right_heel_y', 'right_heel_z',
            'right_foot_direction', 'right_knee_angle', 'right_knee_correct',
        ])

    # Segment lengths
    left_thigh_length = 0.0
    right_thigh_length = 0.0
    left_calf_length = 0.0
    right_calf_length = 0.0

    frame_count = 0

    # Timing accumulators
    total_start = time.perf_counter()
    gpu_util_start, gpu_mem_start = get_gpu_stats()
    read_time_sum = 0.0          # OpenCV read/decoding
    mediapipe_time_sum = 0.0     # MediaPipe inference
    other_time_sum = 0.0         # Python loop + drawing + CSV, etc.

    while True:
        loop_start = time.perf_counter()
        ret_file, frame_file = cap_file.read()
        if not ret_file:
            break
        read_done = time.perf_counter()
        read_time_sum += (read_done - loop_start)

        # Convert the image color space from BGR to RGB
        mp_start = time.perf_counter()
        frame_file_rgb = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = mp_pose.process(frame_file_rgb)

        # Convert back to BGR (even though we don't display, we keep it for realistic timing)
        frame_file = cv2.cvtColor(frame_file_rgb, cv2.COLOR_RGB2BGR)
        mp_done = time.perf_counter()
        mediapipe_time_sum += (mp_done - mp_start)

        # Initialize default values
        left_hip = left_knee = left_ankle = left_foot_index = left_heel = None
        right_hip = right_knee = right_ankle = right_foot_index = right_heel = None
        angle_left_knee = angle_right_knee = 0.0
        left_foot_direction = right_foot_direction = ""
        left_knee_correct = right_knee_correct = 1

        if results.pose_landmarks:
            # LEFT KNEE
            if export_knee in ("left", "both"):
                left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                left_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                left_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HEEL]

                left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))

                angle_left_knee = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle,
                    view_type="front" if direction == "forward" else "side",
                    max_ab=left_thigh_length,
                    max_bc=left_calf_length,
                )

                if direction:
                    left_foot_direction = direction
                else:
                    if left_foot_index.x < left_ankle.x:
                        left_foot_direction = "left"
                    elif left_foot_index.x > left_ankle.x:
                        left_foot_direction = "right"
                    else:
                        left_foot_direction = "forward"

                if left_foot_direction == "forward":
                    if delta(left_hip, left_knee, left_ankle) < 0:
                        left_knee_correct = 0
                    else:
                        left_knee_correct = 1
                else:
                    if left_foot_direction == "right":
                        angle_left_knee = 360 - angle_left_knee

                    if (
                        left_foot_direction == "left"
                        and left_knee.x < left_foot_index.x
                    ) or (
                        left_foot_direction == "right"
                        and left_knee.x > left_foot_index.x
                    ):
                        left_knee_correct = 0
                    else:
                        left_knee_correct = 1
            else:
                # Non-selected left knee – still compute angle
                left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]

                left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))

                angle_left_knee = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle,
                    view_type="front" if direction == "forward" else "side",
                    max_ab=left_thigh_length,
                    max_bc=left_calf_length,
                )
                left_foot_direction = ""
                left_knee_correct = 1

            # RIGHT KNEE
            if export_knee in ("right", "both"):
                right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                right_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                right_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]

                right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))

                angle_right_knee = calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle,
                    view_type="front" if direction == "forward" else "side",
                    max_ab=right_thigh_length,
                    max_bc=right_calf_length,
                )

                if direction:
                    right_foot_direction = direction
                else:
                    if right_foot_index.x < right_ankle.x:
                        right_foot_direction = "left"
                    elif right_foot_index.x > right_ankle.x:
                        right_foot_direction = "right"
                    else:
                        right_foot_direction = "forward"

                if right_foot_direction == "forward":
                    if delta(right_hip, right_knee, right_ankle) > 0:
                        right_knee_correct = 0
                    else:
                        right_knee_correct = 1
                else:
                    if (
                        right_foot_direction == "left"
                        and right_knee.x < right_foot_index.x
                    ) or (
                        right_foot_direction == "right"
                        and right_knee.x > right_foot_index.x
                    ):
                        right_knee_correct = 0
                    else:
                        right_knee_correct = 1
            else:
                right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]

                right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))

                angle_right_knee = calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle,
                    view_type="front" if direction == "forward" else "side",
                    max_ab=right_thigh_length,
                    max_bc=right_calf_length,
                )
                right_foot_direction = ""
                right_knee_correct = 1

            # Write CSV every frame_interval frames
            if output_csv and writer is not None and frame_count % frame_interval == 0:
                timeframe = frame_count / fps
                writer.writerow([
                    timeframe,
                    left_hip.x if left_hip else None, left_hip.y if left_hip else None, left_hip.z if left_hip else None,
                    left_knee.x if left_knee else None, left_knee.y if left_knee else None, left_knee.z if left_knee else None,
                    left_ankle.x if left_ankle else None, left_ankle.y if left_ankle else None, left_ankle.z if left_ankle else None,
                    left_foot_index.x if left_foot_index else None, left_foot_index.y if left_foot_index else None, left_foot_index.z if left_foot_index else None,
                    left_heel.x if left_heel else None, left_heel.y if left_heel else None, left_heel.z if left_heel else None,
                    left_foot_direction, angle_left_knee, left_knee_correct,
                    right_hip.x if right_hip else None, right_hip.y if right_hip else None, right_hip.z if right_hip else None,
                    right_knee.x if right_knee else None, right_knee.y if right_knee else None, right_knee.z if right_knee else None,
                    right_ankle.x if right_ankle else None, right_ankle.y if right_ankle else None, right_ankle.z if right_ankle else None,
                    right_foot_index.x if right_foot_index else None, right_foot_index.y if right_foot_index else None, right_foot_index.z if right_foot_index else None,
                    right_heel.x if right_heel else None, right_heel.y if right_heel else None, right_heel.z if right_heel else None,
                    right_foot_direction, angle_right_knee, right_knee_correct,
                ])

        frame_count += 1
        loop_end = time.perf_counter()
        other_time_sum += (loop_end - mp_done)

    total_end = time.perf_counter()
    gpu_util_end, gpu_mem_end = get_gpu_stats()

    total_time = total_end - total_start

    if csv_file:
        csv_file.close()

    cap_file.release()
    cv2.destroyAllWindows()

    print(f"\nFinished processing the video: {video_file}")
    print(f"  Total frames   : {frame_count}")
    print(f"  Total time     : {total_time:.3f} s")
    if frame_count > 0:
        print(f"  Effective FPS  : {frame_count / total_time:.2f}")
        print(f"  Avg read (ms)  : {read_time_sum / frame_count * 1000:.3f}")
        print(f"  Avg MP (ms)    : {mediapipe_time_sum / frame_count * 1000:.3f}")
        print(f"  Avg other (ms) : {other_time_sum / frame_count * 1000:.3f}")

    # Benchmark file path (next to CSV if provided, else next to video)
    if output_csv:
        base, _ = os.path.splitext(output_csv)
    else:
        base, _ = os.path.splitext(video_file)
    benchmark_path = base + "_benchmark.csv"

    write_benchmark(
        benchmark_path=benchmark_path,
        video_file=video_file,
        total_time_s=total_time,
        num_frames=frame_count,
        fps_input=fps,
        read_time_s=read_time_sum,
        mp_time_s=mediapipe_time_sum,
        other_time_s=other_time_sum,
        gpu_util_start=gpu_util_start,
        gpu_mem_start=gpu_mem_start,
        gpu_util_end=gpu_util_end,
        gpu_mem_end=gpu_mem_end,
    )

    print(f"Benchmark written to: {benchmark_path}")


def main(video_file, output_csv, export_knee, direction=None):
    """Main function with error handling"""
    try:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")

        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        process_video(video_file, export_knee, output_csv, direction)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for knee angle analysis (headless).")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument(
        "--export_knee",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which knee angle(s) to export",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["left", "right", "forward"],
        required=True,
        help="Movement direction (determines view mode)",
    )

    args = parser.parse_args()
    main(args.video_file, args.output_csv, args.export_knee, args.direction)
