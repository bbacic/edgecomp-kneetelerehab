import cv2
import mediapipe as mp
import math
import argparse
import csv
import matplotlib.pyplot as plt
import threading
import queue
import os
import sys

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
    # Calculate the Euclidean distance between two points
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

def distance(a, b):
    # Calculate the Euclidean distance between two points
    return math.sqrt(distance2(a, b))

def angle(a, b, c):
    # Calculate the measure of the angle B 
    # that is between two lines (AB and BC) 
    # determined by three points (A, B, C)
    # using the cosine theorem
    if distance2(a, b) * distance2(b, c) == 0:
        return 0
    return math.degrees(math.acos((distance2(a, b) + distance2(b, c) - distance2(a, c)) / (2 * distance(a, b) * distance(b, c))))

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
        
        ab_ratio = proj_ab / max_ab if max_ab != 0 else 0
        bc_ratio = proj_bc / max_bc if max_bc != 0 else 0
        
        ab_ratio = max(-1, min(1, ab_ratio))
        bc_ratio = max(-1, min(1, bc_ratio))
        
        angle_ab = math.degrees(math.acos(ab_ratio))
        angle_bc = math.degrees(math.acos(bc_ratio))
        
        return 180 - angle_ab - angle_bc
    else:
        ab = math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)
        bc = math.sqrt((c.x - b.x)**2 + (c.y - b.y)**2)
        ac = math.sqrt((c.x - a.x)**2 + (c.y - a.y)**2)
        
        if ab * bc == 0:
            return 0
            
        cos_angle = (bc**2 + ab**2 - ac**2)/(2*bc*ab)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))

def process_video(video_file, export_knee, output_csv=None, direction=None):
    """Process video file for knee angle analysis"""
    # Initialize video capture
    cap_file = cv2.VideoCapture(video_file)
    if not cap_file.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")
    
    # Get video properties
    fps = cap_file.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        print(f"Warning: Using default FPS value: {fps}")
    
    delay = int(1000 / fps)
    frame_interval = int(fps / 2)

    # Set up MediaPipe Pose with a lighter model
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=0,  # Use the lightest model (0, 1, or 2)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    if output_csv:
        csv_file = open(output_csv, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['timeframe', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee_x', 'left_knee_y', 'left_knee_z',
                         'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z',
                         'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_foot_direction', 'left_knee_angle', 'left_knee_correct',
                         'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee_x', 'right_knee_y', 'right_knee_z',
                         'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z',
                         'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_foot_direction', 'right_knee_angle', 'right_knee_correct'])

    # Initialize plots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)  # Increase space between plots
    ax1.set_title('Left Knee Angle')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax2.set_title('Right Knee Angle')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    left_knee_angles = []
    right_knee_angles = []
    timeframes = []

    # Compute the length of the left thigh and right thigh
    left_thigh_length = 0
    right_thigh_length = 0
    # Compute the length of the left calf and right calf
    left_calf_length = 0
    right_calf_length = 0

    frame_count = 0
    while True:
        # Read frame from video file
        ret_file, frame_file = cap_file.read()

        if not ret_file:
            break

        # Convert the image color space from BGR to RGB
        frame_file_rgb = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = mp_pose.process(frame_file_rgb)

        # Convert the image color space back to BGR
        frame_file = cv2.cvtColor(frame_file_rgb, cv2.COLOR_RGB2BGR)

        # Initialize default values
        left_hip = left_knee = left_ankle = left_foot_index = left_heel = None
        right_hip = right_knee = right_ankle = right_foot_index = right_heel = None
        angle_left_knee = angle_right_knee = 0
        left_foot_direction = right_foot_direction = ""
        left_knee_correct = right_knee_correct = 1

        timeframes.append(frame_count / fps)
        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            if export_knee in ('left', 'both'):
                left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                left_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                left_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HEEL]

                # Calculate the left thigh and left calf
                left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))
                
                angle_left_knee = calculate_angle(
                    left_hip, 
                    left_knee, 
                    left_ankle, 
                    view_type='front' if direction == 'forward' else 'side',
                    max_ab=left_thigh_length,
                    max_bc=left_calf_length
                )


                text_left_knee = f"LEFT KNEE\nANGLE: {angle_left_knee:.2f}"
                text_x_left = 10  # Left side of the frame
                text_y_left = frame_file.shape[0] - 150
                for i, line in enumerate(text_left_knee.split('\n')):
                    cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)  # Purple color for left knee text
                
                left_knee_x = int(left_knee.x * frame_file.shape[1])
                left_knee_y = int(left_knee.y * frame_file.shape[0])
                left_hip_x = int(left_hip.x * frame_file.shape[1])
                left_hip_y = int(left_hip.y * frame_file.shape[0])
                left_ankle_x = int(left_ankle.x * frame_file.shape[1])
                left_ankle_y = int(left_ankle.y * frame_file.shape[0])
                left_foot_index_x = int(left_foot_index.x * frame_file.shape[1])
                left_foot_index_y = int(left_foot_index.y * frame_file.shape[0])
                
                cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (128, 0, 128), -1)  # Purple color for left knee dot
                
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
                    if delta(left_hip, left_knee, left_ankle) < 0: # check if left knee is on the inside
                        line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                        left_knee_correct = 0
                    else:
                        line_color = (255, 255, 255)  # White color for lines
                        left_knee_correct = 1
                else:
                    if left_foot_direction == "right":
                        angle_left_knee = 360 - angle_left_knee

                    if (left_foot_direction == "left" and left_knee.x < left_foot_index.x) or (left_foot_direction == "right" and left_knee.x > left_foot_index.x):
                        line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                        left_knee_correct = 0
                    else:
                        line_color = (255, 255, 255)  # White color for lines
                        left_knee_correct = 1

                cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), line_color, 2)
                cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), line_color, 2)
                cv2.circle(frame_file, (left_foot_index_x, left_foot_index_y), 10, (0, 0, 255), -1)  # Red color for left foot index dot

                left_knee_angles.append(angle_left_knee)
                if len(left_knee_angles) > 1:
                    if left_knee_correct == 0:
                        ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'r')
                    else:
                        ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'purple')

            else:
                # Non-selected left knee
                left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]

                # Calculate the left thigh and left calf
                left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))
                
                angle_left_knee = calculate_angle(
                    left_hip, 
                    left_knee, 
                    left_ankle, 
                    view_type='front' if direction == 'forward' else 'side',
                    max_ab=left_thigh_length,
                    max_bc=left_calf_length
                )

                
                text_left_knee = f"LEFT KNEE\nANGLE: {angle_left_knee:.2f}"
                text_x_left = 10  # Left side of the frame
                text_y_left = frame_file.shape[0] - 150
                for i, line in enumerate(text_left_knee.split('\n')):
                    cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color for left knee text
                
                left_knee_x = int(left_knee.x * frame_file.shape[1])
                left_knee_y = int(left_knee.y * frame_file.shape[0])
                left_hip_x = int(left_hip.x * frame_file.shape[1])
                left_hip_y = int(left_hip.y * frame_file.shape[0])
                left_ankle_x = int(left_ankle.x * frame_file.shape[1])
                left_ankle_y = int(left_ankle.y * frame_file.shape[0])
                
                cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (0, 255, 0), -1)  # Green color for left knee dot
                cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)  # White color for lines
                cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)  # White color for lines
                left_foot_direction = ""
                left_knee_correct = 1

                left_knee_angles.append(angle_left_knee)
                if len(left_knee_angles) > 1:
                    ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'green')

            if export_knee in ('right', 'both'):
                right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                right_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                right_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]

                # Calculate the right thigh and right calf
                right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))
                
                angle_right_knee = calculate_angle(
                    right_hip, 
                    right_knee, 
                    right_ankle, 
                    view_type='front' if direction == 'forward' else 'side',
                    max_ab=right_thigh_length,
                    max_bc=right_calf_length
                )

                
                text_right_knee = f"RIGHT KNEE\nANGLE: {angle_right_knee:.2f}"
                text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x_right = frame_file.shape[1] - text_size[0] - 10  # Right side of the frame
                text_y_right = frame_file.shape[0] - 150
                for i, line in enumerate(text_right_knee.split('\n')):
                    cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Blue color for right knee text
                
                right_knee_x = int(right_knee.x * frame_file.shape[1])
                right_knee_y = int(right_knee.y * frame_file.shape[0])
                right_hip_x = int(right_hip.x * frame_file.shape[1])
                right_hip_y = int(right_hip.y * frame_file.shape[0])
                right_ankle_x = int(right_ankle.x * frame_file.shape[1])
                right_ankle_y = int(right_ankle.y * frame_file.shape[0])
                right_foot_index_x = int(right_foot_index.x * frame_file.shape[1])
                right_foot_index_y = int(right_foot_index.y * frame_file.shape[0])
                
                cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (255, 0, 0), -1)  # Blue color for right knee dot
                
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
                    if delta(right_hip, right_knee, right_ankle) > 0: # check if right knee is on the inside
                        line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                        right_knee_correct = 0
                    else:
                        line_color = (255, 255, 255)  # White color for lines
                        right_knee_correct = 1
                else:

                    if (right_foot_direction == "left" and right_knee.x < right_foot_index.x) or (right_foot_direction == "right" and right_knee.x > right_foot_index.x):
                        line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                        right_knee_correct = 0
                    else:
                        line_color = (255, 255, 255)  # White color for lines
                        right_knee_correct = 1

                cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), line_color, 2)
                cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), line_color, 2)
                cv2.circle(frame_file, (right_foot_index_x, right_foot_index_y), 10, (0, 0, 255), -1)  # Red color for right foot index dot

                right_knee_angles.append(angle_right_knee)
                if len(right_knee_angles) > 1:
                    if right_knee_correct == 0:
                        ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'r')
                    else:
                        ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'blue')

            else:
                # Non-selected right knee
                right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]

                # Calculate the right thigh and right calf
                right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))
                
                angle_right_knee = calculate_angle(
                    right_hip, 
                    right_knee, 
                    right_ankle, 
                    view_type='front' if direction == 'forward' else 'side',
                    max_ab=right_thigh_length,
                    max_bc=right_calf_length
                )

                text_right_knee = f"RIGHT KNEE\nANGLE: {angle_right_knee:.2f}"
                text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x_right = frame_file.shape[1] - text_size[0] - 10  # Right side of the frame
                text_y_right = frame_file.shape[0] - 150
                for i, line in enumerate(text_right_knee.split('\n')):
                    cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color for right knee text
                
                right_knee_x = int(right_knee.x * frame_file.shape[1])
                right_knee_y = int(right_knee.y * frame_file.shape[0])
                right_hip_x = int(right_hip.x * frame_file.shape[1])
                right_hip_y = int(right_hip.y * frame_file.shape[0])
                right_ankle_x = int(right_ankle.x * frame_file.shape[1])
                right_ankle_y = int(right_ankle.y * frame_file.shape[0])
                
                cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (0, 255, 0), -1)  # Green color for right knee dot
                cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)  # White color for lines
                cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)  # White color for lines
                right_foot_direction = ""
                right_knee_correct = 1

                right_knee_angles.append(angle_right_knee)
                if len(right_knee_angles) > 1:
                    ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'green')

            if output_csv and frame_count % frame_interval == 0:
                writer.writerow([frame_count / fps,
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
                                 right_foot_direction, angle_right_knee, right_knee_correct])
        
        # Show the frame
        cv2.imshow("Video and Pose Estimation", frame_file)

        # Update the plot
        plt.pause(0.01)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        frame_count += 1

    if output_csv:
        csv_file.close()

    # Release the video capture and close the window
    cap_file.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Finished processing the video.")

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
    parser = argparse.ArgumentParser(description="Process video for knee angle analysis.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("--export_knee", type=str, 
                       choices=['left', 'right', 'both'], 
                       default='both', 
                       help="Which knee angle(s) to export")
    parser.add_argument("--direction", type=str, 
                       choices=['left', 'right', 'forward'], 
                       required=True,
                       help="Movement direction (determines view mode)")

    args = parser.parse_args()

    main(args.video_file, args.output_csv, args.export_knee, args.direction)
