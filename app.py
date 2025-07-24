# Vehicle Tracking and Counting for VS Code
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import os
import threading
import time

# Load model
print("Loading YOLO model...")
model = YOLO('yolov8x.pt')
model.fuse()

# Dict mapping class_id to class_name
CLASS_NAME_DICT = model.model.names
print("Available classes loaded!")

# Class IDs of interest: car, motorcycle, bus, truck
classes = [2, 3, 5, 7]

# Check video file
video_path = 'highway-vehicles.mp4'
print("Checking video file...")

if os.path.exists(video_path):
    try:
        video_info = sv.VideoInfo.from_video_path(video_path)
        print(f"Video Info: {video_info}")
        video_available = True
    except Exception as e:
        print(f"Error loading video: {e}")
        video_available = False
else:
    print(f"Video file '{video_path}' not found!")
    video_available = False

if not video_available:
    print("Will use webcam instead...")

# Line configuration for counting
LINE_START = sv.Point(0, 400)
LINE_END = sv.Point(1280, 400)

# Global variables for video control
current_frame = None
is_paused = False
frame_count = 0
cap = None
byte_tracker = None
line_counter = None
annotators = None

def process_frame(frame, byte_tracker, line_counter, annotators):
    """Process each frame for vehicle detection, tracking, and counting"""
    
    # Run YOLO detection
    results = model(frame, verbose=False)[0]
    
    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter only vehicle classes
    detections = detections[np.isin(detections.class_id, classes)]
    
    # Update tracker
    detections = byte_tracker.update_with_detections(detections)
    
    # Start with original frame
    annotated_frame = frame.copy()
    
    # Draw traces if available
    if 'trace_annotator' in annotators:
        annotated_frame = annotators['trace_annotator'].annotate(
            scene=annotated_frame, detections=detections
        )
    
    # Draw bounding boxes
    annotated_frame = annotators['box_annotator'].annotate(
        scene=annotated_frame, detections=detections
    )
    
    # Add custom labels
    for detection_idx, (xyxy, confidence, class_id, tracker_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)
    ):
        x1, y1, x2, y2 = xyxy.astype(int)
        
        # Create label
        if tracker_id is not None:
            label = f"ID:{int(tracker_id)} {CLASS_NAME_DICT[class_id]} {confidence:.2f}"
        else:
            label = f"{CLASS_NAME_DICT[class_id]} {confidence:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Update line counter
    line_counter.trigger(detections)
    
    # Draw counting line
    annotated_frame = annotators['line_zone_annotator'].annotate(
        frame=annotated_frame, line_counter=line_counter
    )
    
    # Add counting statistics
    count_text = f"IN: {line_counter.in_count} | OUT: {line_counter.out_count} | NET: {line_counter.in_count - line_counter.out_count}"
    cv2.putText(annotated_frame, count_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Add current detections count
    current_count = len(detections)
    current_text = f"Current Vehicles: {current_count}"
    cv2.putText(annotated_frame, current_text, (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Add frame counter
    frame_text = f"Frame: {frame_count}"
    cv2.putText(annotated_frame, frame_text, (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame, len(detections)

def update_frame(frame_num):
    """Update function for matplotlib animation"""
    global current_frame, frame_count, is_paused, cap, byte_tracker, line_counter, annotators
    
    if is_paused or cap is None:
        return [im]
    
    ret, frame = cap.read()
    
    if not ret:
        print("End of video - restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return [im]
    
    frame_count += 1
    
    # Resize frame for consistent processing
    original_height, original_width = frame.shape[:2]
    if original_width != 1280 or original_height != 720:
        frame = cv2.resize(frame, (1280, 720))
    
    # Process frame
    processed_frame, current_vehicles = process_frame(frame, byte_tracker, line_counter, annotators)
    
    # Convert BGR to RGB for matplotlib
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Update image
    im.set_array(processed_frame_rgb)
    
    # Update title with statistics
    title = f"Vehicle Tracking & Counting | Frame: {frame_count} | IN: {line_counter.in_count} | OUT: {line_counter.out_count} | Current: {current_vehicles}"
    ax.set_title(title, pad=20)
    
    # Print statistics every 60 frames
    if frame_count % 60 == 0:
        print(f"Frame {frame_count} | IN: {line_counter.in_count} | OUT: {line_counter.out_count} | Current: {current_vehicles}")
    
    return [im]

def toggle_pause(event):
    """Toggle pause/play"""
    global is_paused
    is_paused = not is_paused
    print("Paused" if is_paused else "Resumed")

def reset_counters(event):
    """Reset counters"""
    global byte_tracker, line_counter
    byte_tracker = sv.ByteTrack(frame_rate=30)
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    print("Counters reset!")

def save_frame(event):
    """Save current frame"""
    global current_frame, frame_count
    if current_frame is not None:
        filename = f'vehicle_tracking_frame_{frame_count}.jpg'
        # Convert RGB back to BGR for saving
        save_img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, save_img)
        print(f"Saved frame as {filename}")

def on_key_press(event):
    """Handle keyboard events"""
    if event.key == 'q':
        plt.close('all')
        if cap:
            cap.release()
        print("Application closed")
    elif event.key == ' ':
        toggle_pause(None)
    elif event.key == 'r':
        reset_counters(None)
    elif event.key == 's':
        save_frame(None)

def main():
    """Main function to run vehicle tracking and counting"""
    global cap, byte_tracker, line_counter, annotators, im, ax, current_frame
    
    # Initialize trackers and counters
    byte_tracker = sv.ByteTrack(frame_rate=30)
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    
    # Initialize annotators
    try:
        box_annotator = sv.BoxAnnotator(thickness=2)
        line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.8)
        trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
        annotators = {
            'box_annotator': box_annotator,
            'line_zone_annotator': line_zone_annotator,
            'trace_annotator': trace_annotator
        }
    except Exception as e:
        print(f"Error initializing annotators: {e}")
        box_annotator = sv.BoxAnnotator()
        line_zone_annotator = sv.LineZoneAnnotator()
        
        annotators = {
            'box_annotator': box_annotator,
            'line_zone_annotator': line_zone_annotator
        }
    
    # Determine video source
    if video_available:
        print(f"Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Trying to use webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get first frame for initialization
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    first_frame = cv2.resize(first_frame, (1280, 720))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Setup matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Vehicle Tracking & Counting System', fontsize=16, fontweight='bold')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Display first frame
    im = ax.imshow(first_frame_rgb)
    
    # Add control buttons
    ax_pause = plt.axes([0.1, 0.02, 0.1, 0.04])
    ax_reset = plt.axes([0.25, 0.02, 0.1, 0.04])
    ax_save = plt.axes([0.4, 0.02, 0.1, 0.04])
    
    btn_pause = Button(ax_pause, 'Pause/Play')
    btn_reset = Button(ax_reset, 'Reset')
    btn_save = Button(ax_save, 'Save Frame')
    
    btn_pause.on_clicked(toggle_pause)
    btn_reset.on_clicked(reset_counters)
    btn_save.on_clicked(save_frame)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    print("Starting vehicle tracking and counting...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press SPACE to pause/resume")
    print("- Press 'r' to reset counters")
    print("- Press 's' to save current frame")
    print("- Use buttons at bottom of window")
    
    # Start animation
    ani = animation.FuncAnimation(
        fig, update_frame, interval=33, blit=False, cache_frame_data=False
    )
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Final statistics
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Vehicles IN: {line_counter.in_count}")
    print(f"Vehicles OUT: {line_counter.out_count}")
    print(f"Net count: {line_counter.in_count - line_counter.out_count}")
    
    # Cleanup
    if cap:
        cap.release()

if __name__ == "__main__":
    main()