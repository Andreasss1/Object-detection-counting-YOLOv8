# 🚗 Vehicle Tracking and Counting System with YOLOv8

A real-time vehicle detection, tracking, and counting system using YOLOv8 and computer vision techniques. This project demonstrates advanced object detection capabilities with precise vehicle tracking across designated counting lines.

![Vehicle Tracking Demo](https://github.com/Andreasss1/Object-detection-counting-YOLOv8/blob/main/object-tracking-and-counting.jpg)

## 🌟 Features

- **Real-time Object Detection**: Uses YOLOv8x model for accurate vehicle detection
- **Multi-class Vehicle Support**: Detects cars, motorcycles, buses, and trucks
- **Object Tracking**: Implements ByteTrack algorithm for consistent vehicle tracking
- **Bidirectional Counting**: Counts vehicles crossing the line in both directions
- **Visual Annotations**: 
  - Bounding boxes with confidence scores
  - Unique tracking IDs for each vehicle
  - Movement traces showing vehicle paths
  - Real-time statistics display
- **Interactive Controls**: Pause/resume, reset counters, save frames
- **Cross-platform Compatibility**: Works with both video files and webcam input

## 🛠️ Technologies Used

- **Python 3.8+**
- **YOLOv8** (Ultralytics) - Object Detection
- **Supervision** - Computer Vision utilities
- **OpenCV** - Image processing
- **Matplotlib** - Visualization and GUI
- **NumPy** - Numerical computations
- **ByteTrack** - Object tracking algorithm

## 📋 Requirements

```bash
pip install ultralytics
pip install supervision
pip install opencv-python
pip install matplotlib
pip install numpy
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle-tracking-system.git
   cd vehicle-tracking-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model** (automatically downloaded on first run)
   - The script will automatically download `yolov8x.pt` model

## 💻 Usage

### Basic Usage
```bash
python app.py
```

### Using with Custom Video
Place your video file in the project directory and rename it to `highway-vehicles.mp4`, or modify the `video_path` variable in the code.

### Using with Webcam
Set `video_available = False` in the code to use webcam input.

## 🎮 Controls

### Keyboard Shortcuts
- **Q**: Quit application
- **SPACE**: Pause/Resume video
- **R**: Reset all counters
- **S**: Save current frame as image

### Interactive Buttons
- **Pause/Play**: Toggle video playback
- **Reset**: Reset vehicle counters to zero
- **Save Frame**: Export current frame with annotations

## 📊 Key Metrics Displayed

- **IN Count**: Vehicles entering the monitored area
- **OUT Count**: Vehicles exiting the monitored area  
- **NET Count**: Difference between IN and OUT (IN - OUT)
- **Current Vehicles**: Number of vehicles currently visible in frame
- **Frame Counter**: Current frame number being processed

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   YOLOv8 Model   │───▶│   Detections    │
│ (File/Webcam)   │    │   (Detection)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Visual Output  │◀───│   Annotations    │◀───│  ByteTracker    │
│  (Matplotlib)   │    │   (Supervision)  │    │   (Tracking)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Line Counter    │◀───│  Line Zone      │
                       │   (Statistics)   │    │  (Detection)    │
                       └──────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Adjustable Parameters

```python
# Vehicle classes to detect (COCO dataset)
classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Counting line position
LINE_START = sv.Point(0, 400)
LINE_END = sv.Point(1280, 400)

# Tracker parameters
byte_tracker = sv.ByteTrack(frame_rate=30)

# Annotation settings
box_annotator = sv.BoxAnnotator(thickness=2)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
```

### Video Resolution
- Input videos are automatically resized to **1280x720** for optimal processing
- Maintains aspect ratio while ensuring consistent performance

## 📈 Performance

- **Processing Speed**: ~30 FPS on modern hardware
- **Detection Accuracy**: >90% for vehicles in good lighting conditions
- **Tracking Stability**: Maintains ID consistency across frames
- **Memory Usage**: ~2-4 GB RAM depending on video resolution

## 🎯 Use Cases

- **Traffic Monitoring**: Count vehicles at intersections or highways
- **Parking Management**: Monitor vehicle entry/exit in parking lots
- **Security Systems**: Track vehicle movement in restricted areas
- **Traffic Analysis**: Analyze traffic patterns and flow rates
- **Research Projects**: Study vehicle behavior and movement patterns

## 📁 Project Structure

```
vehicle-tracking-system/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── highway-vehicles.mp4  # Sample video file
├── .venv/                # Virtual environment
└── saved_frames/         # Directory for saved frames
```

## 🔍 Code Highlights

### Detection and Tracking
```python
# YOLO detection
results = model(frame, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

# Vehicle filtering
detections = detections[np.isin(detections.class_id, classes)]

# ByteTrack tracking
detections = byte_tracker.update_with_detections(detections)
```

### Counting Logic
```python
# Line zone counting
line_counter.trigger(detections)

# Statistics
in_count = line_counter.in_count
out_count = line_counter.out_count
net_count = in_count - out_count
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenCV GUI Error**
   - Solution: The project uses matplotlib instead of cv2.imshow() for VS Code compatibility

2. **Model Download Issues**
   - Solution: Ensure stable internet connection for YOLOv8 model download

3. **Video File Not Found**
   - Solution: Check video file path and ensure it exists in the project directory

4. **Performance Issues**
   - Solution: Reduce video resolution or use lighter YOLO model (yolov8n.pt)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [Supervision](https://github.com/roboflow/supervision) for computer vision utilities
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking algorithm
- OpenCV community for image processing tools

<div align="center">

## 📬 Need a Similar Project? Let's Collaborate!
If you need a **custom IoT project** for **smart home, agriculture, industrial monitoring**, or other use cases,  
I’m ready to assist you!  

📧 **Reach out at:**  
### andreas.sebayang9999@gmail.com  

Let’s create something amazing together! 🚀

</div>
