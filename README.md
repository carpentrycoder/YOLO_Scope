# YOLOScope Pro

**Advanced Computer Vision Platform with Real-time Multi-modal AI Analysis**

YOLOScope Pro is a comprehensive video analysis application that combines multiple computer vision technologies including YOLO object detection, MediaPipe pose estimation, hand tracking, face analysis, and real-time analytics dashboard.

## 🌟 Features

### Core Computer Vision Capabilities
- **🎯 Object Detection**: YOLOv8 integration with configurable model sizes (nano, small, medium)
- **🤸 Pose Detection**: Full-body pose estimation with angle calculations
- **👋 Hand Tracking**: Real-time hand landmark detection and tracking
- **😊 Face Analysis**: Face detection with basic emotion recognition
- **👤 Person Segmentation**: Background removal and person isolation
- **🎯 Object Tracking**: Multi-object tracking across video frames

### Advanced Analytics
- **📊 Real-time Dashboard**: Live metrics and performance monitoring
- **📈 Live Charts**: Detection trends, pose angles, and performance graphs
- **🔥 Motion Heatmap**: Visualization of movement patterns
- **🏃 Activity Recognition**: Automatic activity classification
- **😄 Emotion Analysis**: Basic facial emotion detection
- **⚡ Performance Metrics**: Detailed processing time analysis

### User Interface
- **🎨 Modern Design**: Glassmorphism UI with gradient backgrounds
- **📱 Responsive Layout**: Multi-column dashboard with live updates
- **🎛️ Real-time Controls**: Adjustable FPS, resize factors, and feature toggles
- **📊 Multiple Tabs**: Organized interface for analysis, analytics, training, and performance

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install streamlit
pip install ultralytics
pip install opencv-python
pip install mediapipe
pip install plotly
pip install pandas
pip install numpy
pip install pillow
pip install yt-dlp
pip install seaborn
pip install matplotlib
pip install torch
```

### Quick Start
```bash
git clone <repository-url>
cd yoloscope-pro
streamlit run app.py
```

## 📖 Usage

### 1. Launch the Application
```bash
streamlit run app.py
```

### 2. Configure Settings
- Select YOLO model size (nano/small/medium)
- Adjust confidence threshold (0.25-0.95)
- Enable/disable detection features:
  - Object Detection (YOLO)
  - Pose Detection
  - Hand Tracking
  - Face Analysis
  - Person Segmentation
  - Object Tracking

### 3. Input Methods
**Upload Video File:**
- Supported formats: MP4, AVI, MOV
- Drag and drop or browse to select

**YouTube Stream:**
- Paste YouTube URL for direct streaming
- Automatic quality optimization

### 4. Real-time Analysis
- **Live Metrics**: FPS, object counts, pose angles
- **Status Indicators**: Real-time feature activity monitoring
- **Live Charts**: Detection trends, pose analysis, performance metrics
- **Activity Recognition**: Automatic classification of human activities

### 5. Analytics Dashboard
- **Detection Trends**: Object detection over time
- **Activity Timeline**: Human activity classification
- **Emotion Distribution**: Facial emotion analysis
- **Performance Statistics**: Processing time breakdown

## 🎛️ Configuration Options

### Detection Features
| Feature | Description | Default |
|---------|-------------|---------|
| Object Detection | YOLOv8-based object detection | ✅ Enabled |
| Pose Detection | MediaPipe pose estimation | ✅ Enabled |
| Hand Tracking | Hand landmark detection | ✅ Enabled |
| Face Analysis | Face detection and emotion | ✅ Enabled |
| Person Segmentation | Background separation | ❌ Disabled |
| Object Tracking | Multi-object tracking | ✅ Enabled |

### Performance Settings
| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| FPS Limit | 1-30 | 15 | Processing frame rate |
| Resize Factor | 0.25-1.0 | 0.75 | Video scaling factor |
| Confidence Threshold | 0.25-0.95 | 0.5 | Detection confidence |

### Analytics Options
| Option | Description | Default |
|--------|-------------|---------|
| Motion Heatmap | Movement visualization | ❌ Disabled |
| Activity Recognition | Human activity classification | ✅ Enabled |
| Emotion Analysis | Facial emotion detection | ❌ Disabled |

## 📊 Output Formats

### Video Output
- **Processed Video**: Annotated video with all detections
- **Format**: AVI with XVID codec
- **Features**: Bounding boxes, pose landmarks, activity labels

### Analytics Data
- **JSON Export**: Complete detection and analytics data
- **CSV Export**: Performance metrics and timing data
- **Real-time Metrics**: Live dashboard updates

### Data Structure
```json
{
  "detections": {
    "person": [frame_numbers],
    "car": [frame_numbers]
  },
  "activities": ["walking", "standing", "waving"],
  "emotions": ["happy", "neutral", "focused"],
  "performance_metrics": {
    "avg_fps": 15.2,
    "total_frames": 1000,
    "processing_times": {
      "mean_total": 65.5,
      "mean_inference": 45.2
    }
  }
}
```

## 🏗️ Architecture

### Core Components
- **Streamlit Frontend**: Web-based user interface
- **YOLOv8**: Object detection engine
- **MediaPipe**: Pose, hand, and face analysis
- **OpenCV**: Video processing and computer vision
- **Plotly**: Interactive charts and visualizations

### Processing Pipeline
1. **Video Input**: File upload or YouTube stream
2. **Preprocessing**: Frame resizing and color conversion
3. **Multi-modal Analysis**: Parallel processing of all features
4. **Post-processing**: Annotation and tracking
5. **Analytics**: Real-time metrics and visualization
6. **Output**: Processed video and data export

## 🎨 UI Features

### Modern Design Elements
- **Glassmorphism Effects**: Translucent cards with backdrop blur
- **Gradient Backgrounds**: Dynamic color schemes
- **Live Animations**: Pulsing status indicators
- **Responsive Layout**: Adaptive column system
- **Interactive Charts**: Real-time data visualization

### Dashboard Components
- **Metrics Cards**: Live performance indicators
- **Status Panel**: Feature activity monitoring
- **Chart Grid**: Multi-chart analytics display
- **Progress Tracking**: Real-time processing updates

## 🔧 Advanced Features

### Custom Object Training
- Upload custom training images
- Label definition for specific objects
- Integration ready for annotation tools
- Extensible training pipeline

### Performance Optimization
- **Frame Skipping**: Adjustable processing rate
- **Resize Scaling**: Memory optimization
- **Batch Processing**: Efficient inference
- **Resource Monitoring**: Real-time performance tracking

### Error Handling
- **Model Loading**: Graceful fallback for missing models
- **Stream Processing**: Robust video handling
- **Memory Management**: Automatic cleanup
- **User Feedback**: Clear error messages and guidance

## 📈 Performance Metrics

### Processing Times
- **Preprocessing**: Frame preparation and conversion
- **Inference**: AI model processing time
- **Postprocessing**: Annotation and output generation
- **Total Pipeline**: End-to-end processing time

### Resource Utilization
- **FPS Monitoring**: Real-time frame rate tracking
- **Memory Usage**: Efficient resource management
- **GPU Acceleration**: CUDA support where available
- **Batch Optimization**: Intelligent processing strategies

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests and linting
5. Submit pull request

### Code Structure
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # YOLO model storage
├── temp/                # Temporary file processing
└── README.md            # Documentation
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **Google MediaPipe**: Pose and hand tracking
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **Plotly**: Interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review code comments for implementation details

---

**Built with ❤️ for Computer Vision**