import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import time
from collections import defaultdict, deque
import yt_dlp
import subprocess
from PIL import Image
import mediapipe as mp
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch

# Fix for PyTorch 2.6 weights_only issue
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.block.SPPF',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.head.Detect',
    'collections.OrderedDict',
    'torch.nn.modules.conv.Conv2d',
    'torch.nn.modules.batchnorm.BatchNorm2d',
    'torch.nn.modules.activation.SiLU',
    'torch.nn.modules.pooling.MaxPool2d',
    'torch.nn.modules.upsampling.Upsample'
])

st.set_page_config(page_title="YOLOScope Pro", layout="wide")

st.markdown("""
    <style>
        /* Global app background & font */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #e0e0e0;
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(145deg, #6a11cb, #2575fc);
            border: none;
            color: #fff;
            border-radius: 12px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(106, 17, 203, 0.4);
        }

        /* Metric Card (Glassmorphism style) */
        .metric-card {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 18px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: transform 0.2s ease-in-out;
        }

        .metric-card:hover {
            transform: scale(1.02);
        }

        /* Live Analytics Container */
        .live-analytics {
            background: rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(8px);
        }

        /* Section headers */
        .feature-header {
            font-size: 1.4rem;
            font-weight: bold;
            color: #76baff;
            margin-bottom: 0.75rem;
            letter-spacing: 0.5px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }

        /* Live status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }

        .status-active {
            background-color: #00ff00;
        }

        .status-inactive {
            background-color: #ff4444;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Optional: Horizontal rule styling */
        hr {
            border: none;
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            margin: 1.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize MediaPipe solutions
@st.cache_resource
def init_mediapipe():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie = mp.solutions.selfie_segmentation
    
    return {
        'pose': mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5),
        'hands': mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7),
        'face_detection': mp_face.FaceDetection(min_detection_confidence=0.5),
        'face_mesh': mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5),
        'selfie_seg': mp_selfie.SelfieSegmentation(model_selection=1),
        'drawing': mp_drawing,
        'pose_connections': mp_pose.POSE_CONNECTIONS,
        'hands_connections': mp_hands.HAND_CONNECTIONS
    }

@st.cache_resource
def load_model(model_size):
    try:
        # Try loading with the new safe globals
        return YOLO(f'yolov8{model_size}.pt')
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        st.info("This might be due to PyTorch version compatibility. Try updating ultralytics: pip install -U ultralytics")
        return None

def get_streamable_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url'], info.get('title', 'YouTube Stream')

def calculate_pose_angles(landmarks):
    """Calculate key body angles for pose analysis"""
    def angle_between_points(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    angles = {}
    if landmarks:
        # Left arm angle
        angles['left_arm'] = angle_between_points(
            landmarks[11], landmarks[13], landmarks[15]  # shoulder, elbow, wrist
        )
        # Right arm angle
        angles['right_arm'] = angle_between_points(
            landmarks[12], landmarks[14], landmarks[16]
        )
        # Left leg angle
        angles['left_leg'] = angle_between_points(
            landmarks[23], landmarks[25], landmarks[27]  # hip, knee, ankle
        )
        # Right leg angle
        angles['right_leg'] = angle_between_points(
            landmarks[24], landmarks[26], landmarks[28]
        )
    return angles

def detect_activity(pose_angles):
    """Simple activity recognition based on pose angles"""
    if not pose_angles:
        return "Unknown"
    
    left_arm = pose_angles.get('left_arm', 180)
    right_arm = pose_angles.get('right_arm', 180)
    left_leg = pose_angles.get('left_leg', 180)  
    right_leg = pose_angles.get('right_leg', 180)
    
    # Simple heuristics
    if left_arm < 90 and right_arm < 90:
        return "Arms Up/Exercising"
    elif left_arm < 120 or right_arm < 120:
        return "Waving/Gesturing"
    elif abs(left_leg - right_leg) > 30:
        return "Walking/Moving"
    else:
        return "Standing/Neutral"

def analyze_emotions(face_landmarks):
    """Basic emotion analysis using MediaPipe facial landmarks"""
    if not face_landmarks:
        return "Neutral"
    
    # Simple emotion detection based on facial geometry
    # This is a basic implementation - for production use a trained model
    emotions = ["Neutral", "Happy", "Focused", "Surprised"]
    
    # In a real implementation, you would analyze landmark positions
    # For now, we'll use a simple random selection as placeholder
    import random
    return random.choice(emotions)

def create_mini_chart(data, chart_type="line", title="", height=150):
    """Create mini charts for live analytics"""
    if not data:
        return None
    
    fig = go.Figure()
    
    if chart_type == "line":
        fig.add_trace(go.Scatter(y=list(data), mode='lines+markers', 
                                line=dict(color='#00ff88', width=2),
                                marker=dict(size=4)))
    elif chart_type == "bar":
        if isinstance(data, dict):
            fig.add_trace(go.Bar(x=list(data.keys()), y=list(data.values()),
                                marker_color='#ff6b6b'))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color='white')),
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, color='white'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
        font=dict(color='white', size=10)
    )
    
    return fig

# Sidebar controls
st.sidebar.title("⚙️ YOLOScope Pro Settings")

# Model selection
model_size = st.sidebar.selectbox("YOLO Model Size", options=["n", "s", "m"], index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.95, 0.5, 0.05)

# Load model with error handling
model = load_model(model_size)
if model is None:
    st.error("❌ Could not load YOLO model. Please check your ultralytics installation.")
    st.stop()

# Feature toggles
st.sidebar.markdown("### 🎯 Detection Features")
enable_yolo = st.sidebar.checkbox("Object Detection (YOLO)", value=True)
enable_pose = st.sidebar.checkbox("Pose Detection", value=True)
enable_hands = st.sidebar.checkbox("Hand Tracking", value=True)
enable_face = st.sidebar.checkbox("Face Analysis", value=True)
enable_segmentation = st.sidebar.checkbox("Person Segmentation", value=False)
enable_tracking = st.sidebar.checkbox("Object Tracking", value=True)

# Analysis options
st.sidebar.markdown("### 📊 Analytics")
show_heatmap = st.sidebar.checkbox("Motion Heatmap", value=False)
activity_recognition = st.sidebar.checkbox("Activity Recognition", value=True)
emotion_analysis = st.sidebar.checkbox("Emotion Analysis", value=False)

# Initialize MediaPipe
try:
    mp_solutions = init_mediapipe()
except Exception as e:
    st.error(f"❌ Failed to initialize MediaPipe: {str(e)}")
    st.info("Please install MediaPipe: pip install mediapipe")
    st.stop()

# Training section (simplified)
st.sidebar.subheader("🧠 Train Custom Object")
name_label = st.sidebar.text_input("Label (e.g., Nikhil, Mug):")
image_files = st.sidebar.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# Main interface
st.title("🎥 YOLOScope Pro - Advanced Computer Vision")
st.markdown("**Multi-modal AI-powered video analysis with pose detection, hand tracking, and real-time analytics**")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Live Analysis", "📊 Analytics Dashboard", "🎯 Object Training", "📈 Performance"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload your video", type=["mp4", "avi", "mov"])
        st.markdown("**OR**")
        youtube_url = st.text_input("📺 Enter YouTube Video URL")
    
    with col2:
        st.markdown("### 🎛️ Real-time Controls")
        fps_limit = st.slider("FPS Limit", 1, 30, 15)
        resize_factor = st.slider("Resize Factor", 0.25, 1.0, 0.75, 0.25)

# Process video
video_path = None
video_stream_url = None
video_title = None

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
elif youtube_url:
    try:
        video_stream_url, video_title = get_streamable_url(youtube_url)
        st.success(f"✅ Streaming from YouTube: {video_title}")
        video_path = video_stream_url
    except Exception as e:
        st.error(f"❌ Failed to get YouTube stream: {e}")

if video_path is not None:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Failed to open video stream.")
    else:
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25
        
        # Output video setup
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        
        # UI elements
        with tab1:
            # Video display
            stframe = st.empty()
            
            # Progress bar
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            # Live Analytics Section - This is the main addition
            st.markdown('<div class="feature-header">📊 Live Analytics Dashboard</div>', unsafe_allow_html=True)
            
            # Create columns for live analytics
            analytics_container = st.container()
            with analytics_container:
                # Row 1: Key Metrics
                metrics_row = st.columns(6)
                fps_metric = metrics_row[0].empty()
                detection_metric = metrics_row[1].empty()
                pose_metric = metrics_row[2].empty()
                activity_metric = metrics_row[3].empty()
                emotion_metric = metrics_row[4].empty()
                performance_metric = metrics_row[5].empty()
                
                # Row 2: Status Indicators
                status_row = st.columns(5)
                yolo_status = status_row[0].empty()
                pose_status = status_row[1].empty()
                hands_status = status_row[2].empty()
                face_status = status_row[3].empty()
                tracking_status = status_row[4].empty()
                
                # Row 3: Live Charts
                charts_row = st.columns(3)
                detection_trend_chart = charts_row[0].empty()
                pose_angles_chart = charts_row[1].empty()
                performance_chart = charts_row[2].empty()
                
                # Row 4: Additional Analytics
                additional_row = st.columns(2)
                activity_distribution_chart = additional_row[0].empty()
                emotion_trend_chart = additional_row[1].empty()
        
        # Analytics containers for other tabs
        with tab2:
            analytics_col1, analytics_col2 = st.columns(2)
            with analytics_col1:
                detection_chart = st.empty()
                pose_chart = st.empty()
            with analytics_col2:
                activity_chart = st.empty()
                emotion_chart = st.empty()
        
        # Initialize tracking variables
        detection_history = defaultdict(list)
        pose_history = []
        activity_history = []
        emotion_history = []
        performance_metrics = {
            'inference_times': deque(maxlen=50),
            'preprocessing_times': deque(maxlen=50),
            'postprocessing_times': deque(maxlen=50),
            'total_times': deque(maxlen=50),
            'fps_history': deque(maxlen=50)
        }
        
        # Live analytics data
        live_detection_counts = deque(maxlen=30)
        live_pose_angles = {'left_arm': deque(maxlen=30), 'right_arm': deque(maxlen=30)}
        live_activities = deque(maxlen=20)
        live_emotions = deque(maxlen=20)
        
        # Motion heatmap
        if show_heatmap:
            motion_accumulator = np.zeros((height, width), dtype=np.float32)
        
        # Object tracking
        if enable_tracking:
            trackers = []
            track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        frame_num = 0
        frame_skip = max(1, int(fps / fps_limit))
        
        # Process video frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1000  # Fallback for streams
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip != 0:
                frame_num += 1
                continue
            
            start_total = time.time()
            
            # Preprocessing
            start_pre = time.time()
            frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            end_pre = time.time()
            
            # Main inference
            start_inf = time.time()
            annotated_frame = frame.copy()
            
            # Initialize frame data
            frame_detections = 0
            current_pose_data = {}
            current_activity = "Unknown"
            current_emotion = "Neutral"
            
            # Status flags
            yolo_active = False
            pose_active = False
            hands_active = False
            face_active = False
            
            # YOLO Detection
            if enable_yolo and model is not None:
                try:
                    results = model(frame, conf=confidence_threshold)
                    annotated_frame = results[0].plot()
                    
                    # Count detections
                    class_count = defaultdict(int)
                    for box in results[0].boxes.data:
                        cls = int(box[-1])
                        class_name = model.model.names[cls]
                        class_count[class_name] += 1
                        detection_history[class_name].append(frame_num)
                        frame_detections += 1
                    
                    yolo_active = len(class_count) > 0
                    live_detection_counts.append(frame_detections)
                    
                except Exception as e:
                    st.warning(f"YOLO detection error: {str(e)}")
            
            # Pose detection
            if enable_pose:
                try:
                    pose_results = mp_solutions['pose'].process(rgb_frame)
                    if pose_results.pose_landmarks:
                        mp_solutions['drawing'].draw_landmarks(
                            annotated_frame, pose_results.pose_landmarks, 
                            mp_solutions['pose_connections']
                        )
                        
                        # Calculate pose angles
                        pose_angles = calculate_pose_angles(pose_results.pose_landmarks.landmark)
                        current_pose_data = pose_angles
                        pose_active = True
                        
                        # Update live pose data
                        if 'left_arm' in pose_angles:
                            live_pose_angles['left_arm'].append(pose_angles['left_arm'])
                        if 'right_arm' in pose_angles:
                            live_pose_angles['right_arm'].append(pose_angles['right_arm'])
                        
                        # Activity recognition
                        if activity_recognition:
                            current_activity = detect_activity(pose_angles)
                            activity_history.append(current_activity)
                            live_activities.append(current_activity)
                            cv2.putText(annotated_frame, f"Activity: {current_activity}", 
                                      (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    st.warning(f"Pose detection error: {str(e)}")
            
            # Hand tracking
            if enable_hands:
                try:
                    hand_results = mp_solutions['hands'].process(rgb_frame)
                    if hand_results.multi_hand_landmarks:
                        hands_active = True
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_solutions['drawing'].draw_landmarks(
                                annotated_frame, hand_landmarks, 
                                mp_solutions['hands_connections']
                            )
                except Exception as e:
                    st.warning(f"Hand tracking error: {str(e)}")
            
            # Face analysis
            if enable_face:
                try:
                    face_results = mp_solutions['face_detection'].process(rgb_frame)
                    if face_results.detections:
                        face_active = True
                        for detection in face_results.detections:
                            mp_solutions['drawing'].draw_detection(annotated_frame, detection)
                        
                        # Emotion analysis (placeholder)
                        if emotion_analysis:
                            current_emotion = analyze_emotions(face_results.detections)
                            emotion_history.append(current_emotion)
                            live_emotions.append(current_emotion)
                            cv2.putText(annotated_frame, f"Emotion: {current_emotion}", 
                                      (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                except Exception as e:
                    st.warning(f"Face analysis error: {str(e)}")
            
            # Person segmentation
            if enable_segmentation:
                try:
                    seg_results = mp_solutions['selfie_seg'].process(rgb_frame)
                    mask = seg_results.segmentation_mask > 0.5
                    annotated_frame[mask] = annotated_frame[mask] * 0.7 + np.array([50, 50, 150]) * 0.3
                except Exception as e:
                    st.warning(f"Segmentation error: {str(e)}")
            
            # Motion heatmap
            if show_heatmap and frame_num > 0:
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if 'prev_gray' in locals():
                    diff = cv2.absdiff(gray_current, prev_gray)
                    motion_accumulator += diff.astype(np.float32) / 255.0
                prev_gray = gray_current
            
            end_inf = time.time()
            
            # Post-processing
            start_post = time.time()
            out.write(annotated_frame)
            
            # Update performance metrics
            preprocessing_time = (end_pre - start_pre) * 1000
            inference_time = (end_inf - start_inf) * 1000
            postprocessing_time = (time.time() - start_post) * 1000
            total_time = (time.time() - start_total) * 1000
            current_fps = 1000 / total_time if total_time > 0 else 0
            
            performance_metrics['preprocessing_times'].append(preprocessing_time)
            performance_metrics['inference_times'].append(inference_time)
            performance_metrics['postprocessing_times'].append(postprocessing_time)
            performance_metrics['total_times'].append(total_time)
            performance_metrics['fps_history'].append(current_fps)
            
            end_post = time.time()
            
            # Display frame
            stframe.image(annotated_frame, channels="BGR")
            
            # Update progress
            progress = min(frame_num / total_frames, 1.0)
            progress_bar.progress(progress)
            progress_text.text(f"Processing frame {frame_num}/{total_frames} | FPS: {current_fps:.1f}")
            
            # Update live analytics every frame
            with analytics_container:
                # Update metrics
                fps_metric.metric("🎯 FPS", f"{current_fps:.1f}")
                detection_metric.metric("🔍 Objects", frame_detections)
                
                if current_pose_data:
                    avg_angle = np.mean(list(current_pose_data.values()))
                    pose_metric.metric("🤸 Pose Angle", f"{avg_angle:.1f}°")
                else:
                    pose_metric.metric("🤸 Pose Angle", "N/A")
                
                activity_metric.metric("🏃 Activity", current_activity)
                emotion_metric.metric("😊 Emotion", current_emotion)
                performance_metric.metric("⚡ Avg Time", f"{np.mean(list(performance_metrics['total_times'])):.1f}ms")
                
                # Update status indicators
                yolo_status.markdown(f"<span class='status-indicator {'status-active' if yolo_active else 'status-inactive'}'></span>YOLO Detection", unsafe_allow_html=True)
                pose_status.markdown(f"<span class='status-indicator {'status-active' if pose_active else 'status-inactive'}'></span>Pose Detection", unsafe_allow_html=True)
                hands_status.markdown(f"<span class='status-indicator {'status-active' if hands_active else 'status-inactive'}'></span>Hand Tracking", unsafe_allow_html=True)
                face_status.markdown(f"<span class='status-indicator {'status-active' if face_active else 'status-inactive'}'></span>Face Detection", unsafe_allow_html=True)
                tracking_status.markdown(f"<span class='status-indicator {'status-active' if enable_tracking else 'status-inactive'}'></span>Object Tracking", unsafe_allow_html=True)
                
                # Update live charts every 5 frames for performance
                if frame_num % 5 == 0:
                    # Detection trend
                    if live_detection_counts:
                        fig = create_mini_chart(live_detection_counts, "line", "Detection Count Trend")
                        if fig:
                            detection_trend_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Pose angles
                    if live_pose_angles['left_arm'] or live_pose_angles['right_arm']:
                        fig = go.Figure()
                        if live_pose_angles['left_arm']:
                            fig.add_trace(go.Scatter(y=list(live_pose_angles['left_arm']), 
                                                   name='Left Arm', line=dict(color='#ff6b6b')))
                        if live_pose_angles['right_arm']:
                            fig.add_trace(go.Scatter(y=list(live_pose_angles['right_arm']), 
                                                   name='Right Arm', line=dict(color='#4ecdc4')))
                        fig.update_layout(
                            title="Pose Angles (degrees)",
                            height=150,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                            font=dict(color='white', size=10),
                            showlegend=True,
                            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
                        )
                        pose_angles_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Performance chart
                    if performance_metrics['fps_history']:
                        fig = create_mini_chart(performance_metrics['fps_history'], "line", "FPS Performance")
                        if fig:
                            performance_chart.plotly_chart(fig, use_container_width=True)
                
                # Update distribution charts every 10 frames
                if frame_num % 10 == 0:
                    # Activity distribution
                    if live_activities:
                        activity_counts = {}
                        for activity in live_activities:
                            activity_counts[activity] = activity_counts.get(activity, 0) + 1
                        
                        fig = go.Figure(data=[go.Pie(labels=list(activity_counts.keys()), 
                                                   values=list(activity_counts.values()),
                                                   hole=0.3)])
                        fig.update_layout(
                            title="Activity Distribution",
                            height=150,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=10),
                            showlegend=False
                        )
                        activity_distribution_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Emotion trend
                    if live_emotions:
                        emotion_counts = {}
                        for emotion in live_emotions:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        fig = go.Figure(data=[go.Bar(x=list(emotion_counts.keys()), 
                                                   y=list(emotion_counts.values()),
                                                   marker_color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])])
                        fig.update_layout(
                            title="Emotion Distribution",
                            height=150,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, color='white'),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
                            font=dict(color='white', size=10)
                        )
                        emotion_trend_chart.plotly_chart(fig, use_container_width=True)
            
            frame_num += 1
            
            # Break condition for very long videos
            if frame_num > 1000:  # Limit for demo
                break
        
        cap.release()
        out.release()
        
        # Final results
        st.success("✅ Analysis Complete!")
        
        # Show final analytics in live section
        with tab1:
            st.markdown("## 🎉 Final Analysis Summary")
            
            final_col1, final_col2, final_col3, final_col4 = st.columns(4)
            with final_col1:
                st.metric("Total Frames Processed", frame_num)
            with final_col2:
                if performance_metrics['fps_history']:
                    avg_fps = np.mean(list(performance_metrics['fps_history']))
                    st.metric("Average FPS", f"{avg_fps:.1f}")
            with final_col3:
                if detection_history:
                    total_detections = sum(len(frames) for frames in detection_history.values())
                    st.metric("Total Detections", total_detections)
            with final_col4:
                if activity_history:
                    unique_activities = len(set(activity_history))
                    st.metric("Unique Activities", unique_activities)
            
            # Final comprehensive charts
            if len(list(performance_metrics['total_times'])) > 10:
                st.markdown("### 📊 Comprehensive Performance Analysis")
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    # Performance over time
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=list(performance_metrics['total_times']), 
                                           name='Total Processing Time', 
                                           line=dict(color='#ff6b6b')))
                    fig.add_trace(go.Scatter(y=list(performance_metrics['inference_times']), 
                                           name='Inference Time', 
                                           line=dict(color='#4ecdc4')))
                    fig.update_layout(
                        title="Processing Performance Over Time",
                        xaxis_title="Frame",
                        yaxis_title="Time (ms)",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with perf_col2:
                    # FPS stability
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=list(performance_metrics['fps_history']), 
                                           mode='lines+markers',
                                           name='FPS',
                                           line=dict(color='#00ff88')))
                    fig.add_hline(y=np.mean(list(performance_metrics['fps_history'])), 
                                line_dash="dash", line_color="yellow",
                                annotation_text=f"Avg: {np.mean(list(performance_metrics['fps_history'])):.1f}")
                    fig.update_layout(
                        title="FPS Stability",
                        xaxis_title="Frame",
                        yaxis_title="FPS",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show final analytics in analytics tab
        with tab2:
            st.markdown("## 📊 Complete Analytics Report")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames Processed", frame_num)
            with col2:
                if performance_metrics['fps_history']:
                    avg_fps = np.mean(list(performance_metrics['fps_history']))
                    st.metric("Average FPS", f"{avg_fps:.1f}")
            with col3:
                if detection_history:
                    total_detections = sum(len(frames) for frames in detection_history.values())
                    st.metric("Total Detections", total_detections)
            with col4:
                if activity_history:
                    unique_activities = len(set(activity_history))
                    st.metric("Unique Activities", unique_activities)
            
            # Detection trends
            if detection_history:
                detection_df = pd.DataFrame([
                    {'Frame': f, 'Object': obj, 'Count': 1} 
                    for obj, frames in detection_history.items() 
                    for f in frames[-200:]  # Last 200 detections
                ])
                if not detection_df.empty:
                    fig = px.line(detection_df.groupby(['Frame', 'Object']).sum().reset_index(), 
                                x='Frame', y='Count', color='Object', 
                                title="Object Detection Trends Over Time")
                    fig.update_layout(height=400, template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Activity and emotion analysis
            analytics_col1, analytics_col2 = st.columns(2)
            
            with analytics_col1:
                # Activity timeline
                if activity_history:
                    activity_df = pd.DataFrame({
                        'Frame': range(len(activity_history)),
                        'Activity': activity_history
                    })
                    fig = px.scatter(activity_df, x='Frame', y='Activity', color='Activity',
                                   title="Activity Timeline")
                    fig.update_layout(height=400, template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            with analytics_col2:
                # Emotion distribution
                if emotion_history:
                    emotion_counts = pd.Series(emotion_history).value_counts()
                    fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                               title="Overall Emotion Distribution")
                    fig.update_layout(height=400, template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Performance tab
        with tab4:
            if performance_metrics['total_times']:
                st.markdown("## ⚡ Detailed Performance Analysis")
                
                # Performance statistics
                perf_data = {
                    'Preprocessing': list(performance_metrics['preprocessing_times']),
                    'Inference': list(performance_metrics['inference_times']),
                    'Postprocessing': list(performance_metrics['postprocessing_times']),
                    'Total': list(performance_metrics['total_times'])
                }
                
                perf_stats = pd.DataFrame({
                    'Metric': ['Preprocessing', 'Inference', 'Postprocessing', 'Total'],
                    'Mean (ms)': [np.mean(perf_data[k]) for k in perf_data.keys()],
                    'Std (ms)': [np.std(perf_data[k]) for k in perf_data.keys()],
                    'Min (ms)': [np.min(perf_data[k]) for k in perf_data.keys()],
                    'Max (ms)': [np.max(perf_data[k]) for k in perf_data.keys()],
                    'Median (ms)': [np.median(perf_data[k]) for k in perf_data.keys()]
                })
                st.dataframe(perf_stats, use_container_width=True)
                
                # Performance distribution
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    # Box plot of processing times
                    fig = go.Figure()
                    for metric, times in perf_data.items():
                        fig.add_trace(go.Box(y=times, name=metric))
                    fig.update_layout(
                        title="Processing Time Distribution",
                        yaxis_title="Time (ms)",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with perf_col2:
                    # Resource utilization over time
                    fig = go.Figure()
                    for metric, times in perf_data.items():
                        fig.add_trace(go.Scatter(y=times, name=metric, mode='lines'))
                    fig.update_layout(
                        title="Resource Utilization Over Time",
                        xaxis_title="Frame",
                        yaxis_title="Time (ms)",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.markdown("## 📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists(output_path):
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Processed Video", 
                        data=file, 
                        file_name="yoloscope_output.avi", 
                        mime="video/avi"
                    )
        
        with col2:
            if detection_history:
                detection_data = {
                    'detections': dict(detection_history),
                    'activities': activity_history,
                    'emotions': emotion_history,
                    'performance_metrics': {
                        'avg_fps': float(np.mean(list(performance_metrics['fps_history']))) if performance_metrics['fps_history'] else 0,
                        'total_frames': frame_num,
                        'processing_times': {
                            'mean_total': float(np.mean(list(performance_metrics['total_times']))) if performance_metrics['total_times'] else 0,
                            'mean_inference': float(np.mean(list(performance_metrics['inference_times']))) if performance_metrics['inference_times'] else 0
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="📊 Download Analytics Data",
                    data=json.dumps(detection_data, indent=2),
                    file_name="analytics_data.json",
                    mime="application/json"
                )
        
        with col3:
            if performance_metrics['total_times']:
                perf_df = pd.DataFrame({
                    'frame': range(len(performance_metrics['total_times'])),
                    'preprocessing_time': list(performance_metrics['preprocessing_times']),
                    'inference_time': list(performance_metrics['inference_times']),
                    'postprocessing_time': list(performance_metrics['postprocessing_times']),
                    'total_time': list(performance_metrics['total_times']),
                    'fps': list(performance_metrics['fps_history'])
                })
                perf_csv = perf_df.to_csv(index=False)
                st.download_button(
                    label="⚡ Download Performance Data",
                    data=perf_csv,
                    file_name="performance_data.csv",
                    mime="text/csv"
                )

# Training tab
with tab3:
    st.markdown("## 🎯 Custom Object Training")
    st.markdown("Upload images of objects you want to detect and train a custom YOLO model.")
    
    if name_label and image_files:
        st.info("🧠 Training functionality requires proper dataset annotation. Consider using tools like Roboflow for annotation.")
        
        # Display uploaded images
        cols = st.columns(min(len(image_files), 4))
        for i, img_file in enumerate(image_files[:4]):
            with cols[i]:
                img = Image.open(img_file)
                st.image(img, caption=f"{name_label}_{i}", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**YOLOScope Pro** - Powered by YOLOv8, MediaPipe, and Streamlit | Built with ❤️ for Computer Vision")