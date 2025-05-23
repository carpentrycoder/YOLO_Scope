import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import time
from collections import defaultdict
import yt_dlp

st.set_page_config(page_title="YOLOScope", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #0f0f0f;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            color: white;
            background-color: #222222;
            border-radius: 10px;
        }
        .stProgress > div > div {
            background-color: #888 !important;
        }
        .plotly-container {
          background-color: #2a2a2a;
          border-radius: 12px;
          padding: 20px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.6);
          margin-bottom: 20px;
          border: 1px solid #444444;
        }
        .plotly-container .js-plotly-plot {
          background-color: #1e1e1e !important;
          border-radius: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_size):
    return YOLO(f'yolov8{model_size}.pt')

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

# Sidebar controls
st.sidebar.title("⚙️ Settings")
model_size = st.sidebar.selectbox("YOLO Model Size", options=["n", "s", "m"], index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.95, 0.5, 0.05)
model = load_model(model_size)

st.title("YOLOScope -🎥 YOLOv8 Object Detection")
st.markdown("Upload a video or enter a YouTube URL to detect objects in real-time with performance visualization.")

uploaded_file = st.file_uploader("📁 Upload your video", type=["mp4", "avi", "mov"])
st.markdown("**OR**")
youtube_url = st.text_input("📺 Enter YouTube Video URL")

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

# Proceed if we have a valid video path or stream
if video_path is not None:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Failed to open video stream.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25  # fallback
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        stframe = st.empty()
        st.sidebar.subheader("📊 Live Metrics")
        plot_area = st.sidebar.empty()

        inference_speeds = []
        preprocess_times = []
        postprocess_times = []
        class_counts_series = []

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_pre = time.time()
            frame = cv2.resize(frame, (width, height))
            end_pre = time.time()

            start_inf = time.time()
            results = model(frame, conf=confidence_threshold)
            end_inf = time.time()

            start_post = time.time()
            annotated = results[0].plot()
            out.write(annotated)
            end_post = time.time()

            stframe.image(annotated, channels="BGR", use_container_width=True)

            class_count = defaultdict(int)
            for box in results[0].boxes.data:
                cls = int(box[-1])
                class_name = model.model.names[cls]
                class_count[class_name] += 1

            class_counts_series.append(class_count)
            inference_speeds.append((end_inf - start_inf) * 1000)
            preprocess_times.append((end_pre - start_pre) * 1000)
            postprocess_times.append((end_post - start_post) * 1000)

            if frame_num % 3 == 0:
                df = pd.DataFrame({
                    "Inference": inference_speeds,
                    "Preprocess": preprocess_times,
                    "Postprocess": postprocess_times
                })
                fig = go.Figure()
                for col in df.columns:
                    fig.add_trace(go.Scatter(y=df[col], name=col, mode='lines+markers'))
                fig.update_layout(height=300, plot_bgcolor='#0f0f0f', paper_bgcolor='#0f0f0f',
                                  font_color='white', title="Speed Metrics (ms)")
                plot_area.plotly_chart(fig, use_container_width=True)

            frame_num += 1

        cap.release()
        out.release()

        st.success("✅ Detection Complete!")
        st.video(output_path)
        with open(output_path, "rb") as file:
            st.download_button(label="⬇️ Download Output Video", data=file, file_name="output_yolo.avi", mime="video/avi")
