# 🛰️ YOLOScope - Real-Time Object Detection with YOLOv8

YOLOScope is a powerful Streamlit-based web application that enables **real-time object detection** using **YOLOv8** on both uploaded videos and YouTube video streams. It provides **live inference, pre-processing, and post-processing metrics**, all visualized with **Plotly** in a sleek dark-themed dashboard.

---

## 🚀 Features

- 🎥 Upload or stream YouTube videos for object detection
- 🧠 Selectable YOLOv8 model sizes (`n`, `s`, `m`)
- 🎯 Adjustable confidence threshold
- 📊 Real-time visualization of detection speed metrics (inference, pre-process, post-process)
- 🧾 Class-wise object count per frame
- 📈 Interactive Plotly charts in sidebar
- 💾 Downloadable output video with bounding boxes
- 🎨 Beautiful dark theme with custom styles

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [Plotly](https://plotly.com/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---

## 🖼️ UI Preview

> A clean, dark-themed interface with video preview and performance graphs on the sidebar.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/yoloscope.git
cd yoloscope

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt (example content)

```txt
streamlit
opencv-python
ultralytics
numpy
pandas
plotly
yt-dlp
```

---

## 🧪 How to Use

1. **Run the app:**
   ```bash
   streamlit run app.py
   ```

2. **Open Streamlit in your browser.**

3. **Choose a YOLOv8 model size and set the confidence threshold in the sidebar.**

4. **Either:**
   - Upload a `.mp4`, `.avi`, or `.mov` file
   - Paste a YouTube URL

5. **The app will:**
   - Perform object detection frame by frame
   - Display the video with detected objects live
   - Show live speed metrics in the sidebar

6. **Once complete:**
   - You can preview the full output video
   - You can download the annotated video

---

## 📊 Performance Metrics

Live graph in the sidebar showing:
- 🧠 **Inference Time** (ms)
- 🧹 **Preprocess Time** (ms)
- 📦 **Postprocess Time** (ms)

---

## 📌 Notes

- If FPS metadata is missing in the video, it defaults to 25 FPS.
- YouTube videos are streamed directly using `yt-dlp`—no need to download the video.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgments

- Thanks to [Ultralytics](https://ultralytics.com/) for YOLOv8
- Thanks to [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube streaming

---

## ✍️ Author

Made with ❤️ by **Nikhil**

