# 🎭 Meme Matcher v3.0 — Real-time Facial Expression to Meme Matching

A real-time computer vision application that matches your facial expressions and hand gestures to iconic internet memes using **MediaPipe's AI-powered face and hand detection**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ What's New in v3.0

This release is a **complete architectural redesign** of the v2.0 monolith:

| Feature | Description |
|---------|-------------|
| 🏗️ **Modular Architecture** | 15+ files across 6 sub-packages — easy to extend and test |
| ⚙️ **Config-Driven** | All thresholds, weights, and UI settings in `config.yaml` |
| 🧵 **Thread-Safe Camera** | Queue-based frame passing — no more cross-thread UI calls |
| 🛡️ **Error Handling** | Graceful degradation for missing camera, models, or assets |
| 🔀 **Match Debouncing** | Configurable frame threshold before switching matched meme |
| 📊 **Status Bar** | Live FPS counter, camera state, meme count, keyboard shortcuts |
| ⌨️ **Keyboard Shortcuts** | Space (screenshot), R (reload), Esc (quit) |
| 🎨 **Improved Widgets** | Rounded buttons, better scrollable gallery |

---

## 🚀 Features

### Core Capabilities
- **Real-time Face Detection**: Uses MediaPipe Face Landmarker to track 478 facial landmarks
- **Hand Gesture Detection**: Tracks hand positions to distinguish similar expressions
- **Advanced Expression Analysis**:
  - Eye openness (surprise, wide eyes)
  - Eyebrow position (raised, furrowed)
  - Mouth shape (smiling, open, concerned)
  - Hand gestures (raised hands, fist pumps)
- **Smart Matching Algorithm**: Weighted similarity scoring with exponential decay

### UI Features
- **Split View**: Live camera feed + matched meme side-by-side
- **Control Center Panel**: All controls in one place
- **Match Score Bar**: Visual indicator of match confidence
- **Scrollable Meme Gallery**: See all loaded memes at a glance

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Webcam

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/danishsyed-dev/meme-matcher.git
cd meme-matcher

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python main.py
```

> 💡 **Note**: The first run will automatically download MediaPipe models (~11MB total).

---

## 🎮 How to Use

1. **Launch** the app with `python main.py`
2. **Your webcam activates** and the GUI opens
3. **Make expressions** — the app matches you to memes in real-time:

| Expression | Meme Match |
|------------|------------|
| 😠 Angry face | Angry Baby |
| 😏 Smirk (no hands) | Disaster Girl |
| 🤔 Smirk + hand on chin | Gene Wilder |
| 😄 Smile + raised hand | Leonardo DiCaprio |
| 👀 Wide eyes/staring | Overly Attached Girlfriend |
| 💪 Happy + fist pump | Success Kid |

4. **Use the Control Panel**:
   - 📷 **Save Screenshot**: Capture the current frame
   - 📂 **Upload Custom Meme**: Add your own `.jpg`/`.png` memes
   - 🔄 **Reload Assets**: Refresh the meme database

---

## 🧠 How It Works

### 1. Face & Hand Detection
- **MediaPipe Face Landmarker**: 478 landmarks per face
- **MediaPipe Hand Landmarker**: 21 landmarks per hand (up to 2 hands)

### 2. Feature Extraction
For each frame, the app calculates:
- **Eye features**: Openness, symmetry
- **Eyebrow features**: Height relative to eyes
- **Mouth features**: Openness, width ratio
- **Hand features**: Count, raised/lowered position
- **Composite scores**: Surprise, smile, concern, cheers

### 3. Similarity Matching
- Compares live features against pre-loaded meme features
- Uses weighted exponential decay scoring
- Returns the best match with a confidence score

---

## 📁 Project Structure

```
meme-matcher/
├── main.py                         # Entry point
├── config.yaml                     # User-configurable settings
├── requirements.txt                # Pinned dependencies
├── PRD.md                          # Product Requirements Document
├── README.md
├── face_landmarker.task            # MediaPipe face model (auto-downloaded)
├── hand_landmarker.task            # MediaPipe hand model (auto-downloaded)
├── assets/                         # Meme images folder
│   ├── angry_baby.jpg
│   ├── disaster_girl.jpg
│   └── ...
└── src/
    ├── __init__.py
    ├── app.py                      # Application controller
    ├── config.py                   # Config loader (reads config.yaml)
    ├── detection/
    │   ├── __init__.py
    │   ├── face_detector.py        # Face landmark detection
    │   ├── hand_detector.py        # Hand landmark detection
    │   └── feature_extractor.py    # Composite feature computation
    ├── matching/
    │   ├── __init__.py
    │   └── matcher.py              # Similarity scoring + debouncing
    ├── camera/
    │   ├── __init__.py
    │   └── camera_manager.py       # Thread-safe camera capture
    ├── ui/
    │   ├── __init__.py
    │   ├── main_window.py          # Root window + layout
    │   ├── video_panel.py          # Camera feed display
    │   ├── control_panel.py        # Side panel (buttons, gallery, stats)
    │   ├── status_bar.py           # Bottom status bar
    │   └── widgets.py              # Custom reusable widgets
    └── utils/
        ├── __init__.py
        ├── image_utils.py          # Image processing helpers
        └── model_downloader.py     # Model download with fallback
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🎨 Add more memes (with distinctive expressions)
- 🧪 Improve the matching algorithm
- 🖌️ Enhance the UI/UX
- 📱 Multi-face support
- ⚡ Performance optimizations

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

- **[MediaPipe](https://developers.google.com/mediapipe)** — Google's ML framework for face and hand detection
- **[OpenCV](https://opencv.org/)** — Open source computer vision library
- **[Pillow](https://pillow.readthedocs.io/)** — Python imaging library
- **Meme Images** — Fair use, iconic internet memes

---

## 📬 Contact

Made with ❤️ by **Danish**

---

<p align="center">
  <i>Point your camera. Make a face. Become a meme.</i>
</p>
