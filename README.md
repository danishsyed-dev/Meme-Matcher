# 🎭 Meme Matcher v2.0 — Real-time Facial Expression to Meme Matching

A real-time computer vision application that matches your facial expressions and hand gestures to iconic internet memes using **MediaPipe's AI-powered face and hand detection**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ What's New in v2.0

This release is a **complete overhaul** with a modern GUI and powerful new features:

| Feature | Description |
|---------|-------------|
| 🖥️ **Modern Dark-Mode GUI** | A sleek, responsive desktop application built with Tkinter |
| 📷 **Screenshot Capture** | Save your best meme matches with one click |
| 📂 **Custom Meme Upload** | Add your own memes directly from the app |
| 📈 **Expression History Graph** | Live visualization of your match scores over time |
| 🔄 **Hot Reload** | Reload memes without restarting the application |
| 🎨 **Meme Gallery** | Browse all active memes in the sidebar |

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
git clone https://github.com/YOUR_USERNAME/meme-matcher.git
cd meme-matcher

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install mediapipe opencv-python numpy pillow

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
├── main.py              # Main application (GUI + Backend)
├── assets/              # Meme images folder
│   ├── angry_baby.jpg
│   ├── disaster_girl.jpg
│   ├── gene_wilder.jpg
│   ├── leonardo_dicaprio.jpg
│   ├── overly_attached_girlfriend.jpg
│   └── success_kid.jpg
├── .gitignore
└── README.md
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
