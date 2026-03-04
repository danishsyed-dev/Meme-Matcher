# Product Requirements Document (PRD)

## Meme Matcher v3.0 — Real-time Expression-to-Meme Matching

**Author:** Danish  
**Date:** March 4, 2026  
**Status:** Draft  

---

## 1. Executive Summary

Meme Matcher is a desktop application that uses computer vision to match a user's facial expressions and hand gestures to internet memes in real-time. Version 3.0 is a full architectural redesign of the existing monolithic v2.0 codebase, targeting modularity, reliability, performance, and an improved user experience.

---

## 2. Problem Statement

### Current State (v2.0)
The existing application works but suffers from several architectural and UX issues:

| Area | Problem |
|------|---------|
| **Architecture** | Entire app lives in a single `main.py` (~280 lines); no separation of concerns |
| **Thread Safety** | Video loop runs on a background thread with unsafe cross-thread UI updates |
| **Error Handling** | No graceful handling for missing camera, corrupt images, or model download failures |
| **Configuration** | All magic numbers (thresholds, weights, window sizes) are hard-coded |
| **Extensibility** | Adding new detection features or UI panels requires editing deeply coupled code |
| **Testing** | Zero test coverage; no way to test backend logic independently |
| **Packaging** | No `requirements.txt` or `pyproject.toml`; manual dependency management |
| **UX Polish** | Basic Tkinter styling; no loading states; no keyboard shortcuts; no status feedback |

### Desired State (v3.0)
A well-structured, modular desktop application with clear separation between detection engine, matching logic, and UI — easy to extend, test, and maintain.

---

## 3. Goals & Non-Goals

### Goals
1. **Modular Architecture** — Separate the project into distinct modules: detection, matching, UI, configuration, and utilities
2. **Robust Error Handling** — Graceful degradation when camera/models/assets are unavailable
3. **Configurable** — Externalize all tunable parameters into a config file
4. **Improved Matching** — Enhance the scoring algorithm with more facial expression features and better normalization
5. **Better UX** — Loading indicators, keyboard shortcuts, status bar, smoother frame rendering
6. **Developer Experience** — Proper packaging, dependency management, and project structure
7. **Thread Safety** — Use proper producer-consumer pattern for video frames with queue-based UI updates

### Non-Goals
- Web or mobile version (desktop Tkinter only)
- Cloud/API integration
- Video recording (screenshot-only capture)
- Multi-user or networked features

---

## 4. Target Users

| Persona | Description |
|---------|-------------|
| **Casual User** | Wants a fun app to match their face to memes; expects simple launch-and-use experience |
| **Content Creator** | Uses the tool for generating meme content; needs screenshot export and custom meme uploads |
| **Developer/Contributor** | Wants to extend detection logic or add UI features; needs clean, modular code |

---

## 5. Functional Requirements

### FR-1: Detection Engine
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Detect facial landmarks using MediaPipe Face Landmarker (478 points) | P0 |
| FR-1.2 | Detect hand landmarks using MediaPipe Hand Landmarker (up to 2 hands) | P0 |
| FR-1.3 | Extract features: eye openness (EAR), mouth aspect ratio, eyebrow height, head tilt | P0 |
| FR-1.4 | Compute composite expression scores: surprise, smile, anger, concern | P0 |
| FR-1.5 | Support both VIDEO and IMAGE running modes for live and static analysis | P0 |
| FR-1.6 | Auto-download models on first run with progress reporting | P1 |

### FR-2: Matching Engine
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Pre-analyze all meme images at startup and cache feature vectors | P0 |
| FR-2.2 | Match user features to meme features using weighted similarity scoring | P0 |
| FR-2.3 | Return top-N matches with confidence scores (not just top-1) | P1 |
| FR-2.4 | Support configurable feature weights via config file | P1 |
| FR-2.5 | Implement match debouncing to prevent rapid flickering between memes | P1 |

### FR-3: Camera Manager
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Open and manage webcam capture in a dedicated thread | P0 |
| FR-3.2 | Use a thread-safe queue to pass frames to the main thread | P0 |
| FR-3.3 | Detect camera availability and show user-friendly error if unavailable | P0 |
| FR-3.4 | Support configurable resolution and FPS cap | P1 |
| FR-3.5 | Clean resource release on shutdown | P0 |

### FR-4: User Interface
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Split-view layout: live camera feed (left) + matched meme (right) | P0 |
| FR-4.2 | Control panel with Save Screenshot, Upload Meme, Reload Assets buttons | P0 |
| FR-4.3 | Status bar showing FPS, match status, and camera state | P1 |
| FR-4.4 | Match confidence progress bar | P0 |
| FR-4.5 | Scrollable meme gallery in the sidebar | P0 |
| FR-4.6 | Expression history line graph | P1 |
| FR-4.7 | Keyboard shortcuts: Space (screenshot), R (reload), Esc (quit) | P1 |
| FR-4.8 | Loading splash screen while models initialize | P1 |

### FR-5: Asset Management
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Load meme images from configurable `assets/` directory | P0 |
| FR-5.2 | Support JPG, PNG, JPEG formats | P0 |
| FR-5.3 | Allow uploading new meme images via file dialog (copied to assets folder) | P0 |
| FR-5.4 | Hot-reload memes without restarting the app | P0 |

---

## 6. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Frame processing latency | < 50ms per frame on modern hardware |
| NFR-2 | Application startup time | < 5 seconds (excluding first-time model download) |
| NFR-3 | Memory usage | < 500MB RSS |
| NFR-4 | Python version support | 3.10+ |
| NFR-5 | OS support | Windows 10+, macOS 12+, Ubuntu 22.04+ |

---

## 7. Architecture (v3.0 Redesign)

### 7.1 Project Structure
```
make_me_a_meme/
├── main.py                     # Entry point
├── config.yaml                 # User-configurable settings
├── requirements.txt            # Pinned dependencies
├── README.md                   # Documentation
├── PRD.md                      # This document
├── face_landmarker.task        # MediaPipe face model
├── hand_landmarker.task        # MediaPipe hand model
├── assets/                     # Meme images
│   ├── angry_baby.jpg
│   ├── disaster_girl.jpg
│   └── ...
└── src/
    ├── __init__.py
    ├── app.py                  # Application controller (orchestrates everything)
    ├── config.py               # Config loader (reads config.yaml)
    ├── detection/
    │   ├── __init__.py
    │   ├── face_detector.py    # Face landmark detection + feature extraction
    │   ├── hand_detector.py    # Hand landmark detection
    │   └── feature_extractor.py# Composite feature computation
    ├── matching/
    │   ├── __init__.py
    │   └── matcher.py          # Similarity scoring + match selection
    ├── camera/
    │   ├── __init__.py
    │   └── camera_manager.py   # Thread-safe camera capture
    ├── ui/
    │   ├── __init__.py
    │   ├── main_window.py      # Root window setup + layout
    │   ├── video_panel.py      # Camera feed display
    │   ├── control_panel.py    # Side panel (buttons, gallery, stats)
    │   ├── status_bar.py       # Bottom status bar
    │   └── widgets.py          # Custom reusable widgets (ModernButton, etc.)
    └── utils/
        ├── __init__.py
        ├── image_utils.py      # Image resize, format conversion helpers
        └── model_downloader.py # Model download with progress
```

### 7.2 Data Flow
```
Camera Thread                    Main Thread
┌──────────┐                    ┌──────────────┐
│ Capture   │──── Frame ───────>│ Detection    │
│ Manager   │    (via Queue)    │ Engine       │
└──────────┘                    └──────┬───────┘
                                       │ Features
                                       ▼
                                ┌──────────────┐
                                │ Matching     │
                                │ Engine       │
                                └──────┬───────┘
                                       │ Match Result
                                       ▼
                                ┌──────────────┐
                                │ UI Update    │
                                │ (Tkinter)    │
                                └──────────────┘
```

### 7.3 Key Design Decisions
1. **Queue-based frame passing**: Camera thread puts frames into a `queue.Queue(maxsize=2)`; main thread polls via `root.after()` — no cross-thread Tkinter calls
2. **Config-driven**: All thresholds, weights, UI dimensions read from `config.yaml` with sensible defaults
3. **Detector abstraction**: Face and hand detectors are independent classes; feature extractor combines their results
4. **Match debouncing**: Matcher holds the current match for a configurable number of frames before switching, preventing flickering

---

## 8. Configuration Schema (`config.yaml`)

```yaml
camera:
  device_index: 0
  width: 640
  height: 480
  fps_cap: 30

detection:
  face_confidence: 0.5
  hand_confidence: 0.3
  tracking_confidence: 0.5
  max_hands: 2

matching:
  weights:
    surprise_score: 20
    mouth_openness: 20
    hand_raised: 20
    eye_openness: 15
  debounce_frames: 5
  decay_factor: 5.0

ui:
  window_width: 1400
  window_height: 900
  theme: dark
  display_height: 600

assets:
  folder: assets
  supported_formats: [".jpg", ".png", ".jpeg"]
```

---

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Code modules | >= 10 separate files (vs 1 today) |
| Avg frame rate | >= 20 FPS |
| Startup time | < 5s after first run |
| Crash rate | 0 crashes from missing camera or assets |

---

## 10. Milestones

| Phase | Deliverable | 
|-------|-------------|
| Phase 1 | Project scaffolding, config system, model downloader |
| Phase 2 | Detection module (face + hand + features) |
| Phase 3 | Matching engine with debouncing |
| Phase 4 | Camera manager with thread-safe queue |
| Phase 5 | UI redesign (all panels) |
| Phase 6 | Integration, polish, README update |

---

## 11. Open Questions

1. Should we add meme metadata (JSON sidecar) to allow hand-labeling expression tags on memes?
2. Should we consider CustomTkinter for a more modern look instead of stock Tkinter + ttk?
3. Is there value in a "training mode" where users can associate their expressions with specific memes?

---

*End of PRD*
