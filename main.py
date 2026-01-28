import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import datetime
import shutil

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=40, bg_color="#333333", hover_color="#444444", text_color="white"):
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        
        self.create_text(width/2, height/2, text=text, fill=text_color, font=("Segoe UI", 10, "bold"))
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def on_enter(self, e):
        self.configure(bg=self.hover_color)
        
    def on_leave(self, e):
        self.configure(bg=self.bg_color)
        
    def on_click(self, e):
        self.command()

class MemeMatcherBackend:
    def __init__(self, assets_folder="assets"):
        self.assets_folder = assets_folder
        self.memes = []
        self.meme_features = []
        
        # Initialize MediaPipe
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._download_face_model()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._download_hand_model()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
        )
        
        # Defining indices (Same as before)
        self.LEFT_EYE_UPPER = [159, 145, 158]
        self.LEFT_EYE_LOWER = [23, 27, 133]
        self.RIGHT_EYE_UPPER = [386, 374, 385]
        self.RIGHT_EYE_LOWER = [253, 257, 362]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336]
        self.MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
        self.MOUTH_INNER = [78, 308, 95, 88]
        self.NOSE_TIP = 4
        self.frame_counter = 0

        self.reload_memes()

    def _download_face_model(self):
        import os, subprocess
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
            except: pass # Assume exists or fail gracefully
        return model_path

    def _download_hand_model(self):
        import os, subprocess
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
            except: pass
        return model_path

    def reload_memes(self):
        self.memes = []
        self.meme_features = []
        assets_path = Path(self.assets_folder)
        if not assets_path.exists():
            assets_path.mkdir()
            
        static_face = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._download_face_model()),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1
            )
        )
        static_hand = mp.tasks.vision.HandLandmarker.create_from_options(
             mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._download_hand_model()),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2
            )
        )

        files = list(assets_path.glob("*.jpg")) + list(assets_path.glob("*.png")) + list(assets_path.glob("*.jpeg"))
        
        for f in sorted(files):
            img = cv2.imread(str(f))
            if img is not None:
                feats = self.extract_features(img, static_face, static_hand, is_static=True)
                if feats:
                    self.memes.append({'image': img, 'name': f.stem.replace('_', ' ').title(), 'path': str(f)})
                    self.meme_features.append(feats)
        
        static_face.close()
        static_hand.close()
        return len(self.memes)

    def extract_features(self, image, face_model=None, hand_model=None, is_static=False):
        if face_model is None: face_model = self.face_mesh
        if hand_model is None: hand_model = self.hand_detector
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        if is_static:
            face_res = face_model.detect(mp_img)
            hand_res = hand_model.detect(mp_img)
        else:
            self.frame_counter += 1
            face_res = face_model.detect_for_video(mp_img, self.frame_counter)
            hand_res = hand_model.detect_for_video(mp_img, self.frame_counter)
            
        if not face_res.face_landmarks: return None
        
        # Simplify feature extraction for brevity, preserving core logic
        landmarks = face_res.face_landmarks[0]
        def get_pt(idx): return np.array([landmarks[idx].x, landmarks[idx].y])
        
        # EAR
        def ear(upper, lower):
            u_pts = [get_pt(i) for i in upper]
            l_pts = [get_pt(i) for i in lower]
            v = sum([np.linalg.norm(u_pts[i]-l_pts[i]) for i in range(len(u_pts))])/len(u_pts)
            h = np.linalg.norm(get_pt(upper[0]) - get_pt(upper[-1]))
            return v / (h + 1e-6)
            
        left_ear = ear(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER)
        right_ear = ear(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear)/2.0
        
        # Mouth
        m_h = np.linalg.norm(get_pt(13) - get_pt(14))
        m_w = np.linalg.norm(get_pt(61) - get_pt(291))
        m_ar = m_h / (m_w + 1e-6)
        
        # Brows
        l_brow_y = np.mean([get_pt(i)[1] for i in self.LEFT_EYEBROW])
        r_brow_y = np.mean([get_pt(i)[1] for i in self.RIGHT_EYEBROW])
        l_eye_cy = np.mean([get_pt(i)[1] for i in self.LEFT_EYE_UPPER+self.LEFT_EYE_LOWER])
        r_eye_cy = np.mean([get_pt(i)[1] for i in self.RIGHT_EYE_UPPER+self.RIGHT_EYE_LOWER])
        avg_brow_h = ((l_eye_cy - l_brow_y) + (r_eye_cy - r_brow_y)) / 2.0
        
        # Hands
        num_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
        hand_raised = False
        if num_hands > 0:
            face_top = min([l.y for l in landmarks])
            for h_marks in hand_res.hand_landmarks:
                if h_marks[0].y < face_top + 0.3: hand_raised = True; break

        # Return dict
        return {
            'eye_openness': avg_ear,
            'mouth_openness': m_ar,
            'eyebrow_height': avg_brow_h,
            'hand_raised': 1.0 if hand_raised else 0.0,
            'surprise_score': avg_ear * avg_brow_h * m_ar,
            'smile_score': (1.0 - m_ar), # Simple
            'num_hands': num_hands
        }

    def find_match(self, user_feats):
        if not user_feats: return None, 0.0
        best_score = -1.0
        best_meme = None
        
        weights = {'surprise_score': 20,'mouth_openness': 20,'hand_raised': 20, 'eye_openness':15}
        
        for i, meme_feats in enumerate(self.meme_features):
            score = 0
            for k, w in weights.items():
                if k in user_feats and k in meme_feats:
                    diff = abs(user_feats[k] - meme_feats[k])
                    score += w * np.exp(-diff * 5)
            if score > best_score:
                best_score = score
                best_meme = self.memes[i]
        return best_meme, best_score

class MemeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meme Matcher Ultimate v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1E1E1E")
        
        self.backend = MemeMatcherBackend()
        self.cap = None
        self.is_running = False
        self.current_match = None
        self.history = []
        
        self._setup_ui()
        self.start_camera()

    def _setup_ui(self):
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Dark.TFrame", background="#1E1E1E")
        style.configure("Dark.TLabel", background="#1E1E1E", foreground="white", font=("Segoe UI", 12))
        
        # Main Layout
        main_container = tk.Frame(self.root, bg="#1E1E1E")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left Panel (Video)
        left_panel = tk.Frame(main_container, bg="black", highlightbackground="#333", highlightthickness=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right Panel (Controls & Info)
        right_panel = tk.Frame(main_container, bg="#252526", width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)
        
        # Header
        tk.Label(right_panel, text="control_center", bg="#252526", fg="#007ACC", font=("Consolas", 16, "bold")).pack(pady=20)
        
        # Match Info
        self.match_label = tk.Label(right_panel, text="No Match", bg="#252526", fg="white", font=("Segoe UI", 14, "bold"))
        self.match_label.pack(pady=10)
        
        self.score_bar = ttk.Progressbar(right_panel, orient="horizontal", length=300, mode="determinate")
        self.score_bar.pack(pady=5)
        
        # Stats Canvas
        tk.Label(right_panel, text="Expression History", bg="#252526", fg="#888").pack(pady=(20,5))
        self.stats_canvas = tk.Canvas(right_panel, width=350, height=100, bg="#333", highlightthickness=0)
        self.stats_canvas.pack(pady=5)
        
        # Controls
        btn_frame = tk.Frame(right_panel, bg="#252526")
        btn_frame.pack(pady=30, fill=tk.X, padx=20)
        
        ModernButton(btn_frame, "📷 Save Screenshot", self.save_screenshot).pack(pady=5, fill=tk.X)
        ModernButton(btn_frame, "📂 Upload Custom Meme", self.upload_meme).pack(pady=5, fill=tk.X)
        ModernButton(btn_frame, "🔄 Reload Assets", self.reload_assets).pack(pady=5, fill=tk.X)
        
        # Gallery
        tk.Label(right_panel, text="Active Memes", bg="#252526", fg="#888").pack(pady=(20,10))
        self.gallery_frame = tk.Frame(right_panel, bg="#252526")
        self.gallery_frame.pack(fill=tk.BOTH, expand=True)
        
        self.update_gallery()

    def update_gallery(self):
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
            
        canvas = tk.Canvas(self.gallery_frame, bg="#252526", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg="#252526")
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for meme in self.backend.memes:
            f = tk.Frame(scrollable, bg="#333", pady=5, padx=5)
            f.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(f, text=meme['name'], bg="#333", fg="white", font=("Segoe UI", 9)).pack(anchor="w")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            user_feats = self.backend.extract_features(frame)
            best_meme, score = self.backend.find_match(user_feats)
            
            # Update UI Data
            if best_meme:
                self.current_match = (best_meme, score)
                # GUI updates must be on main thread logic, but we'll use after() or simple var updates
                # For complex UI updates from thread, usually queue or after is best.
                # Here we will do minimal processing and render the combined frame
                
                # Resize meme to match height
                h, w = frame.shape[:2]
                meme_img = best_meme['image']
                scale = h / meme_img.shape[0]
                new_w = int(meme_img.shape[1] * scale)
                meme_resized = cv2.resize(meme_img, (new_w, h))
                
                combined = np.zeros((h, w + new_w, 3), dtype=np.uint8)
                combined[:, :w] = frame
                combined[:, w:] = meme_resized
                
                # Draw separating line
                cv2.line(combined, (w, 0), (w, h), (0, 255, 255), 2)
                
                final_frame = combined
            else:
                self.current_match = None
                final_frame = frame

            # Convert to RGB for PIL
            cv2_img = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            
            # Resize for specific UI fit if needed (optional)
            # Keeping aspect ratio
            display_h = 600
            aspect = img.width / img.height
            display_w = int(display_h * aspect)
            img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Schedule UI Update safely
            self.root.after(0, self.update_frame, imgtk, best_meme, score if best_meme else 0)
            self.last_frame = final_frame # Store for screenshot
            
            time.sleep(0.01)

    def update_frame(self, imgtk, match, score):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        if match:
            self.match_label.config(text=f"{match['name']} ({int(score)})")
            self.score_bar['value'] = min(score, 100)
            
            # Update Graph
            self.history.append(score)
            if len(self.history) > 50: self.history.pop(0)
            self.draw_stats()
        else:
            self.match_label.config(text="No Match")
            self.score_bar['value'] = 0

    def draw_stats(self):
        self.stats_canvas.delete("all")
        if not self.history: return
        
        w = 350
        h = 100
        step = w / 50
        
        points = []
        for i, val in enumerate(self.history):
            x = i * step
            y = h - (val / 100 * h) # Scale 0-100 to y
            points.append(x)
            points.append(y)
            
        if len(points) >= 4:
            self.stats_canvas.create_line(points, fill="#007ACC", width=2, smooth=True)

    def save_screenshot(self):
        if hasattr(self, 'last_frame'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.jpg"
            cv2.imwrite(path, self.last_frame)
            messagebox.showinfo("Saved", f"Screenshot saved to {path}")

    def upload_meme(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            dest = Path(self.backend.assets_folder) / Path(path).name
            shutil.copy(path, dest)
            self.reload_assets()

    def reload_assets(self):
        count = self.backend.reload_memes()
        self.update_gallery()
        messagebox.showinfo("Reloaded", f"Loaded {count} memes!")

    def on_close(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MemeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
