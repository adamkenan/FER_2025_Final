import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import time
import os

from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH      = "C:/adam/AMIT_Diploma/grad_project/FER_2025/checkpoints/best_model.pth"
IMG_SIZE        = 224
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FACE_CASCADE    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
CAM_W, CAM_H    = 640, 480
INFER_INTERVAL  = 0.12   # seconds between inference calls

EMOTION_COLORS = {
    "angry":    ("#e05252", (82,  82, 224)),
    "disgust":  ("#a0c84b", (75, 200, 160)),
    "fear":     ("#9b59b6", (182, 89, 155)),
    "happy":    ("#f0c040", (64, 192, 240)),
    "neutral":  ("#78909c", (156,144,120)),
    "sad":      ("#4fc3f7", (247,195, 79)),
    "surprise": ("#ff7043", (67,112,255)),
}

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes, dropout_rate=0.3):
    weights = models.ResNet34_Weights.IMAGENET1K_V1
    model   = models.resnet34(weights=weights)
    in_f    = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(in_f, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),
    )
    return model

def load_model(path):
    ck           = torch.load(path, map_location=DEVICE)
    class_names  = ck.get("class_names", DEFAULT_CLASSES)
    dropout_rate = ck.get("config", {}).get("dropout", 0.3)
    model = build_model(len(class_names), dropout_rate)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE).eval()
    return model, class_names

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(model, class_names, pil_image):
    t = val_transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0]
    idx       = probs.argmax().item()
    all_probs = {class_names[i]: probs[i].item() * 100 for i in range(len(class_names))}
    return class_names[idx], probs[idx].item() * 100, all_probs


# ── App ───────────────────────────────────────────────────────────────────────
class FerApp:
    def __init__(self, root):
        self.root        = root
        self.model       = None
        self.class_names = []
        self.cap         = None
        self.running     = False

        # Camera thread -> main thread: raw BGR numpy frames
        self.frame_q = queue.Queue(maxsize=2)

        # Shared prediction state (written only by inference thread)
        self._pred_lock  = threading.Lock()
        self._last_label = "---"
        self._last_conf  = 0.0
        self._last_probs = {}

        self._current_photo   = None
        self._canvas_image_id = None

        root.title("FER - Live Emotion Detector")
        root.configure(bg="#0d0d0f")
        root.resizable(False, False)

        self._build_ui()
        self._try_autoload()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        header = tk.Frame(root, bg="#0d0d0f")
        header.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(header, text="EMOTION DETECTOR",
                 font=("Courier New", 15, "bold"),
                 bg="#0d0d0f", fg="#e8e8e8").pack(side="left")
        self.status_dot = tk.Label(header, text="*", font=("Courier New", 12),
                                   bg="#0d0d0f", fg="#444")
        self.status_dot.pack(side="right", padx=(0, 4))
        self.status_var = tk.StringVar(value="no model loaded")
        tk.Label(header, textvariable=self.status_var,
                 font=("Courier New", 8), bg="#0d0d0f", fg="#555").pack(side="right", padx=6)

        cam_frame = tk.Frame(root, bg="#0d0d0f", padx=20, pady=10)
        cam_frame.pack()
        self.canvas = tk.Canvas(cam_frame, width=CAM_W, height=CAM_H,
                                bg="#111114", highlightthickness=1,
                                highlightbackground="#2a2a30")
        self.canvas.pack()
        self.canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="[ webcam feed will appear here ]",
            fill="#2a2a30", font=("Courier New", 13),
            tags="placeholder"
        )

        ctrl = tk.Frame(root, bg="#0d0d0f")
        ctrl.pack(padx=20, pady=(0, 6))
        btn_cfg = dict(font=("Courier New", 10), relief="flat",
                       padx=12, pady=5, cursor="hand2")
        self.btn_model = tk.Button(ctrl, text="[ LOAD MODEL ]",
                                   command=self.load_model_dialog,
                                   bg="#1e1e24", fg="#888", **btn_cfg)
        self.btn_model.pack(side="left", padx=(0, 8))
        self.btn_cam = tk.Button(ctrl, text="[ START CAMERA ]",
                                 command=self.toggle_camera,
                                 bg="#1e1e24", fg="#888", **btn_cfg)
        self.btn_cam.pack(side="left")

        readout = tk.Frame(root, bg="#0d0d0f")
        readout.pack(padx=20, pady=(4, 0), fill="x")

        left_col = tk.Frame(readout, bg="#0d0d0f")
        left_col.pack(side="left", fill="y", padx=(0, 20))
        tk.Label(left_col, text="DETECTED EMOTION",
                 font=("Courier New", 7), bg="#0d0d0f", fg="#444").pack(anchor="w")
        self.result_var = tk.StringVar(value="---")
        self.result_lbl = tk.Label(left_col, textvariable=self.result_var,
                                   font=("Courier New", 30, "bold"),
                                   bg="#0d0d0f", fg="#f0c040")
        self.result_lbl.pack(anchor="w")
        self.conf_var = tk.StringVar(value="")
        tk.Label(left_col, textvariable=self.conf_var,
                 font=("Courier New", 10), bg="#0d0d0f", fg="#555").pack(anchor="w")

        right_col = tk.Frame(readout, bg="#0d0d0f")
        right_col.pack(side="left", fill="both", expand=True)
        tk.Label(right_col, text="ALL EMOTIONS",
                 font=("Courier New", 7), bg="#0d0d0f", fg="#444").pack(anchor="w")

        self._bar_vars = {}
        BAR_W = 280
        for emo in DEFAULT_CLASSES:
            row = tk.Frame(right_col, bg="#0d0d0f")
            row.pack(fill="x", pady=1)
            hex_color = EMOTION_COLORS.get(emo, ("#78909c", None))[0]
            tk.Label(row, text=f"{emo[:7]:<7}", width=7,
                     font=("Courier New", 8), bg="#0d0d0f", fg="#666",
                     anchor="w").pack(side="left")
            bc = tk.Canvas(row, width=BAR_W, height=10,
                           bg="#16161a", highlightthickness=0)
            bc.pack(side="left", padx=(4, 6))
            bc.create_rectangle(0, 0, 0, 10, fill=hex_color, outline="", tags="bar")
            pct_var = tk.StringVar(value="  0.0%")
            tk.Label(row, textvariable=pct_var, font=("Courier New", 8),
                     bg="#0d0d0f", fg="#555", width=6).pack(side="left")
            self._bar_vars[emo] = (bc, pct_var, BAR_W)

        tk.Frame(root, bg="#0d0d0f", height=12).pack()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _try_autoload(self):
        if os.path.exists(MODEL_PATH):
            self._do_load(MODEL_PATH)

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self._do_load(path)

    def _do_load(self, path):
        try:
            self.model, self.class_names = load_model(path)
            self.status_var.set(f"OK {os.path.basename(path)}  |  {', '.join(self.class_names)}")
            self.status_dot.config(fg="#4caf50")
            self.btn_model.config(fg="#4caf50")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.status_var.set("model failed to load")
            self.status_dot.config(fg="#e05252")

    # ── Camera toggle ─────────────────────────────────────────────────────────
    def toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a model first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Cannot open webcam (index 0).")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.running = True
        self.btn_cam.config(text="[ STOP CAMERA ]", fg="#e05252")
        self.status_dot.config(fg="#f0c040")

        threading.Thread(target=self._camera_thread,   daemon=True).start()
        threading.Thread(target=self._inference_thread, daemon=True).start()
        self._poll_frames()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.config(text="[ START CAMERA ]", fg="#888")
        self.status_dot.config(fg="#4caf50" if self.model else "#444")
        self.result_var.set("---")
        self.result_lbl.config(fg="#f0c040")
        self.conf_var.set("")
        self.canvas.delete("all")
        self._canvas_image_id = None
        self.canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="[ webcam feed will appear here ]",
            fill="#2a2a30", font=("Courier New", 13),
            tags="placeholder"
        )
        for emo, (bc, pv, bw) in self._bar_vars.items():
            bc.coords("bar", 0, 0, 0, 10)
            pv.set("  0.0%")

    # ── Camera thread: ONLY reads frames, puts raw BGR arrays in queue ─────────
    def _camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass
        self.running = False

    # ── Inference thread: face detect + model, writes to shared state ─────────
    def _inference_thread(self):
        last_infer = 0.0
        while self.running:
            now = time.time()
            if now - last_infer < INFER_INTERVAL:
                time.sleep(0.01)
                continue

            # Peek at latest frame without consuming it
            try:
                with self.frame_q.mutex:
                    if not self.frame_q.queue:
                        time.sleep(0.01)
                        continue
                    frame = list(self.frame_q.queue)[-1]
            except Exception:
                time.sleep(0.01)
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                time.sleep(0.01)
                continue

            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_rgb    = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            face_pil    = Image.fromarray(face_rgb)

            try:
                label, conf, all_probs = predict(self.model, self.class_names, face_pil)
                with self._pred_lock:
                    self._last_label = label
                    self._last_conf  = conf
                    self._last_probs = all_probs
            except Exception:
                pass

            last_infer = time.time()

    # ── Main-thread poll: dequeue frame, annotate, convert, draw ─────────────
    def _poll_frames(self):
        if not self.running:
            return

        try:
            frame = self.frame_q.get_nowait()
        except queue.Empty:
            self.root.after(15, self._poll_frames)
            return

        with self._pred_lock:
            label     = self._last_label
            conf      = self._last_conf
            all_probs = dict(self._last_probs)

        # Face detection for bounding box overlay (fast, ok on main thread)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )
        annotated = self._annotate(frame, faces, label, conf)

        # ImageTk.PhotoImage MUST be created on the main thread
        rgb   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            self.canvas.itemconfig(self._canvas_image_id, image=photo)
        self._current_photo = photo   # keep reference to prevent GC

        # Update readout
        if label != "---":
            hex_color = EMOTION_COLORS.get(label.lower(), ("#f0c040", None))[0]
            self.result_var.set(label.upper())
            self.result_lbl.config(fg=hex_color)
            self.conf_var.set(f"confidence  {conf:.1f}%")

        for emo, (bc, pv, bw) in self._bar_vars.items():
            pct = all_probs.get(emo, 0.0)
            bc.coords("bar", 0, 0, int(bw * pct / 100), 10)
            pv.set(f"{pct:5.1f}%")

        self.root.after(15, self._poll_frames)

    # ── Annotation (pure OpenCV, no Tk) ───────────────────────────────────────
    def _annotate(self, frame, faces, label, conf):
        color = EMOTION_COLORS.get(label.lower(), ("#78909c", (156, 144, 120)))[1]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), color, 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if len(faces) > 0 and label != "---":
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            tag = f"{label.upper()}  {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ty = max(y - 8, th + 4)
            cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 8, ty + 2), color, -1)
            cv2.putText(frame, tag, (x + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(frame, f"device: {DEVICE}", (8, CAM_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        return frame

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = FerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.update_idletasks()
    W, H   = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    root.mainloop()
