import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH      = "C:/adam/AMIT_Diploma/grad_project/FER_2025/checkpoints/best_model.pth"
IMG_SIZE        = 224
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
PREVIEW_PX      = 200   # preview box size in pixels

# ── Model builder ──────────────────────────────────────────────────────────────
def build_model(num_classes: int, dropout_rate: float = 0.3) -> nn.Module:
    weights = models.ResNet34_Weights.IMAGENET1K_V1
    model   = models.resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(in_features, 1024),
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

# ── Load checkpoint ────────────────────────────────────────────────────────────
def load_model(path: str):
    checkpoint   = torch.load(path, map_location=DEVICE)
    class_names  = checkpoint.get("class_names", DEFAULT_CLASSES)
    num_classes  = len(class_names)
    dropout_rate = checkpoint.get("config", {}).get("dropout", 0.3)
    model = build_model(num_classes=num_classes, dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

# ── Preprocessing ──────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, class_names, pil_image: Image.Image):
    tensor = val_transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return class_names[idx], probs[idx].item() * 100

# ── GUI ────────────────────────────────────────────────────────────────────────
class FerApp:
    def __init__(self, root: tk.Tk):
        self.root        = root
        self.model       = None
        self.class_names = []
        self.photo       = None

        root.title("Emotion Detector")
        root.configure(bg="#1a1a1a")
        root.resizable(False, False)

        PAD = dict(padx=16)

        # Title
        tk.Label(root, text="Facial Emotion Recognition",
                 font=("Courier", 14, "bold"),
                 bg="#1a1a1a", fg="#e0e0e0").pack(pady=(12, 2), **PAD)

        # Status
        self.status_var = tk.StringVar(value="No model loaded")
        tk.Label(root, textvariable=self.status_var,
                 font=("Courier", 8), bg="#1a1a1a", fg="#888",
                 wraplength=360).pack(pady=(0, 4), **PAD)

        # Load model button
        tk.Button(root, text="Load Model (.pth)",
                  command=self.load_model_dialog,
                  font=("Courier", 10), bg="#2e2e2e", fg="#ccc",
                  relief="flat", padx=10, pady=4).pack(pady=(0, 8), **PAD)

        # ── Image preview — use tk.Canvas so width/height are in PIXELS ──
        self.canvas = tk.Canvas(root, width=PREVIEW_PX, height=PREVIEW_PX,
                                bg="#111111", highlightthickness=0)
        self.canvas.pack(pady=(0, 8))

        # Choose & predict button
        tk.Button(root, text="Choose Image & Predict",
                  command=self.choose_and_predict,
                  font=("Courier", 11, "bold"), bg="#3a3a3a", fg="#ffffff",
                  relief="flat", padx=14, pady=6).pack(pady=(0, 6), **PAD)

        # Emotion result
        self.result_var = tk.StringVar(value="—")
        tk.Label(root, textvariable=self.result_var,
                 font=("Courier", 26, "bold"),
                 bg="#1a1a1a", fg="#f0c040").pack(pady=(4, 0), **PAD)

        # Confidence
        self.conf_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.conf_var,
                 font=("Courier", 10), bg="#1a1a1a", fg="#888").pack(pady=(0, 14), **PAD)

        self._try_autoload()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _try_autoload(self):
        import os
        if os.path.exists(MODEL_PATH):
            self._do_load(MODEL_PATH)

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self._do_load(path)

    def _do_load(self, path: str):
        try:
            self.model, self.class_names = load_model(path)
            self.status_var.set(f"✓ Loaded  |  {', '.join(self.class_names)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.status_var.set("Model failed to load")

    def choose_and_predict(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a model first.")
            return
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        img   = Image.open(path).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((PREVIEW_PX, PREVIEW_PX))
        self.photo = ImageTk.PhotoImage(thumb)
        # Draw image centred on canvas
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW_PX // 2, PREVIEW_PX // 2,
                                 anchor="center", image=self.photo)

        label, confidence = predict(self.model, self.class_names, img)
        self.result_var.set(label.upper())
        self.conf_var.set(f"confidence: {confidence:.1f}%")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = FerApp(root)

    # Let Tkinter calculate the natural window size from its contents,
    # then centre that exact size on screen — no hardcoded height needed.
    root.update_idletasks()
    W = root.winfo_reqwidth()
    H = root.winfo_reqheight()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    root.mainloop()
