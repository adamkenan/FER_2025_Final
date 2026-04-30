[README.md](https://github.com/user-attachments/files/27241391/README.md)
# Facial Emotion Recognition (FER) — Graduation Project
### Data Science & AI Diploma

A deep learning system that classifies **7 human emotions** from facial images using computer vision and transfer learning. Built as a graduation project with the long-term goal of powering more empathetic AI systems.

**Emotions classified:** Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

## 🧠 Project Overview

The core idea was to bridge a gap that most AI systems ignore entirely — human emotion. By training a model to recognise what someone is feeling from their face, the system can serve as the perception layer for emotionally intelligent applications like empathetic chatbots or adaptive interfaces.

The project went through **6 full iterations**, each with a different architecture and philosophy, progressively improving from ~25% to **72.8% validation accuracy**.

---

## 🏗️ Final Model Architecture (Attempt #6)

| Component | Detail |
|---|---|
| **Backbone** | ResNet-34 (pretrained on ImageNet, fine-tuned) |
| **Classifier Head** | Custom 3-layer MLP (Linear → BN → ReLU → Dropout) |
| **Attention** | Squeeze-and-Excitation (SE) block on backbone |
| **Loss Function** | Label Smoothing Cross-Entropy + class frequency weights |
| **Optimizer** | AdamW (lr=3e-4, weight_decay=2e-4) |
| **Scheduler** | OneCycleLR |
| **Augmentations** | Random flip, rotation, affine, color jitter, random erasing, MixUp, CutMix |
| **Evaluation** | Test-Time Augmentation (TTA) |
| **Input Size** | 224×224 RGB |
| **Dataset** | FER2013 (35,887 images, 7 classes) |

**Final Results:** 78.3% train accuracy · 72.8% validation accuracy

---

## 📁 Project Structure

```
├── FER_4th_attempt.ipynb   # Best performing model (Attempt #6 code)
├── FER_GRAD_Amit.pptx      # Full presentation covering all 6 attempts
└── README.md
```

---

## 🔁 Iteration History

| Attempt | Architecture | Val Accuracy | Key Issue |
|---|---|---|---|
| #1 | EfficientNet + RetinaFace | ~59% | Face detection ran every epoch |
| #2 | RetinaFace embeddings + MLP | ~25% | RetinaFace is expression-invariant |
| #3 | Custom CNN (48×48 grayscale) | ~61% | Baseline — info ceiling at low resolution |
| #4 | ResNet18 + custom MLP | ~68% | High overfitting (92.9% train acc) |
| #5 | ResNet34 + focal loss + unfreezing | ~64% | Silent loss double-accumulation bug |
| #6 | ResNet34 + MixUp/CutMix + SE attention | **72.8%** | ✅ Best result |

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch torchvision numpy pillow tqdm scikit-learn
```

### Dataset

Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle and extract it so the folder structure looks like:

```
dataset/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

### Training

Open `FER_4th_attempt.ipynb` and update the `data_root` path in the `CONFIG` dictionary to point to your dataset folder:

```python
CONFIG = {
    'data_root': '/your/path/to/dataset',
    ...
}
```

Then run all cells. The best model checkpoint will be saved automatically.

---

## 🧪 Key Learnings

- Pipeline inefficiencies compound — preprocessing should happen once, not every epoch
- Tool-task alignment matters — RetinaFace is built to *ignore* expressions, making it wrong for detecting them
- Always establish a simple baseline before adding complexity
- A large train/val gap is a pipeline signal, not just an overfitting flag
- Silent bugs in training loops (e.g. double-accumulating loss) can corrupt entire experiments
- Augmentation has a ceiling — too much destroys the semantic signal you're trying to learn
- Transfer learning only works when the source task aligns with the target task

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Python 3.x
