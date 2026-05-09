# 🏥 Multi-Organ Multi-Disease Detection System

An AI-powered medical image analysis system that detects diseases across **Brain**, **Chest**, and **Kidney** CT/MRI/X-Ray scans using deep learning (EfficientNet-B0).

---

## 📁 Project Structure

```
Image-Processing-Project/
│
├── Multi organ-Multi disease Project.ipynb   # Full training pipeline (Kaggle)
│
└── DIP interfacee/                           # Streamlit web interface
    ├── app.py                                # Main application
    └── models/                               # Trained model weights
        ├── best_brain_model.pth
        ├── best_chest_model.pth
        └── best_kidney_model.pth
```

---

## 🧠 Supported Organs & Diseases

| Organ  | Diseases Detected |
|--------|-------------------|
| Brain  | Aneurysm, Cancer, Hemorrhagic, Normal, Tumor |
| Chest  | Normal, COVID-19, Large Cell Carcinoma, Adenocarcinoma, Non-COVID, Squamous Cell Carcinoma |
| Kidney | Cyst, Normal, Stone, Tumor |

---

## 🚀 Running the Interface

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
cd "DIP interfacee"
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🏋️ Training

The full training pipeline is in `Multi organ-Multi disease Project.ipynb`, designed to run on **Kaggle** with GPU acceleration.

**Datasets used:**
- Chest CT Scan Images
- COVID-19 Lung CT Scans
- Brain Tumor / Hemorrhage datasets
- Kidney CT Images

---

## 🏗️ Model Architecture

- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Custom Head:** Dropout → Linear(512) → ReLU → BN → Linear(256) → ReLU → BN → Linear(num_classes)
- **Preprocessing:** Gaussian + Bilateral denoising, normalization, resize to 224×224

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only** and is not intended for clinical diagnosis.
