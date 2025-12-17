# Multi-Organ Multi-Disease Prediction System

An AI-powered diagnostic system for automated disease classification from CT scans across multiple organs using deep learning.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## 🔍 Overview

This project implements a comprehensive deep learning framework for automated multi-organ, multi-disease classification using CT imaging. The system can detect and classify various diseases across three critical body organs:

- **Kidney**: Cyst, Stone, Tumor, Normal (5,600 images)
- **Chest**: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, COVID-19, Non-COVID-19, Normal (9,439 images)
- **Brain**: Aneurysm, Cancer, Hemorrhagic, Tumor, Normal (7,290 images)

The modular architecture employs organ-specific models to maximize diagnostic accuracy and minimize cross-organ confusion, making it suitable for real-world clinical deployment.

## ✨ Features

- **Multi-Organ Support**: Unified framework covering kidney, chest, and brain CT scans
- **High Accuracy**: Achieves 98.67% (kidney), 95.49% (chest), and 78.90% (brain) test accuracy
- **Organ-Specific Models**: Separate optimized models prevent disease confusion across organs
- **Advanced Preprocessing**: Custom pipelines for each organ type including CLAHE, noise reduction, and contrast enhancement
- **Interactive Interface**: Streamlit-based web application for easy inference
- **Modular Design**: Extensible architecture for adding new organs or disease classes
- **Clinical Ready**: Comprehensive evaluation with confusion matrices and performance metrics

## 🏗️ Architecture

### Model Selection
- **Kidney**: EfficientNet-B2 (selected after comparison with B0 for superior accuracy)
- **Chest**: EfficientNet-B0 (lightweight with excellent performance)
- **Brain**: EfficientNet-B0 (balanced efficiency and accuracy)

### Key Components
1. **Preprocessing Pipeline**: Organ-specific image enhancement and normalization
2. **Transfer Learning**: Pre-trained ImageNet weights for feature extraction
3. **Data Augmentation**: Rotation, flipping, translation, intensity variations
4. **Class Balancing**: Weighted loss functions to handle imbalanced datasets
5. **Regularization**: Dropout, early stopping, and learning rate scheduling

## 📊 Datasets

### Kidney CT Dataset
- **Size**: ~5,600 images (512×512 pixels, RGB)
- **Classes**: Cyst, Stone, Tumor, Normal
- **Format**: JPG
- **Preprocessing**: Grayscale conversion, CLAHE, intensity normalization

### Chest CT Dataset
- **Size**: 9,439 images (combined from two datasets)
  - Cancer dataset: 1,000 images
  - COVID-19 dataset: 8,439 images
- **Classes**: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, COVID-19, Non-COVID-19, Normal
- **Format**: PNG
- **Preprocessing**: Otsu thresholding, intensity windowing, bilateral filtering, unsharp masking

### Brain CT Dataset
- **Size**: ~7,290 images (512×512 pixels)
- **Classes**: Aneurysm, Cancer, Hemorrhagic, Tumor, Normal
- **Format**: JPG, DICOM
- **Preprocessing**: Intensity normalization, gamma correction, CLAHE, Gaussian blur

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-organ-disease-prediction.git
cd multi-organ-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
opencv-python>=4.6.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
pillow>=9.3.0
```

## 💻 Usage

### Training Models
```python
# Train kidney model
python train_kidney.py --epochs 50 --batch_size 32 --model efficientnet-b2

# Train chest model
python train_chest.py --epochs 50 --batch_size 32 --model efficientnet-b0

# Train brain model
python train_brain.py --epochs 50 --batch_size 32 --model efficientnet-b0
```

### Running the Streamlit Interface
```bash
streamlit run app.py
```

The interface will open in your browser where you can:
1. Select the organ type (Kidney/Chest/Brain)
2. Upload a CT scan image
3. View prediction results with confidence scores

### Programmatic Inference
```python
from predictor import MultiOrganPredictor

# Initialize predictor
predictor = MultiOrganPredictor()

# Load and predict
result = predictor.predict(
    image_path='path/to/ct_scan.jpg',
    organ='kidney'  # or 'chest', 'brain'
)

print(f"Predicted Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3 Predictions: {result['top3']}")
```

## 📈 Model Performance

### Kidney CT Classification

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cyst | 1.00 | 0.98 | 0.99 | 172 |
| Normal | 1.00 | 0.96 | 0.98 | 136 |
| Stone | 0.96 | 1.00 | 0.98 | 142 |
| Tumor | 0.98 | 1.00 | 0.99 | 150 |

**Overall Accuracy**: 98.67% (Test), 97.67% (Validation)

### Chest CT Classification

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Adenocarcinoma | 0.91 | 0.97 | 0.94 | 64 |
| COVID-19 Positive | 0.99 | 0.99 | 0.99 | 1499 |
| Large Cell Carcinoma | 1.00 | 0.89 | 0.94 | 38 |
| Lung Normal | 0.96 | 1.00 | 0.98 | 43 |
| Non-COVID-19 | 0.90 | 0.92 | 0.91 | 189 |
| Squamous Cell Carcinoma | 0.94 | 0.92 | 0.93 | 52 |

**Overall Accuracy**: 95.49% (Test), 97.3% (Validation)

### Brain CT Classification

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Aneurysm | 0.78 | 0.88 | 0.82 |
| Cancer | 0.79 | 0.87 | 0.83 |
| Hemorrhagic | 0.76 | 0.74 | 0.75 |
| Normal | 0.80 | 0.71 | 0.75 |
| Tumor | 0.83 | 0.88 | 0.85 |

**Overall Accuracy**: 78.90% (Test), 80.25% (Validation)

## 📁 Project Structure
```
multi-organ-disease-prediction/
│
├── data/
│   ├── kidney/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── chest/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── brain/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   ├── kidney_efficientnet_b2.h5
│   ├── chest_efficientnet_b0.h5
│   └── brain_efficientnet_b0.h5
│
├── preprocessing/
│   ├── kidney_preprocessor.py
│   ├── chest_preprocessor.py
│   └── brain_preprocessor.py
│
├── training/
│   ├── train_kidney.py
│   ├── train_chest.py
│   └── train_brain.py
│
├── evaluation/
│   ├── evaluate_models.py
│   └── visualize_results.py
│
├── predictor.py
├── app.py
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras, EfficientNet
- **Image Processing**: OpenCV, scikit-image
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Medical Imaging**: DICOM processing libraries

## 🎯 Results

The system demonstrates excellent performance across all three organs:

- **Kidney Model**: Exceptional accuracy (98.67%) with near-perfect classification for all disease types
- **Chest Model**: Strong performance (95.49%) with particularly high accuracy for COVID-19 detection (99%)
- **Brain Model**: Good baseline performance (78.90%) with room for improvement in distinguishing hemorrhagic and normal cases

Key achievements:
- Minimal overfitting across all models
- Robust generalization to unseen data
- Fast inference times suitable for clinical use
- Clear, interpretable probability outputs

## 🔮 Future Work

- [ ] Expand to additional organs (liver, pancreas, abdomen)
- [ ] Implement attention mechanisms for better explainability
- [ ] Add GradCAM visualizations for interpretable predictions
- [ ] Integrate with PACS systems for clinical deployment
- [ ] Develop ensemble models combining multiple architectures
- [ ] Create mobile application for point-of-care diagnostics
- [ ] Incorporate patient metadata for multi-modal predictions
- [ ] Add real-time inference optimization for edge devices

## 👥 Contributors

- **Kavya Goswami** - BML Munjal University
- **Vanshika Valecha** - BML Munjal University
- **Angelina Gupta** - BML Munjal University

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@article{goswami2025multiorgandisease,
  title={Multi-organ multi-disease prediction using a single framework},
  author={Goswami, Kavya and Valecha, Vanshika and Gupta, Angelina},
  year={2025},
  institution={BML Munjal University}
}
```

## 🤝 Acknowledgments

- BML Munjal University for providing computational resources
- Research papers cited in the literature review for methodological guidance
- Open-source medical imaging datasets used in training

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Email: [your-email@example.com]
- Institution: BML Munjal University, Haryana, India

---

**Note**: This is a research project intended for educational purposes. Clinical deployment requires appropriate validation and regulatory approval.
