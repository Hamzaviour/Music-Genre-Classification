# 🎵 Music Genre Classification System

A professional Streamlit application for music genre classification using advanced machine learning algorithms and deep learning models.

**Created by Hamza Younas | Elevvo Pathways**

![Music Classification](https://img.shields.io/badge/ML-Music%20Classification-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo

🌐 **[Try the Live Application](https://music-genre-classifier-by-hamza.streamlit.app/)**

## Features

### 🎵 Audio Classification
- Upload audio files (WAV, MP3, M4A) for automatic genre classification
- Real-time audio feature extraction and analysis
- Interactive waveform and spectrogram visualizations
- Support for all 4 trained models

### 🖼️ Image Classification  
- Direct spectrogram image upload and classification
- CNN and Transfer Learning model support
- Audio-to-spectrogram conversion demo
- Detailed model architecture visualization

### 📊 Model Comparison
- Performance comparison of Random Forest, SVM, Custom CNN, and Transfer Learning
- Accuracy metrics and training time analysis
- Tabular vs Image approach comparison

### 🔍 Feature Analysis
- Comprehensive audio feature exploration
- MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz analysis
- Genre characteristics breakdown
- Feature importance rankings

### 🎯 Advanced Predictions
- Multi-model prediction interface
- Confidence scoring and analysis
- Real-time processing with progress indicators
- Professional animated UI with gradient design

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**
   - Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
   - Extract and place in `gtzan_dataset/` folder
   - Dataset structure: 10 genres × 100 songs each


4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Models Used

### 🔢 Tabular Models
- **Random Forest**: 76% accuracy, 193 audio features
- **SVM**: 73% accuracy, RBF kernel with feature scaling

### 🖼️ Image Models  
- **Custom CNN**: 66% accuracy, 3-layer convolutional network
- **Transfer Learning (VGG16)**: 55% accuracy, pre-trained on ImageNet

## Technologies

- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning models
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Traditional machine learning algorithms
- **Plotly**: Interactive visualizations
- **OpenCV**: Image processing
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Statistical visualizations

## Application Structure

```
├── app.py                           # Main Streamlit application
├── music_genre_classification.ipynb # Jupyter analysis notebook
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── models/                         # Trained model files
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── cnn_model.h5
│   ├── transfer_model.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── gtzan_dataset/                  # GTZAN music dataset
    ├── genres_original/
    │   ├── blues/
    │   ├── classical/
    │   ├── country/
    │   └── ... (10 genres total)
    └── images_original/
```

## Usage

1. **Audio Classification**: Upload audio files and select from 4 different models
2. **Image Classification**: Upload spectrogram images or generate them from audio
3. **Model Comparison**: Analyze performance differences between approaches
4. **Feature Analysis**: Explore audio characteristics and genre patterns
5. **Interactive Demos**: See real-time audio-to-spectrogram conversion

## 📈 Performance Results

| Model | Accuracy | Data Type | Features | Training Time |
|-------|----------|-----------|----------|---------------|
| **Random Forest** | **76.0%** | Tabular | 193 audio features | ~2 min |
| **SVM** | **73.0%** | Tabular | 193 audio features | ~3 min |
| **Custom CNN** | **66.0%** | Image | 128×128 spectrograms | ~15 min |
| **Transfer Learning** | **55.0%** | Image | 224×224 spectrograms | ~20 min |

### Key Findings
- **Tabular models outperformed image models** on this dataset
- **Hand-crafted audio features** proved more effective than raw spectrograms
- **Random Forest achieved best overall performance** (76% accuracy)
- **Transfer learning** showed potential but needs larger datasets

## 🎼 Supported Genres

- **Blues**: Strong rhythm, blue notes, call-and-response
- **Classical**: Complex harmonies, orchestral instruments  
- **Country**: Storytelling lyrics, acoustic guitar, fiddle
- **Disco**: Four-on-the-floor beat, orchestral elements
- **Hip-Hop**: Rhythmic speech, sampling, strong beat
- **Jazz**: Improvisation, swing rhythm, complex chords
- **Metal**: Distorted guitars, aggressive vocals, fast tempo
- **Pop**: Catchy melodies, mainstream appeal
- **Reggae**: Off-beat rhythm, bass-heavy, relaxed tempo
- **Rock**: Electric guitars, strong rhythm section

## 📊 Screenshots

<img width="987" height="588" alt="image" src="https://github.com/user-attachments/assets/5be59cf6-0f96-4d29-9091-3f15c5c8c38d" />

## 🛠️ Technical Implementation

### Audio Feature Extraction
```python
# 193 features total:
- MFCC: 40 coefficients (timbral characteristics)
- Chroma: 12 coefficients (pitch class profiles) 
- Mel Spectrogram: 128 coefficients (perceptual frequency)
- Spectral Contrast: 7 coefficients (peak vs valley)
- Tonnetz: 6 coefficients (harmonic analysis)
```

### Deep Learning Architecture
```python
# Custom CNN
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → 
Conv2D(128) → MaxPool → Flatten → Dense(512) → 
Dropout(0.5) → Dense(10)

# Transfer Learning
VGG16(frozen) → GlobalAvgPool → Dense(512) → 
Dropout(0.5) → Dense(10)
```

## 🙏 Acknowledgments

- **GTZAN Dataset** creators for the comprehensive music collection
- **Streamlit team** for the amazing web framework
- **TensorFlow/Keras** for deep learning capabilities
- **Librosa** for professional audio processing tools

## 📧 Contact

**Hamza Younas**
- GitHub: [@HamzaYounas](https://github.com/Hamzaviour)
- LinkedIn: [Hamza Younas](https://linkedin.com/in/hamza-younas)
- Email: hamzavelous@gmail.com

---

🎵 **Built with ❤️ using Streamlit, TensorFlow, and Machine Learning | Elevvo Pathways**
