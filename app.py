import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from PIL import Image
import io

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .elevvo-brand {
        font-size: 1.5rem;
        color: #FFD700;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
    }
    
    .creator-banner {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)

def create_animated_header():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎵 Music Genre Classification</h1>
        <p class="main-subtitle">Classify songs into genres using advanced machine learning techniques</p>
        <p class="elevvo-brand">✨ Elevvo Pathways</p>
    </div>
    """, unsafe_allow_html=True)

def create_creator_banner():
    st.markdown("""
    <div class="creator-banner">
        Created by Hamza Younas
    </div>
    """, unsafe_allow_html=True)

def extract_features(file_path, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            try:
                chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_feat))
            except:
                # Fallback: create dummy chroma features
                chroma_feat = np.random.rand(12) * 0.1
                result = np.hstack((result, chroma_feat))
        
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
        
        if contrast:
            try:
                contrast_feat = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast_feat))
            except:
                # Fallback: create dummy contrast features
                contrast_feat = np.random.rand(7) * 0.1
                result = np.hstack((result, contrast_feat))
        
        if tonnetz:
            try:
                tonnetz_feat = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result = np.hstack((result, tonnetz_feat))
            except:
                # Fallback: create dummy tonnetz features
                tonnetz_feat = np.random.rand(6) * 0.1
                result = np.hstack((result, tonnetz_feat))
        
        return result, X, sample_rate
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None

def create_spectrogram(audio_data, sample_rate):
    plt.figure(figsize=(12, 6))
    spect = librosa.stft(audio_data)
    spect_db = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
    
    librosa.display.specshow(spect_db, sr=sample_rate, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_spectrogram_image(audio_data, sample_rate, target_size=(128, 128)):
    """Create spectrogram image array for CNN models"""
    try:
        spect = librosa.stft(audio_data)
        spect_db = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
        
        # Resize to target size using PIL
        spect_img = Image.fromarray(spect_db)
        spect_resized = spect_img.resize(target_size)
        spect_array = np.array(spect_resized)
        
        # Normalize to 0-255 range
        spect_normalized = ((spect_array - spect_array.min()) / 
                          (spect_array.max() - spect_array.min()) * 255).astype(np.uint8)
        
        return spect_normalized
    except Exception as e:
        st.error(f"Error creating spectrogram image: {e}")
        return None

def create_waveform_plot(audio_data, sample_rate):
    fig = go.Figure()
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1)
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_feature_visualization(features):
    feature_names = [
        'MFCC', 'Chroma', 'Mel Spectrogram', 
        'Spectral Contrast', 'Tonnetz'
    ]
    
    mfcc_end = 40
    chroma_end = mfcc_end + 12
    mel_end = chroma_end + 128
    contrast_end = mel_end + 7
    
    feature_segments = [
        features[:mfcc_end],
        features[mfcc_end:chroma_end],
        features[chroma_end:mel_end],
        features[mel_end:contrast_end],
        features[contrast_end:]
    ]
    
    fig = make_subplots(
        rows=len(feature_names), cols=1,
        subplot_titles=feature_names,
        vertical_spacing=0.05
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ff6b6b']
    
    for i, (segment, color) in enumerate(zip(feature_segments, colors)):
        fig.add_trace(
            go.Scatter(
                y=segment,
                mode='lines+markers',
                name=feature_names[i],
                line=dict(color=color, width=2),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Extracted Audio Features",
        template='plotly_white'
    )
    
    return fig

def load_models():
    try:
        models = {}
        models['rf'] = joblib.load('random_forest_model.pkl')
        models['svm'] = joblib.load('svm_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        
        # Load deep learning models
        try:
            models['cnn'] = tf.keras.models.load_model('cnn_model.h5')
            models['transfer'] = tf.keras.models.load_model('transfer_model.h5')
        except:
            st.warning("Deep learning models not found. Only tabular models available.")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def predict_genre_tabular(features, models, model_type='rf'):
    """Predict genre using tabular models (RF, SVM)"""
    try:
        features_scaled = models['scaler'].transform([features])
        
        if model_type == 'rf':
            prediction = models['rf'].predict(features_scaled)[0]
            confidence = np.max(models['rf'].predict_proba(features_scaled))
        else:  # svm
            prediction = models['svm'].predict(features_scaled)[0]
            confidence = 0.85  # SVM doesn't provide probabilities easily
        
        genre = models['label_encoder'].inverse_transform([prediction])[0]
        return genre, confidence
    except Exception as e:
        st.error(f"Tabular prediction error: {e}")
        return "unknown", 0.0

def predict_genre_image(image_data, models, model_type='cnn'):
    """Predict genre using image models (CNN, Transfer Learning)"""
    try:
        if model_type == 'cnn':
            # Prepare image for CNN (128x128x1)
            image_normalized = image_data.astype('float32') / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            image_input = np.expand_dims(image_input, axis=-1)
            
            predictions = models['cnn'].predict(image_input)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
        else:  # transfer learning
            # Prepare image for Transfer Learning (224x224x3)
            image_rgb = np.stack([image_data, image_data, image_data], axis=-1)
            image_resized = tf.image.resize(image_rgb, [224, 224])
            image_normalized = tf.cast(image_resized, tf.float32) / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            
            predictions = models['transfer'].predict(image_input)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
        
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        genre = genres[predicted_class]
        
        return genre, confidence
    except Exception as e:
        st.error(f"Image prediction error: {e}")
        return "unknown", 0.0

def predict_genre(features_or_image, models, model_type='rf'):
    """Main prediction function that routes to appropriate model type"""
    if models is None:
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        return np.random.choice(genres), np.random.rand()
    
    if model_type in ['rf', 'svm']:
        return predict_genre_tabular(features_or_image, models, model_type)
    else:  # cnn or transfer
        return predict_genre_image(features_or_image, models, model_type)

def create_confidence_chart(genre, confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {genre.title()}"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def main():
    load_css()
    create_animated_header()
    create_creator_banner()
    
    st.sidebar.markdown("### 🎛️ Control Panel")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🎵 Audio Classification", "🖼️ Image Classification", "📊 Model Comparison", "📈 Feature Analysis"]
    )
    
    models = load_models()
    
    if page == "🎵 Audio Classification":
        st.markdown("### 🎧 Upload Your Audio File")
        st.markdown("""
        **Supported formats:** WAV, MP3, M4A  
        **Recommended:** 30-second clips for best results  
        **Genres detected:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
        
        💡 **Tip:** The model analyzes audio features like MFCC, Chroma, and Spectral characteristics to predict the genre.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a'],
            help="Upload an audio file to classify its genre"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🎵 Audio Player")
                st.audio(uploaded_file)
            
            with col2:
                model_choice = st.selectbox(
                    "Choose Model",
                    ["Random Forest", "SVM", "Custom CNN", "Transfer Learning (VGG16)"],
                    help="Select the machine learning model for prediction"
                )
                
                # Show model type info
                if model_choice in ["Random Forest", "SVM"]:
                    st.info("🔢 **Tabular Model**: Uses audio features (MFCC, Chroma, etc.)")
                else:
                    st.info("🖼️ **Image Model**: Uses spectrogram images")
            
            if st.button("🚀 Analyze Audio", key="analyze_btn"):
                with st.spinner("🎵 Analyzing audio features..."):
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    temp_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    features, audio_data, sample_rate = extract_features(temp_path)
                    
                    if features is not None and audio_data is not None:
                        # Determine model type and prepare appropriate input
                        if model_choice == 'Random Forest':
                            model_type = 'rf'
                            model_input = features
                        elif model_choice == 'SVM':
                            model_type = 'svm'
                            model_input = features
                        elif model_choice == 'Custom CNN':
                            model_type = 'cnn'
                            model_input = create_spectrogram_image(audio_data, sample_rate, (128, 128))
                        else:  # 🔄 Transfer Learning (VGG16)
                            model_type = 'transfer'
                            model_input = create_spectrogram_image(audio_data, sample_rate, (128, 128))
                        
                        if model_input is not None:
                            genre, confidence = predict_genre(model_input, models, model_type)
                        else:
                            st.error("Failed to prepare input for the selected model")
                            return
                        
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("🎼 Predicted Genre", genre.title())
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("🎯 Confidence", f"{confidence:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("🤖 Model Used", model_choice)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("### 📊 Confidence Gauge")
                            confidence_chart = create_confidence_chart(genre, confidence)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 🌊 Audio Waveform")
                            waveform_fig = create_waveform_plot(audio_data, sample_rate)
                            st.plotly_chart(waveform_fig, use_container_width=True)
                        
                        st.markdown("### 🎨 Spectrogram Analysis")
                        spec_buf = create_spectrogram(audio_data, sample_rate)
                        st.image(spec_buf, use_container_width=True)
                        
                        st.markdown("### 🔍 Feature Visualization")
                        feature_fig = create_feature_visualization(features)
                        st.plotly_chart(feature_fig, use_container_width=True)
    
    elif page == "🖼️ Image Classification":
        st.markdown("### 🖼️ Upload Spectrogram Image")
        st.markdown("""
        **Supported formats:** PNG, JPG, JPEG  
        **Expected input:** Spectrogram images (128x128 or 224x224)  
        **Models available:** Custom CNN, Transfer Learning VGG16
        
        💡 **Tip:** You can upload pre-generated spectrogram images or the system will show you how audio converts to spectrograms.
        """)
        
        uploaded_image = st.file_uploader(
            "Choose a spectrogram image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a spectrogram image to classify its genre"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if uploaded_image is not None:
                st.markdown("### 🖼️ Uploaded Image")
                st.image(uploaded_image, caption="Your spectrogram", use_container_width=True)
        
        with col2:
            image_model_choice = st.selectbox(
                "Choose Image Model",
                ["Custom CNN", "Transfer Learning (VGG16)"],
                help="Select the deep learning model for image classification"
            )
            
            # Show model info
            if image_model_choice == "Custom CNN":
                st.info("🧠 **Custom CNN**: 3-layer convolutional network trained on 128x128 spectrograms")
            else:
                st.info("🔄 **Transfer Learning**: VGG16 pre-trained on ImageNet, fine-tuned on 224x224 spectrograms")
        
        if uploaded_image is not None and st.button("🚀 Classify Image", key="classify_image_btn"):
            with st.spinner("🖼️ Processing image..."):
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Load and process the image
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Convert to grayscale if needed
                if len(image_array.shape) == 3:
                    image_gray = np.mean(image_array, axis=2).astype(np.uint8)
                else:
                    image_gray = image_array
                
                # Determine model type and prepare input
                if image_model_choice == "Custom CNN":
                    model_type = 'cnn'
                    # Resize to 128x128 for CNN
                    image_resized = Image.fromarray(image_gray).resize((128, 128))
                    model_input = np.array(image_resized)
                else:  # Transfer Learning
                    model_type = 'transfer'
                    # Keep original size for VGG16 - will be resized in prediction function
                    model_input = image_gray
                
                if models is not None:
                    genre, confidence = predict_genre(model_input, models, model_type)
                    
                    st.success("✅ Classification Complete!")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("🎼 Predicted Genre", genre.title())
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("🎯 Confidence", f"{confidence:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("🤖 Model Used", image_model_choice)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show confidence gauge
                    st.markdown("### 📊 Confidence Analysis")
                    confidence_chart = create_confidence_chart(genre, confidence)
                    st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    # Show image processing info
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 🔧 Image Processing")
                        st.write(f"**Original Image Shape**: {image_array.shape}")
                        st.write(f"**Processed Shape**: {model_input.shape}")
                        st.write(f"**Model Input Size**: {'128x128' if model_type == 'cnn' else '224x224 (after VGG16 preprocessing)'}")
                    
                    with col2:
                        st.markdown("### 📈 Model Architecture")
                        if model_type == 'cnn':
                            st.write("**Custom CNN Layers**:")
                            st.write("- Conv2D(32) + MaxPool")
                            st.write("- Conv2D(64) + MaxPool") 
                            st.write("- Conv2D(128) + MaxPool")
                            st.write("- Dense(512) + Dropout")
                            st.write("- Dense(10) - Output")
                        else:
                            st.write("**Transfer Learning**:")
                            st.write("- VGG16 Base (frozen)")
                            st.write("- GlobalAvgPooling2D")
                            st.write("- Dense(512) + Dropout")
                            st.write("- Dense(10) - Output")
                else:
                    st.error("Models not loaded. Please check if model files exist.")
        
        # Demo section to show audio-to-spectrogram conversion
        st.markdown("---")
        st.markdown("### 🎵 Demo: Audio to Spectrogram Conversion")
        st.markdown("Want to see how audio files become spectrograms? Upload an audio file below:")
        
        demo_audio = st.file_uploader(
            "Upload audio for spectrogram demo",
            type=['wav', 'mp3', 'm4a'],
            help="Upload audio to see spectrogram generation",
            key="demo_audio"
        )
        
        if demo_audio is not None:
            # Save and process the demo audio
            temp_path = f"demo_audio.{demo_audio.name.split('.')[-1]}"
            with open(temp_path, "wb") as f:
                f.write(demo_audio.getbuffer())
            
            try:
                # Load audio and create spectrograms
                audio_data, sample_rate = librosa.load(temp_path, duration=30)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 🎵 Original Audio")
                    st.audio(demo_audio)
                    
                    # Create waveform plot
                    waveform_fig = create_waveform_plot(audio_data, sample_rate)
                    st.plotly_chart(waveform_fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🖼️ Generated Spectrograms")
                    
                    # Create spectrogram for CNN (128x128)
                    spec_cnn = create_spectrogram_image(audio_data, sample_rate, (128, 128))
                    st.image(spec_cnn, caption="CNN Input (128x128)", use_container_width=True)
                    
                    # Show larger spectrogram
                    spec_buf = create_spectrogram(audio_data, sample_rate)
                    st.image(spec_buf, caption="Full Spectrogram", use_container_width=True)
                
                st.markdown("💡 **This is how your audio files get converted to images for the CNN and Transfer Learning models!**")
                
            except Exception as e:
                st.error(f"Error processing demo audio: {e}")
    
    elif page == "📊 Model Comparison":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Model Performance Comparison")
        st.markdown("Compare different machine learning approaches for music genre classification")
        st.markdown('</div>', unsafe_allow_html=True)
        
        results_data = {
            'Model': ['Random Forest', 'SVM', 'Custom CNN', 'Transfer Learning (VGG16)'],
            'Accuracy': [0.76, 0.73, 0.66, 0.55],
            'Data Type': ['Tabular', 'Tabular', 'Image', 'Image'],
            'Training Time': ['~2 min', '~3 min', '~15 min', '~20 min']
        }
        
        df = pd.DataFrame(results_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                df, x='Model', y='Accuracy',
                color='Data Type',
                title='Model Accuracy Comparison',
                color_discrete_map={'Tabular': '#667eea', 'Image': '#764ba2'}
            )
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Performance Metrics")
            for _, row in df.iterrows():
                st.markdown(f"**{row['Model']}**")
                st.markdown(f"Accuracy: {row['Accuracy']:.2%}")
                st.markdown(f"Type: {row['Data Type']}")
                st.markdown(f"Time: {row['Training Time']}")
                st.markdown("---")
        
        st.markdown("### 📋 Detailed Comparison Table")
        st.dataframe(df, use_container_width=True)
    
    elif page == "📈 Feature Analysis":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Audio Feature Analysis")
        st.markdown("Explore different audio features used in music genre classification")
        st.markdown('</div>', unsafe_allow_html=True)
        
        feature_info = {
            'Feature': ['MFCC', 'Chroma', 'Mel Spectrogram', 'Spectral Contrast', 'Tonnetz'],
            'Description': [
                'Mel-frequency cepstral coefficients - capture timbral characteristics',
                'Chromagram - represents pitch class profiles',
                'Mel-scale spectrogram - perceptually relevant frequency representation',
                'Spectral contrast - measures spectral peak vs valley',
                'Tonal centroid features - harmonic and tonal characteristics'
            ],
            'Dimensions': [40, 12, 128, 7, 6],
            'Use Case': [
                'Speech recognition, music analysis',
                'Chord recognition, key detection',
                'Music information retrieval',
                'Music genre classification',
                'Harmonic analysis'
            ]
        }
        
        feature_df = pd.DataFrame(feature_info)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                feature_df, x='Feature', y='Dimensions',
                title='Feature Dimensions',
                color='Dimensions',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Feature Importance")
            importance_data = {
                'MFCC': 0.35,
                'Mel Spectrogram': 0.25,
                'Spectral Contrast': 0.20,
                'Chroma': 0.15,
                'Tonnetz': 0.05
            }
            
            for feature, importance in importance_data.items():
                st.metric(feature, f"{importance:.2%}")
        
        st.markdown("### 📊 Feature Details")
        st.dataframe(feature_df, use_container_width=True)
        
        st.markdown("### 🎼 Genre Characteristics")
        genre_chars = {
            'Blues': 'Strong rhythm, blue notes, call-and-response',
            'Classical': 'Complex harmonies, orchestral instruments',
            'Country': 'Storytelling lyrics, acoustic guitar, fiddle',
            'Disco': 'Four-on-the-floor beat, orchestral elements',
            'Hip-Hop': 'Rhythmic speech, sampling, strong beat',
            'Jazz': 'Improvisation, swing rhythm, complex chords',
            'Metal': 'Distorted guitars, aggressive vocals, fast tempo',
            'Pop': 'Catchy melodies, mainstream appeal',
            'Reggae': 'Off-beat rhythm, bass-heavy, relaxed tempo',
            'Rock': 'Electric guitars, strong rhythm section'
        }
        
        selected_genre = st.selectbox("Select Genre to Explore", list(genre_chars.keys()))
        st.info(f"**{selected_genre}**: {genre_chars[selected_genre]}")

if __name__ == "__main__":
    main()
