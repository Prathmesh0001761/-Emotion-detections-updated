import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
from keras.models import load_model
import time
import base64
from io import BytesIO
import soundfile as sf  # For better audio handling

# Function to encode image in base64
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS styling with animations and better UI
def apply_custom_css():
    background_image = "background.jpg"  # Ensure this file is in the same directory
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600&display=swap');

            .stApp {{
                background: url("data:image/png;base64,{get_base64(background_image)}") no-repeat center center fixed;
                background-size: cover;
                font-family: 'Poppins', sans-serif;
            }}

            .content-container {{
                background-color: rgba(16, 29, 44, 0.85);
                backdrop-filter: blur(8px);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: fadeIn 1s ease-out;
            }}

            h1 {{
                color: #e91e63;
                text-align: center;
                font-weight: 700;
                text-shadow: 0 2px 10px rgba(233, 30, 99, 0.3);
                margin-bottom: 5px;
                font-size: 3.5rem;
                font-family: 'Dancing Script', cursive;
                transition: color 0.3s ease, text-shadow 0.3s ease;
            }}

            h1:hover {{
                color: #ad1457;
                text-shadow: 0 3px 12px rgba(173, 20, 87, 0.5);
            }}

            .subheader {{
                color: #bdbdbd;
                text-align: center;
                font-size: 1.5rem;
                margin-bottom: 30px;
                transition: color 0.3s ease;
            }}

            .subheader:hover {{
                color: #9e9e9e;
            }}

            .stTabs {{
                background-color: transparent;
            }}

            [data-baseweb="tab-list"] {{
                gap: 10px;
            }}

            [data-baseweb="tab"] {{
                background-color: rgba(16, 29, 44, 0.7) !important;
                border-radius: 10px !important;
                padding: 12px 25px !important;
                transition: all 0.3s ease !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                font-size: 1.2rem;
            }}

            [data-baseweb="tab"]:hover {{
                background-color: rgba(233, 30, 99, 0.1) !important;
                transform: translateY(-2px);
            }}

            [aria-selected="true"] {{
                background-color: rgba(233, 30, 99, 0.2) !important;
                color: #e91e63 !important;
                font-weight: 600;
                border: 1px solid #e91e63 !important;
            }}

            .stFileUploader {{
                margin: 20px 0;
                border: 2px dashed rgba(233, 30, 99, 0.3) !important;
                border-radius: 10px !important;
                padding: 30px !important;
                transition: all 0.3s ease !important;
                background-color: rgba(0, 0, 0, 0.2) !important;
                font-size: 1.1rem;
            }}

            .stFileUploader:hover {{
                border-color: #e91e63 !important;
                box-shadow: 0 0 20px rgba(233, 30, 99, 0.2) !important;
                transform: scale(1.02);
            }}

            .stButton>button {{
                background: linear-gradient(45deg, #e91e63, #9c27b0) !important;
                color: white !important;
                border: none !important;
                border-radius: 25px !important;
                padding: 15px 35px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3) !important;
                width: 100%;
                font-size: 1.2rem;
            }}

            .stButton>button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(156, 39, 176, 0.4) !important;
            }}

            .audio-container {{
                margin: 20px 0;
                border-radius: 10px;
                overflow: hidden;
                background-color: rgba(0, 0, 0, 0.3);
                transition: opacity 0.5s ease;
            }}

            .result-box {{
                background: rgba(0, 0, 0, 0.4);
                border-radius: 10px;
                padding: 25px;
                margin: 20px 0;
                border-left: 4px solid #e91e63;
                animation: fadeIn 0.8s ease-out;
            }}

            .emotion-display {{
                font-size: 2.5rem;
                margin: 15px 0;
                text-align: center;
                transition: font-size 0.3s ease;
            }}

            .emotion-display:hover {{
                font-size: 2.7rem;
            }}

            .confidence {{
                color: #bdbdbd;
                text-align: center;
                font-size: 1.2rem;
                transition: color 0.3s ease;
            }}

            .confidence:hover {{
                color: #cfcfcf;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
                100% {{ transform: scale(1); }}
            }}

            .pulse {{
                animation: pulse 2s infinite;
            }}

            .file-info {{
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-size: 1rem;
            }}
        </style>
    """, unsafe_allow_html=True)

# Enhanced waveform plot with dark theme
def wave_plot(data, sampling_rate):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    
    # Updated librosa waveform display
    librosa.display.waveshow(y=data, sr=sampling_rate, color='#e91e63', ax=ax)
    
    ax.spines['bottom'].set_color('#e91e63')
    ax.spines['left'].set_color('#e91e63')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel("Time (s)", color='white', fontsize=12)
    ax.set_ylabel("Amplitude", color='white', fontsize=12)
    ax.set_title("Audio Waveform Analysis", fontweight="bold", color='white', fontsize=14, pad=20)
    ax.grid(color='#333333', alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Enhanced CNN model prediction with emoji visualization
def prediction(data, sampling_rate, file_name):
    emotion_dict = {
        0: "😐 Neutral",
        1: "😌 Calm",
        2: "😊 Happy",
        3: "😢 Sad",
        4: "😠 Angry",
        5: "😨 Fear",
        6: "🤢 Disgust",
        7: "😲 Surprise"
    }
    
    try:
        model = load_model("models/CnnModel.h5")
        # Updated MFCC extraction with better parameters
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40, 
                                           n_fft=2048, hop_length=512).T, axis=0)
        X_test = np.expand_dims([mfccs], axis=2)
        
        with st.spinner("🔍 Analyzing emotions with CNN..."):
            time.sleep(1.5)
            predict = model.predict(X_test, verbose=0)
        
        detected_emotion = emotion_dict[np.argmax(predict)]
        confidence = np.max(predict) * 100
        
        st.markdown(f"""
            <div class="file-info">
                <strong>File:</strong> {file_name}<br>
                <strong>Duration:</strong> {len(data)/sampling_rate:.2f} seconds<br>
                <strong>Sample Rate:</strong> {sampling_rate} Hz
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="result-box pulse">
                <h3 style="color: #e91e63; text-align: center; margin-bottom: 15px;">Emotion Detection Result</h3>
                <div class="emotion-display">{detected_emotion}</div>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in CNN prediction: {str(e)}")

# Enhanced MLP model prediction
def prediction_mlp(data, sampling_rate, file_name):
    emotion_dict = {
        0: "😐 Neutral",
        1: "😌 Calm",
        2: "😊 Happy",
        3: "😢 Sad",
        4: "😠 Angry",
        5: "😨 Fear",
        6: "🤢 Disgust",
        7: "😲 Surprise"
    }
    
    try:
        model = joblib.load("models/MLP_model.pkl")
        # Updated MFCC extraction with better parameters
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40,
                                           n_fft=2048, hop_length=512).T, axis=0)
        
        with st.spinner("🔍 Analyzing emotions with MLP..."):
            time.sleep(1.5)
            probabilities = model.predict_proba([mfccs])[0]
            detected_emotion = emotion_dict[model.predict([mfccs])[0]]
            confidence = max(probabilities) * 100
        
        st.markdown(f"""
            <div class="file-info">
                <strong>File:</strong> {file_name}<br>
                <strong>Duration:</strong> {len(data)/sampling_rate:.2f} seconds<br>
                <strong>Sample Rate:</strong> {sampling_rate} Hz
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="result-box pulse">
                <h3 style="color: #e91e63; text-align: center; margin-bottom: 15px;">Emotion Detection Result</h3>
                <div class="emotion-display">{detected_emotion}</div>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in MLP prediction: {str(e)}")

# Main app function with ultimate UI
def main():
    apply_custom_css()
    
    st.markdown("""
        <div class="content-container">
            <h1>🎤 SPEECH EMOTION CLASSIFIER</h1>
            <div class="subheader">Discover the emotions hidden in voice patterns</div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🧠 CNN Model", "🤖 MLP Model"])
    
    # Initialize session state variables
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
        st.session_state.audio_sr = None
        st.session_state.audio_name = None
    
    audio_file = st.file_uploader("Drag and drop your audio file here", 
                                type=['wav', 'mp3', 'ogg', 'flac'],  # Added more formats
                                help="Supported formats: WAV, MP3, OGG, FLAC | Max size: 200MB")
    
    if audio_file is not None:
        try:
            # Use soundfile for more reliable audio loading
            with BytesIO(audio_file.read()) as f:
                st.session_state.audio_data, st.session_state.audio_sr = librosa.load(f, sr=None)  # Keep original sample rate
            st.session_state.audio_name = audio_file.name
            
            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
            st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
    
    if st.session_state.audio_data is not None and st.session_state.audio_sr is not None:
        with tab1:
            with st.spinner("🔄 Processing audio file with CNN..."):
                try:
                    wave_plot(st.session_state.audio_data, st.session_state.audio_sr)
                    prediction(st.session_state.audio_data, st.session_state.audio_sr, st.session_state.audio_name)
                except Exception as e:
                    st.error(f"CNN processing error: {str(e)}")
        
        with tab2:
            with st.spinner("🔄 Processing audio file with MLP..."):
                try:
                    wave_plot(st.session_state.audio_data, st.session_state.audio_sr)
                    prediction_mlp(st.session_state.audio_data, st.session_state.audio_sr, st.session_state.audio_name)
                except Exception as e:
                    st.error(f"MLP processing error: {str(e)}")

if __name__ == '__main__':
    main()