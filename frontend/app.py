"""
Streamlit frontend for Emotion Detection application.
"""
import streamlit as st
import sys
import os

# Add parent directory to path to import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_loader import EmotionDetector

# Page configuration
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .emotion-name {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    .stProgress > div > div > div {
        background-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    # Get the project root directory (parent of frontend)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "final_emotion_model")
    st.session_state.detector = EmotionDetector()

    with st.spinner("Loading emotion detection model... This may take a moment."):
        success = st.session_state.detector.load_model()
        if not success:
            st.error("Failed to load the model. Please check if the model files are in the correct location.")
            st.stop()

# Header
st.markdown('<h1 class="main-header">😊 Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter a sentence and discover the emotion it describes</p>', unsafe_allow_html=True)

# Input section
st.markdown("---")
text_input = st.text_area(
    "Enter your sentence:",
    height=100,
    placeholder="Type your sentence here...\n\nExample: 'I am so happy today!' or 'This makes me feel anxious.'",
    help="Enter any sentence and the model will predict the emotion it describes."
)

# Prediction button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Detect Emotion", type="primary", use_container_width=True)

# Display results
if predict_button:
    if text_input.strip():
        with st.spinner("Analyzing emotion..."):
            try:
                result = st.session_state.detector.predict(text_input)
                
                # Display main prediction
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                # Emotion box with gradient
                emotion_emoji = {
                    "joy": "😊",
                    "sadness": "😢",
                    "anger": "😠",
                    "fear": "😨",
                    "love": "❤️",
                    "surprise": "😲"
                }
                
                emoji = emotion_emoji.get(result["predicted_emotion"], "😊")
                
                st.markdown(f"""
                    <div class="emotion-box">
                        <div class="emotion-name">{emoji} {result["predicted_emotion"].upper()}</div>
                        <div class="confidence-score">Confidence: {result["confidence"]*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display confidence bar
                st.progress(result["confidence"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter a sentence to analyze.")

# Sidebar information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This Emotion Detector uses a fine-tuned DistilBERT model 
    to classify emotions in text.
    
    **Supported Emotions:**
    - 😊 Joy
    - 😢 Sadness
    - ❤️ Love
    - 😠 Anger
    - 😨 Fear
    - 😲 Surprise
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Enter complete sentences for better accuracy
    - The model analyzes the overall sentiment
    - Results show the predicted emotion with confidence score
    """)

