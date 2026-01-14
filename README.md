# 😊 Emotion Detector

A web application built with Streamlit that detects emotions in text using a fine-tuned DistilBERT model. Simply enter a sentence and discover which emotion it describes!

## 🌐 Live Demo

**Try it now!** 👉 [https://emotions-detector.streamlit.app/](https://emotions-detector.streamlit.app/)

## 🎯 Features

- **Real-time Emotion Detection**: Analyze emotions in text instantly
- **Multiple Emotion Classes**: Detects 6 different emotions:
  - 😊 Joy
  - 😢 Sadness
  - ❤️ Love
  - 😠 Anger
  - 😨 Fear
  - 😲 Surprise
- **Confidence Scores**: View confidence level for the predicted emotion
- **Beautiful UI**: Modern and intuitive user interface

## 📁 Project Structure

```
emotions-detector/
│
├── frontend/
│   └── app.py                 # Streamlit frontend application
│
├── backend/
│   └── model_loader.py        # Model loading and prediction logic
│
├── models/                     # (Optional) Additional model files
│
├── final_emotion_model/        # Pre-trained model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd emotions-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   Option 1: Using the run script
   ```bash
   python run_app.py
   ```
   
   Option 2: Direct Streamlit command
   ```bash
   streamlit run frontend/app.py
   ```

4. **Open your browser**
   - The app will automatically open in your default browser
   - Or navigate to `http://localhost:8501`

## 💻 Usage

1. Enter a sentence in the text area
2. Click the "🔍 Detect Emotion" button
3. View the predicted emotion and confidence score

### Example Sentences

- "I am so happy today!" → 😊 Joy
- "This situation makes me anxious" → 😨 Fear
- "I love spending time with you" → ❤️ Love
- "I feel so disappointed" → 😢 Sadness

## 🧠 Model Information

- **Model Type**: DistilBERT (fine-tuned for sequence classification)
- **Task**: Single-label emotion classification
- **Classes**: 6 emotions (sadness, joy, love, anger, fear, surprise)
- **Model Location**: `final_emotion_model/`

## 📦 Dependencies

- `streamlit`: Web application framework
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `numpy`: Numerical operations

## 🔧 Configuration

The model configuration is stored in `final_emotion_model/config.json`. The application automatically loads the model from this directory.

## 📝 Notes

- The model files are already trained and stored in `final_emotion_model/`
- No training is required to run the application
- The model uses GPU if available, otherwise falls back to CPU

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Model based on [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/)

