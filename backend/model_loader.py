"""
Backend module for loading the emotion detection model and making predictions.
Loads the model from Hugging Face Hub.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionDetector:
    """Class to handle emotion detection model loading and predictions."""

    def __init__(self, model_name="Kishorets/emotions-detector-distilbert"):
        """
        Initialize the emotion detector with the Hugging Face model name.

        Args:
            model_name (str): Hugging Face model repository ID
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.id2label = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }

    def load_model(self):
        """Load the model and tokenizer from Hugging Face Hub."""
        try:
            print(f"Loading model from Hugging Face: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully.")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, text: str):
        """
        Predict the emotion from input text.

        Args:
            text (str): Input sentence

        Returns:
            dict: Prediction result with confidence scores
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_id = torch.argmax(probabilities).item()

        emotion_scores = {
            self.id2label[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }

        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "predicted_emotion": self.id2label[predicted_id],
            "confidence": probabilities[predicted_id].item(),
            "all_emotions": emotion_scores,
            "sorted_emotions": sorted_emotions
        }
